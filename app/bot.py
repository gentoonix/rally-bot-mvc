"""
Rally bot — two bot tokens, each capable of joining one VC.
Data is scoped per (guild_id, voice_channel_id).

Audio pipeline (zero-pop design):
  - Kokoro-82M ONNX (int8 quantized) for TTS synthesis
  - Every clip synthesized ONCE, then encoded to Opus frames at rest
  - Full joiner countdown sequences pre-rendered per group as a single
    Opus frame list — one play() call, encoder never restarted mid-cue
  - Single persistent QueuedPCMAudio source per VC; discord's encoder runs continuously
    between clips, feeding silence when idle so the encoder never resets
  - is_opus() = True — Discord receives pre-encoded packets, no runtime encoding

Voice: bf_emma (British English female)
Model: kokoro-v1.0.int8.onnx + voices-v1.0.bin
"""

import discord
from discord import app_commands
from discord.ext import commands
import os, asyncio, json, math, numpy as np
from kokoro_onnx import Kokoro
from dotenv import load_dotenv
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Script detection & name romanisation
# ---------------------------------------------------------------------------
# Keeps the container tiny: pypinyin (~2MB) for Chinese, pure lookup for
# Japanese kana, and a graceful fallback (skip name) for unsupported scripts.
# Arabic, Hebrew, Thai, Devanagari, etc. → "Group N start rally" with no name.

import unicodedata, re

try:
    from pypinyin import lazy_pinyin, Style as PinyinStyle
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False

# Hiragana/katakana → romaji lookup (Hepburn). Pure Python, no deps.
_KANA_ROM = {
    'あ':'a','い':'i','う':'u','え':'e','お':'o',
    'か':'ka','き':'ki','く':'ku','け':'ke','こ':'ko',
    'さ':'sa','し':'shi','す':'su','せ':'se','そ':'so',
    'た':'ta','ち':'chi','つ':'tsu','て':'te','と':'to',
    'な':'na','に':'ni','ぬ':'nu','ね':'ne','の':'no',
    'は':'ha','ひ':'hi','ふ':'fu','へ':'he','ほ':'ho',
    'ま':'ma','み':'mi','む':'mu','め':'me','も':'mo',
    'や':'ya','ゆ':'yu','よ':'yo',
    'ら':'ra','り':'ri','る':'ru','れ':'re','ろ':'ro',
    'わ':'wa','を':'o','ん':'n',
    'が':'ga','ぎ':'gi','ぐ':'gu','げ':'ge','ご':'go',
    'ざ':'za','じ':'ji','ず':'zu','ぜ':'ze','ぞ':'zo',
    'だ':'da','ぢ':'ji','づ':'zu','で':'de','ど':'do',
    'ば':'ba','び':'bi','ぶ':'bu','べ':'be','ぼ':'bo',
    'ぱ':'pa','ぴ':'pi','ぷ':'pu','ぺ':'pe','ぽ':'po',
    'きゃ':'kya','きゅ':'kyu','きょ':'kyo',
    'しゃ':'sha','しゅ':'shu','しょ':'sho',
    'ちゃ':'cha','ちゅ':'chu','ちょ':'cho',
    'にゃ':'nya','にゅ':'nyu','にょ':'nyo',
    'ひゃ':'hya','ひゅ':'hyu','ひょ':'hyo',
    'みゃ':'mya','みゅ':'myu','みょ':'myo',
    'りゃ':'rya','りゅ':'ryu','りょ':'ryo',
    'ぎゃ':'gya','ぎゅ':'gyu','ぎょ':'gyo',
    'じゃ':'ja', 'じゅ':'ju', 'じょ':'jo',
    'びゃ':'bya','びゅ':'byu','びょ':'byo',
    'ぴゃ':'pya','ぴゅ':'pyu','ぴょ':'pyo',
}
# Katakana offset: katakana = hiragana + 0x60
_KATA_ROM = {chr(ord(k) + 0x60): v for k, v in _KANA_ROM.items() if len(k) == 1}
_KANA_ROM.update(_KATA_ROM)


def _script_of(text: str) -> str:
    """Return dominant script: latin, chinese, japanese, arabic, hebrew, other."""
    scripts = {'latin': 0, 'chinese': 0, 'japanese': 0, 'arabic': 0, 'hebrew': 0}
    for ch in text:
        cp = ord(ch)
        if cp < 0x0250:
            scripts['latin'] += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            scripts['chinese'] += 1
        elif (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF) or (0xFF65 <= cp <= 0xFF9F):
            scripts['japanese'] += 1
        elif 0x0600 <= cp <= 0x06FF:
            scripts['arabic'] += 1
        elif 0x0590 <= cp <= 0x05FF:
            scripts['hebrew'] += 1
    # Japanese names often mix kana + kanji — if any kana present, treat as japanese
    if scripts['japanese'] > 0:
        return 'japanese'
    return max(scripts, key=scripts.get)


def romanise_name(name: str) -> str | None:
    """
    Convert a name to a Kokoro-friendly ASCII string.
    Returns None if the script is unsupported (caller should skip the name).
    Latin names pass through unchanged.
    """
    script = _script_of(name)

    if script == 'latin':
        return name  # already ASCII-compatible

    if script == 'chinese':
        if not _HAS_PYPINYIN:
            return None
        parts = lazy_pinyin(name, style=PinyinStyle.NORMAL)
        return ' '.join(p for p in parts if p.strip())

    if script == 'japanese':
        # Two-pass: digraphs first, then single kana
        result, i = [], 0
        while i < len(name):
            two = name[i:i+2]
            if two in _KANA_ROM:
                result.append(_KANA_ROM[two])
                i += 2
            elif name[i] in _KANA_ROM:
                result.append(_KANA_ROM[name[i]])
                i += 1
            elif 0x4E00 <= ord(name[i]) <= 0x9FFF:
                # Kanji in a Japanese name — try pypinyin as crude fallback
                if _HAS_PYPINYIN:
                    result.append(' '.join(lazy_pinyin(name[i])))
                i += 1
            else:
                result.append(name[i])   # pass ASCII/punctuation through
                i += 1
        return ''.join(result).strip() or None

    # Arabic, Hebrew, Thai, Devanagari, etc. — unsupported without large models
    return None


def tts_name_prompt(lead_name: str, grp_idx: int, grp_label: str | None = None) -> str:
    """
    Build the TTS prompt for '[Name] start rally'.
    Uses group label if set, otherwise lead name (romanised if needed).
    Falls back to 'Group N start rally' if name can't be romanised.
    """
    if grp_label and grp_label != f"Group {grp_idx + 1}":
        # Group has a custom name — romanise it for TTS
        rom = romanise_name(grp_label)
        return f"{rom} start rally" if rom else f"Group {grp_idx + 1} start rally"
    rom = romanise_name(lead_name)
    if rom:
        return f"{rom} start rally"
    return f"Group {grp_idx + 1} start rally"


load_dotenv()

TOKENS = [
    os.getenv("BOT_TOKEN_1"),
    os.getenv("BOT_TOKEN_2"),
]

DATA_FILE        = "rally_data.json"
MAX_MEMBERS      = 5
KOKORO_MODEL     = "/app/voice/kokoro-v1.0.int8.onnx"
KOKORO_VOICE_BIN = "/app/voice/voices-v1.0.bin"
KOKORO_VOICE     = "af_heart"
KOKORO_LANG      = "en-us"

TARGET_RATE  = 48000
CHANNELS     = 2
FRAME_MS     = 20
FRAME_BYTES  = TARGET_RATE * CHANNELS * 2 * FRAME_MS // 1000  # 3840

WORD_GAP_S   = 0.85   # seconds per word slot in merged countdown

# Root frequencies for each group — one octave lower for menace
GROUP_ROOT_FREQS = [220, 277, 311, 370, 415]   # A3 C#4 Eb4 F#4 Ab4

# All members in a group share the same root tone — only groups differ

def _tone_key(gid: int, vid: int, grp_idx: int, member_idx: int) -> str:
    return f"g{gid}_vc{vid}_tone_g{grp_idx}_m{member_idx}"

def _tone_freq(grp_idx: int) -> float:
    return GROUP_ROOT_FREQS[grp_idx % len(GROUP_ROOT_FREQS)]

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

vc_groups:          dict = {}
vc_panels:          dict = {}
active_views:       dict = {}
button_cooldowns:   dict = {}
bot_vc_assignment:  dict = {0: None, 1: None}
vc_text_channel:    dict = {}   # (gid, vid) → text channel id where panel lives
registered_players: dict = {}
player_current_vc:  dict = {}

# PCM cache: raw 16-bit 48kHz stereo bytes, played through discord's continuous encoder
pcm_cache:      dict[str, bytes] = {}
# Secondary raw cache used only for building joiner sequences
_raw_pcm_cache: dict[str, bytes] = {}

kokoro_inst = None
_current_grp_idx = 0   # set before run_member_countdowns so seq keys resolve


# ---------------------------------------------------------------------------
# PCM helpers
# ---------------------------------------------------------------------------

def float32_to_stereo_48k(samples: np.ndarray, source_rate: int) -> bytes:
    if source_rate != TARGET_RATE:
        new_len     = int(len(samples) * TARGET_RATE / source_rate)
        old_indices = np.linspace(0, len(samples) - 1, new_len)
        samples     = np.interp(old_indices, np.arange(len(samples)), samples)
    pcm    = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    stereo = np.empty(len(pcm) * 2, dtype=np.int16)
    stereo[0::2] = pcm
    stereo[1::2] = pcm
    return stereo.tobytes()


def apply_fade(pcm: bytes, fade_ms: float = 10.0) -> bytes:
    s           = np.frombuffer(pcm, dtype=np.int16).copy().astype(np.float32)
    fade_frames = int(TARGET_RATE * CHANNELS * fade_ms / 1000)
    fade_frames = min(fade_frames, len(s) // 4)
    if fade_frames > 0:
        s[:fade_frames]  *= np.linspace(0.0, 1.0, fade_frames)
        s[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames)
    return np.clip(s, -32768, 32767).astype(np.int16).tobytes()


def silence_pcm(ms: int) -> bytes:
    return bytes(TARGET_RATE * CHANNELS * 2 * ms // 1000)


def make_beep_pcm(freq: int, duration: float, volume: float = 1.0,
                   pulses: int = 1, gap_ms: int = 60) -> bytes:
    """
    Generate a menacing tone: overdrive + odd harmonics + hard transient attack.
    Envelope: instant crack, flat sustain, abrupt cut.
    """
    def one_pulse(f, dur, vol):
        n      = int(TARGET_RATE * dur)
        t      = np.arange(n) / TARGET_RATE
        att    = int(n * 0.02)    # 2% — hair-trigger attack
        sus    = int(n * 0.78)
        dec    = n - att - sus
        env    = np.concatenate([
            np.linspace(0.0, 1.0, att),
            np.ones(sus),
            np.linspace(1.0, 0.0, dec) ** 2,   # squared — abrupt close
        ])
        # Fundamental + odd harmonics for a hollow, ominous timbre
        wave   = 1.00 * np.sin(2 * math.pi * f * t)
        wave  += 0.60 * np.sin(2 * math.pi * f * 3 * t)   # 3rd — power
        wave  += 0.30 * np.sin(2 * math.pi * f * 5 * t)   # 5th — edge
        wave  += 0.15 * np.sin(2 * math.pi * f * 7 * t)   # 7th — grit
        wave  /= 2.05
        # Soft overdrive — clips peaks gently, adds harmonic dirt
        drive  = 2.5
        wave   = np.tanh(wave * drive) / np.tanh(drive)
        s      = np.clip(vol * env * wave * 32767, -32768, 32767).astype(np.int16)
        stereo = np.empty(len(s) * 2, dtype=np.int16)
        stereo[0::2] = s
        stereo[1::2] = s
        return stereo.tobytes()

    gap   = silence_pcm(gap_ms)
    parts = []
    for i in range(pulses):
        if i > 0:
            parts.append(gap)
        parts.append(one_pulse(freq, duration, volume))
    return b"".join(parts)


def make_foghorn_pcm(volume: float = 1.0) -> bytes:
    """
    Synthetic foghorn: deep 90Hz fundamental + strong odd harmonics (3rd, 5th, 7th)
    with slow tremolo and a resonant swell envelope. Unmistakably NOT a beep.
    Two blasts separated by a short silence.
    """
    def one_blast(dur: float) -> bytes:
        n    = int(TARGET_RATE * dur)
        t    = np.arange(n) / TARGET_RATE
        f0   = 90.0   # deep fundamental

        # Rich odd-harmonic stack — gives the brass/horn timbre
        wave  = 1.00 * np.sin(2 * math.pi * f0 * t)
        wave += 0.80 * np.sin(2 * math.pi * f0 * 3 * t)   # 3rd harmonic
        wave += 0.50 * np.sin(2 * math.pi * f0 * 5 * t)   # 5th
        wave += 0.25 * np.sin(2 * math.pi * f0 * 7 * t)   # 7th
        wave += 0.10 * np.sin(2 * math.pi * f0 * 9 * t)   # 9th — slight edge
        wave /= 2.65  # normalise sum

        # Slow tremolo at 6Hz — foghorn "waver"
        tremolo = 0.85 + 0.15 * np.sin(2 * math.pi * 6.0 * t)
        wave   *= tremolo

        # Envelope: fast swell in (8%), long sustain, tail out (15%)
        att = int(n * 0.08)
        dec = int(n * 0.15)
        sus = n - att - dec
        env = np.concatenate([
            np.linspace(0.0, 1.0, att) ** 0.5,   # sqrt curve — snappy open
            np.ones(sus),
            np.linspace(1.0, 0.0, dec) ** 2,     # squared — natural close
        ])
        wave *= env

        s      = np.clip(volume * wave * 32767, -32768, 32767).astype(np.int16)
        stereo = np.empty(len(s) * 2, dtype=np.int16)
        stereo[0::2] = s
        stereo[1::2] = s
        return stereo.tobytes()

    blast1 = one_blast(0.9)
    gap    = silence_pcm(220)   # short pause between blasts
    blast2 = one_blast(0.65)    # second blast slightly shorter
    return _frame_align(blast1 + gap + blast2)


def tts_pcm(text: str) -> bytes:
    samples, sr = kokoro_inst.create(text, voice=KOKORO_VOICE, speed=1.0, lang=KOKORO_LANG)
    return apply_fade(float32_to_stereo_48k(samples, sr))


def pad_pcm(pcm: bytes, target_s: float) -> bytes:
    target = int(TARGET_RATE * CHANNELS * 2 * target_s)
    if len(pcm) < target:
        return pcm + bytes(target - len(pcm))
    return pcm[:target]


def store(key: str, pcm: bytes):
    """Store PCM in both caches (raw for sequence building, pcm_cache for playback)."""
    _raw_pcm_cache[key] = pcm
    pcm_cache[key]      = pcm


def clip_dur(key: str) -> float:
    return len(pcm_cache.get(key, b"")) / (TARGET_RATE * CHANNELS * 2)

# ---------------------------------------------------------------------------
# Joiner sequence builder
# ---------------------------------------------------------------------------

def _min_gap_separation(members: list) -> int:
    """
    Return the minimum separation (seconds) between any two consecutive
    joiner fire times within a group.
    All joiners fire at t0 + their gap (gap = lead_time - march_time).
    If two gaps differ by < 5s their voice countdowns will overlap.
    Returns 999 if there is only one joiner (no overlap possible).
    """
    if len(members) <= 2:
        return 999   # only one joiner, no overlap
    lead_time = members[0][1]
    gaps      = sorted(lead_time - m[1] for m in members[1:])
    min_sep   = min(gaps[i+1] - gaps[i] for i in range(len(gaps) - 1))
    return min_sep


def _frame_align(pcm: bytes) -> bytes:
    """Pad PCM to a multiple of FRAME_BYTES so boundaries never split a frame."""
    rem = len(pcm) % FRAME_BYTES
    return pcm + bytes(FRAME_BYTES - rem) if rem else pcm


def _crossfade_join(a: bytes, b: bytes, fade_ms: float = 8.0) -> bytes:
    """
    Overlap the tail of `a` with the head of `b` using a linear crossfade.
    Eliminates clicks at hard PCM boundaries caused by waveform discontinuities.
    """
    fade_samples = int(TARGET_RATE * CHANNELS * fade_ms / 1000)
    fade_samples -= fade_samples % 2          # keep stereo pairs intact
    if fade_samples <= 0 or len(a) < fade_samples * 2 or len(b) < fade_samples * 2:
        return a + b                          # too short to crossfade safely
    a_arr    = np.frombuffer(a, dtype=np.int16).astype(np.float32)
    b_arr    = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in  = np.linspace(0.0, 1.0, fade_samples)
    a_tail   = a_arr[-fade_samples:] * fade_out
    b_head   = b_arr[:fade_samples]  * fade_in
    mixed    = np.clip(a_tail + b_head, -32768, 32767).astype(np.int16)
    return a[:-fade_samples * 2] + mixed.tobytes() + b[fade_samples * 2:]


def _build_joiner_seq_pcm(gap: int, gid: int = 0, vid: int = 0,
                           grp_idx: int = 0, j_idx: int = 1,
                           beep_only: bool = False) -> bytes:
    """
    Build full countdown PCM for a joiner with `gap` seconds before lead fires.

    beep_only=True  → frame-aligned silence + tone (no voice)
    beep_only=False → silence + "5" + silence + "3" + "2" + "1" + tone
                      all boundaries crossfaded to eliminate clicks
    """
    tone_key: str   = _tone_key(gid, vid, grp_idx, j_idx)
    tone_pcm: bytes = _raw_pcm_cache.get(tone_key, b"")

    def get(w):
        return _raw_pcm_cache.get(w, b"")

    def silence_aligned(seconds: float) -> bytes:
        ms = max(0, int(seconds * 1000))
        return _frame_align(silence_pcm(ms))

    if beep_only:
        return _frame_align(silence_aligned(gap) + tone_pcm)

    # Build each segment, then crossfade-join them in order
    parts: list[bytes] = []

    if gap >= 5:
        parts.append(silence_aligned(gap - 5))   # wait until "5" cue
        parts.append(get("5"))
        parts.append(silence_aligned(2.0))        # 2s gap before countdown
    else:
        parts.append(silence_aligned(max(0, gap - 3)))

    if gap >= 3:
        parts.append(get("3"))
        parts.append(silence_aligned(WORD_GAP_S - clip_dur("3")))
        parts.append(get("2"))
        parts.append(silence_aligned(WORD_GAP_S - clip_dur("2")))
        parts.append(get("1"))
        parts.append(silence_aligned(0.05))       # tiny breath before tone
    
    parts.append(tone_pcm)

    # Filter empty segments, then crossfade-join sequentially
    parts = [p for p in parts if p]
    if not parts:
        return b""
    result = parts[0]
    for nxt in parts[1:]:
        result = _crossfade_join(result, nxt, fade_ms=6.0)
    return _frame_align(result)

# ---------------------------------------------------------------------------
# Cache population
# ---------------------------------------------------------------------------

def generate_static_audio():
    global kokoro_inst
    print(f"Loading Kokoro: {KOKORO_MODEL}")
    kokoro_inst = Kokoro(KOKORO_MODEL, KOKORO_VOICE_BIN)
    print(f"  voice={KOKORO_VOICE}  lang={KOKORO_LANG}")

    # Cross-VC alert tone (neutral, used when firing into another channel)
    store("beep_cross", make_foghorn_pcm())
    print(f"  foghorn (cross-VC): {clip_dur('beep_cross'):.3f}s")

    store("5", tts_pcm("5"))
    print(f"  tts '5': {clip_dur('5'):.3f}s")

    # Pre-generate "3", "2", "1" individually for use in joiner sequences
    for word in ("3", "2", "1"):
        store(word, tts_pcm(word))
        print(f"  tts '{word}': {clip_dur(word):.3f}s")

    print("Static audio ready.")


def _grp_key(gid, vid, idx, suffix):
    return f"g{gid}_vc{vid}_grp{idx}{suffix}"


def _seq_key(gid, vid, grp_idx, joiner_idx):
    return f"g{gid}_vc{vid}_grp{grp_idx}_j{joiner_idx}_seq"


def _ensure_group_tones(gid: int, vid: int, grp_idx: int, num_members: int):
    """Pre-generate tones for every member slot in a group — same freq, lead gets 2 pulses."""
    freq = _tone_freq(grp_idx)
    for m_idx in range(num_members):
        tk = _tone_key(gid, vid, grp_idx, m_idx)
        if tk not in pcm_cache:
            store(tk, make_beep_pcm(freq, 0.35, pulses=2))
            print(f"  tone {tk}: {freq:.0f}Hz x2 → {clip_dur(tk):.3f}s")


def generate_group_audio(gid: int, vid: int, groups: list):
    """Synthesise all audio for every group — TTS prompts, member tones, joiner sequences."""
    for grp_idx, group in enumerate(groups):
        members              = group["members"]
        lead_name, lead_time = members[0]

        # Per-group per-member tones (keyed globally by grp_idx, not per-VC)
        _ensure_group_tones(gid, vid, grp_idx, len(members))

        grp_label    = group.get("name") or f"Group {grp_idx + 1}"
        getready_txt = f"{grp_label} get ready"
        for suffix, text in [
            ("_getready",   getready_txt),
            ("_startrally", tts_name_prompt(lead_name, grp_idx, grp_label)),
        ]:
            k = _grp_key(gid, vid, grp_idx, suffix)
            if k not in pcm_cache:
                print(f"  Synth: {k}")
                store(k, tts_pcm(text))

        # Use beep-only mode if any two joiner fire times are closer than 5s —
        # voice countdowns would overlap and become unintelligible
        VOICE_THRESHOLD = 5   # seconds minimum separation between joiner tones
        beep_only = _min_gap_separation(members) < VOICE_THRESHOLD
        if beep_only:
            print(f"  Group {grp_idx+1}: beep-only mode (min gap separation < {VOICE_THRESHOLD}s)")

        for j_idx, (_, march_time) in enumerate(members[1:], start=1):
            sk = _seq_key(gid, vid, grp_idx, j_idx)
            if sk not in pcm_cache:
                gap = lead_time - march_time
                print(f"  Pre-render seq: gap={gap}s beep_only={beep_only} → {sk}")
                pcm = _build_joiner_seq_pcm(gap, gid, vid, grp_idx, j_idx, beep_only=beep_only)
                store(sk, pcm)
                print(f"    {clip_dur(sk):.2f}s ({int(clip_dur(sk)*50)} frames)")


def invalidate_group_audio(gid: int, vid: int):
    prefix = f"g{gid}_vc{vid}_"
    for k in [k for k in pcm_cache if k.startswith(prefix)]:
        del pcm_cache[k]
    for k in [k for k in _raw_pcm_cache if k.startswith(prefix)]:
        del _raw_pcm_cache[k]

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_data():
    global vc_groups, vc_panels, registered_players
    if not os.path.exists(DATA_FILE):
        print("No saved data.")
        return
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)
        for k, v in data.get("vc_groups", {}).items():
            gid, vid = map(int, k.split(","))
            vc_groups[(gid, vid)] = v
            generate_group_audio(gid, vid, v)
        for k, v in data.get("vc_panels", {}).items():
            gid, vid = map(int, k.split(","))
            vc_panels[(gid, vid)] = (
                v if isinstance(v, dict) else {"msg_id": v, "channel_id": None})
        registered_players = {
            int(u): n for u, n in data.get("registered_players", {}).items()}
        print(f"Loaded. {len(registered_players)} player(s).")
    except Exception as e:
        print(f"Load error: {e}")


def save_data():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump({
                "vc_groups":          {f"{g},{v}": gs for (g, v), gs in vc_groups.items()},
                "vc_panels":          {f"{g},{v}": p  for (g, v), p  in vc_panels.items()},
                "registered_players": {str(u): n for u, n in registered_players.items()},
            }, f, indent=2)
    except Exception as e:
        print(f"Save error: {e}")

# ---------------------------------------------------------------------------
# Audio playback — raw PCM queued through discord's single continuous encoder
# ---------------------------------------------------------------------------
#
# WHY PCM NOT PRE-ENCODED OPUS:
# Opus is stateful — the encoder builds pitch/LPC models across frames.
# Pre-encoding clips in separate sessions produces frames with discontinuous
# internal state. The decoder sees the discontinuity and produces a pop.
# The fix: keep everything as raw PCM, let discord.py's one encoder instance
# process a continuous stream. Silence between clips = zero PCM bytes.
# One vc.play() call per connection, never restarted.

import threading

_vc_sources:      dict[int, "QueuedPCMAudio"] = {}
_vc_started:      dict[int, bool]             = {}
_repost_locks:    dict[int, bool]             = {}   # channel_id → reposting in progress
_rally_tasks:     dict                        = {}   # (gid, vid, grp_idx) → asyncio.Task


class QueuedPCMAudio(discord.AudioSource):
    """
    Persistent raw-PCM AudioSource per VC.
    read() always returns exactly FRAME_BYTES of PCM (silence or audio).
    discord.py's internal Opus encoder processes one unbroken stream →
    no encoder state resets → no pops between clips.
    vc.play() is called exactly once per connection.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop          = loop
        self._queue: list   = []   # (pcm_bytes, done_event)
        self._cur_pcm       = b""
        self._cur_pos       = 0
        self._cur_event     = None
        self._lock          = threading.Lock()
        self._SILENCE       = bytes(FRAME_BYTES)   # 20ms of zeros

    def add(self, pcm: bytes, done_event: asyncio.Event):
        # Pad to frame boundary so read() never straddles a clip
        rem = len(pcm) % FRAME_BYTES
        if rem:
            pcm = pcm + bytes(FRAME_BYTES - rem)
        with self._lock:
            self._queue.append((pcm, done_event))

    def read(self) -> bytes:
        """Called every 20 ms by discord's audio thread. Never returns b''."""
        with self._lock:
            # Advance past exhausted clip
            while self._cur_pos >= len(self._cur_pcm):
                if self._cur_event is not None:
                    ev = self._cur_event
                    self._cur_event = None
                    self._loop.call_soon_threadsafe(ev.set)
                if self._queue:
                    self._cur_pcm, self._cur_event = self._queue.pop(0)
                    self._cur_pos = 0
                else:
                    return self._SILENCE

            chunk         = self._cur_pcm[self._cur_pos:self._cur_pos + FRAME_BYTES]
            self._cur_pos += FRAME_BYTES
            return chunk

    def is_opus(self) -> bool:
        return False   # discord.py encodes this PCM stream continuously

    def clear(self):
        """Flush all queued clips and signal any pending done events."""
        with self._lock:
            for _, ev in self._queue:
                self._loop.call_soon_threadsafe(ev.set)
            self._queue.clear()
            self._cur_pcm   = b""
            self._cur_pos   = 0
            if self._cur_event is not None:
                self._loop.call_soon_threadsafe(self._cur_event.set)
                self._cur_event = None

    def cleanup(self):
        pass


def _ensure_source(vc: discord.VoiceClient) -> "QueuedPCMAudio":
    """Return the persistent QueuedPCMAudio for this VC, starting it if needed."""
    cid = vc.channel.id
    if cid not in _vc_sources:
        _vc_sources[cid] = QueuedPCMAudio(asyncio.get_event_loop())

    src = _vc_sources[cid]

    if not _vc_started.get(cid):
        def _after(err):
            if err:
                print(f"[audio:{cid}] error: {err}")
            _vc_started.pop(cid, None)
        vc.play(src, after=_after)
        _vc_started[cid] = True

    return src


async def play_clip(vc: discord.VoiceClient, key: str):
    pcm = pcm_cache.get(key)
    if not pcm:
        print(f"WARNING: not in pcm_cache: {key}")
        return
    done = asyncio.Event()
    _ensure_source(vc).add(pcm, done)
    await done.wait()

# ---------------------------------------------------------------------------
# Countdown
# ---------------------------------------------------------------------------

async def run_member_countdowns(vc, members, t0, cross_clients=None):
    """
    Play all joiner sequences as ONE mixed PCM clip.

    Every joiner sequence has its own silence prefix baked in (timed from t0),
    so mixing all sequences into a single clip is the correct superposition —
    each joiner's countdown fires at exactly the right moment relative to t0,
    regardless of whether march times differ.

    Queueing them separately is wrong: the queue is serial, so clip 2 would
    only start after clip 1 finishes, adding its full duration as extra delay.
    """
    global _current_grp_idx
    gid     = vc.guild.id
    vid     = vc.channel.id
    grp_idx = _current_grp_idx
    loop    = asyncio.get_event_loop()

    if len(members) <= 1:
        if cross_clients:
            wait = t0 - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            await asyncio.gather(*[play_clip(c, "beep_cross") for c in cross_clients])
        return

    # Collect all joiner sequence PCM clips
    clips = []
    for j_idx in range(1, len(members)):
        sk  = _seq_key(gid, vid, grp_idx, j_idx)
        pcm = pcm_cache.get(sk, b"")
        if pcm:
            clips.append(pcm)

    if clips:
        # Mix all clips into one — pad shorter clips to match the longest
        max_len = max(len(c) for c in clips)
        rem = max_len % FRAME_BYTES
        if rem:
            max_len += FRAME_BYTES - rem
        if len(clips) == 1:
            mixed = clips[0] + bytes(max_len - len(clips[0]))
        else:
            arrays = []
            for c in clips:
                padded = c + bytes(max_len - len(c))
                arrays.append(np.frombuffer(padded, dtype=np.int16).astype(np.float32))
            mixed = np.clip(np.mean(arrays, axis=0), -32768, 32767).astype(np.int16).tobytes()

        wait = t0 - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)

        done = asyncio.Event()
        _ensure_source(vc).add(mixed, done)
        await done.wait()

    # Cross-VC alert fires in parallel (separate VC queue, unaffected)
    cross_task = None
    if cross_clients:
        cross_task = asyncio.create_task(
            asyncio.gather(*[play_clip(c, "beep_cross") for c in cross_clients]))

    if cross_task:
        await cross_task


def find_cross_vc_clients(group, initiating_vid, all_bots):
    cross, seen = [], set()
    for name, _ in group["members"]:
        for uid, reg_name in registered_players.items():
            if reg_name and reg_name.lower() == name.lower():
                cur = player_current_vc.get(uid)
                if cur and cur != initiating_vid and cur not in seen:
                    for b in all_bots:
                        for vc in b.voice_clients:
                            if vc.channel.id == cur:
                                cross.append(vc)
                                seen.add(cur)
    return cross

# ---------------------------------------------------------------------------
# Panel embed
# ---------------------------------------------------------------------------

def _group_label(group: dict) -> str:
    """Display name for a group — custom name if set, otherwise lead member name."""
    return group.get("name") or group["members"][0][0]


def build_panel_embed(groups, vc_name=""):
    embed = discord.Embed(
        title=f"Rally Groups — #{vc_name}" if vc_name else "Rally Groups",
        color=discord.Color.blue())
    embed.description = (
        "**Start X** — normal (get ready prompt).\n"
        "**⚡ X** — instant.\n"
        "**+** — add member (max 5). 10s cooldown.\n\n**Groups:**\n")
    for idx, g in enumerate(groups):
        label = _group_label(g)
        name_tag = f" *({label})*" if g.get("name") else ""
        parts = [f"**{n}** ({t}s)" + (" *(lead)*" if i == 0 else "")
                 for i, (n, t) in enumerate(g["members"])]
        embed.description += f"**{idx + 1}**{name_tag}: " + " + ".join(parts) + "\n"
    return embed

# ---------------------------------------------------------------------------
# Modals
# ---------------------------------------------------------------------------

class AddGroupModal(discord.ui.Modal, title="Add Rally Group"):
    grp_name = discord.ui.TextInput(label="Group Name (optional)", placeholder="e.g., Alpha, Bravo…",
                                    required=False, max_length=32)
    m1n = discord.ui.TextInput(label="Member 1 Name", placeholder="e.g., Y")
    m1t = discord.ui.TextInput(label="Member 1 March Time (s)", placeholder="e.g., 40",
                               style=discord.TextStyle.short)
    m2n = discord.ui.TextInput(label="Member 2 Name (optional)", placeholder="e.g., X",
                               required=False)
    m2t = discord.ui.TextInput(label="Member 2 March Time (optional)", placeholder="e.g., 35",
                               required=False, style=discord.TextStyle.short)

    def __init__(self, all_bots_ref):
        super().__init__()
        self._bots = all_bots_ref

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.user.voice or not interaction.user.voice.channel:
            return await interaction.response.send_message("Join a VC first.", ephemeral=True)
        vc       = interaction.user.voice.channel
        gid, vid = interaction.guild_id, vc.id
        key      = (gid, vid)
        if not bot_for_vc(gid, vid, self._bots):
            return await interaction.response.send_message("Use /join_voice first.", ephemeral=True)
        try:
            members = [(self.m1n.value.strip(), int(self.m1t.value.strip()))]
            if self.m2n.value.strip() and self.m2t.value.strip():
                members.append((self.m2n.value.strip(), int(self.m2t.value.strip())))
        except ValueError:
            return await interaction.response.send_message("Times must be integers.", ephemeral=True)
        members.sort(key=lambda x: x[1], reverse=True)
        grp = {"members": members}
        if self.grp_name.value.strip():
            grp["name"] = self.grp_name.value.strip()
        vc_groups.setdefault(key, []).append(grp)
        # Defer immediately — TTS synthesis blocks the event loop and Discord times out at 3s
        await interaction.response.defer(ephemeral=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_group_audio, gid, vid, vc_groups[key])
        save_data()
        summary = " + ".join(f"**{n}** ({t}s)" + (" *(lead)*" if i == 0 else "")
                              for i, (n, t) in enumerate(members))
        await interaction.followup.send(
            f"Added to **{vc.name}**: {summary}", ephemeral=True)

        # Post or refresh panel after every group add
        groups   = vc_groups.get(key, [])
        assigned = bot_for_vc(gid, vid, self._bots)
        if assigned:
            # Delete old panel message if present
            if key in vc_panels:
                pm = vc_panels[key]
                for b in self._bots:
                    ch = b.get_channel(pm.get("channel_id"))
                    if ch:
                        try:
                            old_msg = await ch.fetch_message(pm["msg_id"])
                            await old_msg.delete()
                        except (discord.NotFound, discord.HTTPException):
                            pass
                        break
            active_views.pop(key, None)
            vc_panels.pop(key, None)
            # Post fresh panel
            view              = RallyStartView(groups, gid, vid, assigned, self._bots)
            active_views[key] = view
            embed             = build_panel_embed(groups, vc.name)
            msg               = await interaction.channel.send(embed=embed, view=view)
            view._message     = msg
            vc_panels[key]       = {"msg_id": msg.id, "channel_id": interaction.channel.id}
            vc_text_channel[key] = interaction.channel.id
            save_data()


class AddMemberModal(discord.ui.Modal, title="Add Member"):
    name = discord.ui.TextInput(label="Name", placeholder="e.g., Z")
    time = discord.ui.TextInput(label="March Time (s)", placeholder="e.g., 30",
                                style=discord.TextStyle.short)

    def __init__(self, gid, vid, idx, all_bots_ref):
        super().__init__()
        self.gid, self.vid, self.idx, self._bots = gid, vid, idx, all_bots_ref

    async def on_submit(self, interaction: discord.Interaction):
        key    = (self.gid, self.vid)
        groups = vc_groups.get(key, [])
        if self.idx >= len(groups):
            return await interaction.response.send_message("Group gone.", ephemeral=True)
        group  = groups[self.idx]
        if len(group["members"]) >= MAX_MEMBERS:
            return await interaction.response.send_message(f"Cap is {MAX_MEMBERS}.", ephemeral=True)
        try:
            new_m = (self.name.value.strip(), int(self.time.value.strip()))
        except ValueError:
            return await interaction.response.send_message("Time must be integer.", ephemeral=True)
        group["members"].append(new_m)
        group["members"].sort(key=lambda x: x[1], reverse=True)
        invalidate_group_audio(self.gid, self.vid)
        # Defer immediately — TTS synthesis blocks the event loop and Discord times out at 3s
        await interaction.response.defer(ephemeral=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_group_audio, self.gid, self.vid, groups)
        save_data()
        # Delete old panel and repost fresh — same as add_group
        assigned = bot_for_vc(self.gid, self.vid, self._bots)
        if assigned and key in vc_panels:
            pm = vc_panels[key]
            for b in self._bots:
                ch = b.get_channel(pm.get("channel_id"))
                if ch:
                    try:
                        old_msg = await ch.fetch_message(pm["msg_id"])
                        await old_msg.delete()
                    except (discord.NotFound, discord.HTTPException):
                        pass
                    break
            active_views.pop(key, None)
            vc_panels.pop(key, None)
            vc_ch   = interaction.guild.get_channel(self.vid)
            vc_name = vc_ch.name if vc_ch else str(self.vid)
            view              = RallyStartView(groups, self.gid, self.vid, assigned, self._bots)
            active_views[key] = view
            msg               = await interaction.channel.send(
                embed=build_panel_embed(groups, vc_name), view=view)
            view._message     = msg
            vc_panels[key]    = {"msg_id": msg.id, "channel_id": interaction.channel.id}
            vc_text_channel[key] = interaction.channel.id
            save_data()
        summary = " + ".join(f"**{n}** ({t}s)" + (" *(lead)*" if i == 0 else "")
                              for i, (n, t) in enumerate(group["members"]))
        await interaction.followup.send(
            f"Group {self.idx + 1} updated: {summary}", ephemeral=True)

# ---------------------------------------------------------------------------
# Rally view
# ---------------------------------------------------------------------------

class RallyStartView(discord.ui.View):
    def __init__(self, groups, gid, vid, bot_ref, all_bots_ref):
        super().__init__(timeout=None)
        self.groups, self.gid, self.vid = groups, gid, vid
        self.bot_ref, self._all_bots    = bot_ref, all_bots_ref
        self._ticker_task = self._message = None
        self.update_buttons()

    def _get_vc(self):
        for vc in self.bot_ref.voice_clients:
            if vc.channel.id == self.vid:
                return vc
        return None

    def update_buttons(self):
        self.clear_items()
        now = datetime.utcnow()
        cd  = button_cooldowns.get((self.gid, self.vid), {})
        for idx, group in enumerate(self.groups):
            label   = _group_label(group)
            exp     = cd.get(idx)
            on_cd   = bool(exp and now < exp)
            rem     = int((exp - now).total_seconds()) if on_cd else 0
            at_cap  = len(group["members"]) >= MAX_MEMBERS
            rallying = (self.gid, self.vid, idx) in _rally_tasks and                        not _rally_tasks[(self.gid, self.vid, idx)].done()

            n = discord.ui.Button(
                label=f"Start {label}" if not on_cd else f"Start {label} ({rem}s)",
                style=discord.ButtonStyle.primary if not on_cd else discord.ButtonStyle.secondary,
                disabled=on_cd or rallying, row=idx)
            n.callback = self._make_cb(idx, False)
            self.add_item(n)

            i = discord.ui.Button(
                label=f"⚡ {label}" if not on_cd else f"⚡ {label} ({rem}s)",
                style=discord.ButtonStyle.success if not on_cd else discord.ButtonStyle.secondary,
                disabled=on_cd or rallying, row=idx)
            i.callback = self._make_cb(idx, True)
            self.add_item(i)

            if rallying:
                c = discord.ui.Button(
                    label=f"✕ Cancel {label}",
                    style=discord.ButtonStyle.danger, row=idx)
                c.callback = self._make_cancel_cb(idx)
                self.add_item(c)
            else:
                a = discord.ui.Button(
                    label="+" if not at_cap else "Full",
                    style=discord.ButtonStyle.secondary, disabled=at_cap, row=idx)
                a.callback = self._make_add_cb(idx)
                self.add_item(a)

    def _any_cd(self):
        now = datetime.utcnow()
        return any(e and now < e
                   for e in button_cooldowns.get((self.gid, self.vid), {}).values())

    async def _ticker(self):
        try:
            while self._any_cd():
                await asyncio.sleep(1)
                if not self._message:
                    break
                self.update_buttons()
                try:
                    await self._message.edit(view=self)
                except discord.NotFound:
                    break
            if self._message:
                self.update_buttons()
                try:
                    await self._message.edit(view=self)
                except discord.NotFound:
                    pass
        except asyncio.CancelledError:
            pass

    def start_ticker(self, msg):
        self._message = msg
        if self._ticker_task and not self._ticker_task.done():
            self._ticker_task.cancel()
        self._ticker_task = asyncio.create_task(self._ticker())

    def _make_add_cb(self, idx):
        async def cb(interaction: discord.Interaction):
            await interaction.response.send_modal(
                AddMemberModal(self.gid, self.vid, idx, self._all_bots))
        return cb

    def _make_cancel_cb(self, group_idx):
        async def cb(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            task_key = (self.gid, self.vid, group_idx)
            task = _rally_tasks.get(task_key)
            if task and not task.done():
                task.cancel()
                # Flush any queued audio immediately
                vc = self._get_vc()
                if vc:
                    src = _vc_sources.get(vc.channel.id)
                    if src:
                        src.clear()
            _rally_tasks.pop(task_key, None)
            # Clear cooldown so buttons re-enable right away
            button_cooldowns.get((self.gid, self.vid), {}).pop(group_idx, None)
            self.update_buttons()
            if self._message:
                try:
                    await self._message.edit(view=self)
                except discord.NotFound:
                    pass
            label = _group_label(self.groups[group_idx])
            await interaction.followup.send(
                f"Rally cancelled: **{label}**", ephemeral=True)
        return cb

    def _make_cb(self, group_idx, instant):
        async def cb(interaction: discord.Interaction):
            global _current_grp_idx
            await interaction.response.defer(ephemeral=True)
            vc = self._get_vc()
            if not vc or not vc.is_connected():
                return await interaction.followup.send(
                    "Bot isn't in this VC. Use /join_voice.", ephemeral=True)

            key      = (self.gid, self.vid)
            task_key = (self.gid, self.vid, group_idx)
            button_cooldowns.setdefault(key, {})[group_idx] = (
                datetime.utcnow() + timedelta(seconds=10))

            group   = self.groups[group_idx]
            members = group["members"]
            cross   = find_cross_vc_clients(group, self.vid, self._all_bots)

            _current_grp_idx = group_idx
            lead_cue = _tone_key(self.gid, self.vid, group_idx, 0)
            loop     = asyncio.get_event_loop()

            async def run_rally():
                try:
                    if instant:
                        t0    = loop.time()
                        coros = [play_clip(vc, lead_cue)]
                        if cross:
                            coros.append(play_clip(cross[0], "beep_cross"))
                        await asyncio.gather(*coros, run_member_countdowns(vc, members, t0, cross))
                    else:
                        await play_clip(vc, _grp_key(self.gid, self.vid, group_idx, "_getready"))
                        await asyncio.sleep(3)
                        await play_clip(vc, _grp_key(self.gid, self.vid, group_idx, "_startrally"))
                        t0    = loop.time()
                        coros = [play_clip(vc, lead_cue)]
                        if cross:
                            coros.append(play_clip(cross[0], "beep_cross"))
                        await asyncio.gather(*coros, run_member_countdowns(vc, members, t0, cross))
                except asyncio.CancelledError:
                    pass
                finally:
                    _rally_tasks.pop(task_key, None)
                    self.update_buttons()
                    if self._message:
                        try:
                            await self._message.edit(view=self)
                        except discord.NotFound:
                            pass

            task = asyncio.create_task(run_rally())
            _rally_tasks[task_key] = task

            # Show cancel button immediately
            self.update_buttons()
            await interaction.message.edit(view=self)
            self.start_ticker(interaction.message)

            label = _group_label(group)
            if instant:
                await interaction.followup.send(
                    f"⚡ **{label}** fired.", ephemeral=True)
            else:
                await interaction.followup.send(
                    f"**{label}** started.", ephemeral=True)
        return cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bot_for_vc(gid, vid, bots):
    for i, b in enumerate(bots):
        if bot_vc_assignment.get(i) == (gid, vid):
            return b
    return None


def free_bot(bots):
    for i, b in enumerate(bots):
        if bot_vc_assignment.get(i) is None:
            return i, b
    return None, None


async def resolve_user_vc(interaction, check_channel=True):
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
        return None
    vc  = interaction.user.voice.channel
    key = (interaction.guild_id, vc.id)
    if key not in vc_groups:
        await interaction.response.send_message(
            "No rally data for your VC. Use /join_voice first.", ephemeral=True)
        return None
    if check_channel:
        expected_cid = vc_text_channel.get(key)
        if expected_cid and interaction.channel_id != expected_cid:
            ch   = interaction.guild.get_channel(expected_cid)
            name = f"<#{expected_cid}>" if ch else "this VC's text channel"
            await interaction.response.send_message(
                f"This VC's panel is in {name} — run commands there.", ephemeral=True)
            return None
    return interaction.guild_id, vc.id, vc


async def _bump_panel(channel: discord.TextChannel):
    """Wait 5s then send a '.' and immediately delete it, pushing the panel above any slash command UI."""
    await asyncio.sleep(5)
    try:
        dot = await channel.send(".")
        await dot.delete()
    except (discord.HTTPException, discord.Forbidden):
        pass


async def repost_panel(gid, vid, bot_ref):
    key = (gid, vid)
    if key not in active_views or key not in vc_panels:
        return
    view = active_views[key]
    pm   = vc_panels[key]
    if not pm.get("channel_id"):
        return
    ch = bot_ref.get_channel(pm["channel_id"])
    if not ch:
        return

    # Debounce — ignore if a repost is already in progress for this channel
    cid = ch.id
    if _repost_locks.get(cid):
        return
    _repost_locks[cid] = True

    try:
        try:
            old = await ch.fetch_message(pm["msg_id"])
            await old.delete()
        except (discord.NotFound, discord.HTTPException):
            pass
        vc_ch = bot_ref.get_channel(vid)
        view.update_buttons()
        new = await ch.send(
            embed=build_panel_embed(view.groups, vc_ch.name if vc_ch else str(vid)),
            view=view)
        view._message  = new
        vc_panels[key] = {"msg_id": new.id, "channel_id": ch.id}
        save_data()
    finally:
        _repost_locks.pop(cid, None)

# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------

async def _panel_refresh(gid, vid, groups, interaction_guild, bots):
    """Module-level panel refresh — safe to call from modal on_submit handlers."""
    key = (gid, vid)
    if key not in active_views or key not in vc_panels:
        return False
    pm      = vc_panels[key]
    vc_ch   = interaction_guild.get_channel(vid)
    vc_name = vc_ch.name if vc_ch else str(vid)
    for b in bots:
        ch = b.get_channel(pm.get("channel_id"))
        if ch:
            try:
                msg = await ch.fetch_message(pm["msg_id"])
                v   = active_views[key]
                v.groups = groups
                v.update_buttons()
                await msg.edit(embed=build_panel_embed(groups, vc_name), view=v)
                return True
            except Exception as e:
                print(f"[panel] _panel_refresh failed: {e}")
                return False
    return False


def make_bot(bot_index: int, all_bots_ref: list) -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members          = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        print(f"[Bot {bot_index}] Ready: {bot.user}")
        if bot_index == 0:
            try:
                synced = await bot.tree.sync()
                print(f"[Bot 0] Synced {len(synced)} commands.")
            except Exception as e:
                print(f"[Bot 0] Sync error: {e}")
        else:
            # Wipe any stale commands registered under Bot 2's application ID
            try:
                bot.tree.clear_commands(guild=None)
                await bot.tree.sync()
                print(f"[Bot {bot_index}] Cleared stale commands.")
            except Exception as e:
                print(f"[Bot {bot_index}] Clear error: {e}")

    @bot.event
    async def on_voice_state_update(member, before, after):
        if member.id == bot.user.id:
            if before.channel and not after.channel:
                # Bot left / was disconnected
                a = bot_vc_assignment.get(bot_index)
                if a:
                    print(f"[Bot {bot_index}] Disconnected — clearing {a}")
                    bot_vc_assignment[bot_index] = None
                    vc_text_channel.pop(a, None)
                cid = before.channel.id
                _vc_sources.pop(cid, None)
                _vc_started.pop(cid, None)
            elif not before.channel and after.channel:
                # Bot joined (or rejoined) a VC — invalidate stale audio source
                # so the next play() call starts fresh without popping
                cid = after.channel.id
                _vc_sources.pop(cid, None)
                _vc_started.pop(cid, None)
                # Notify panel channel that the panel needs a reset
                gid = after.channel.guild.id
                vid = after.channel.id
                key = (gid, vid)
                if key in vc_panels:
                    pm = vc_panels[key]
                    for b in all_bots_ref:
                        ch = b.get_channel(pm.get("channel_id"))
                        if ch:
                            try:
                                await ch.send(
                                    "⚠️ Bot reconnected — old panel buttons are dead. "
                                    "Use `/reset_panel` to post a fresh one.",
                                    delete_after=30)
                            except Exception:
                                pass
                            break
            return
        if member.id in registered_players:
            if after.channel:
                player_current_vc[member.id] = after.channel.id
            else:
                player_current_vc.pop(member.id, None)

    if bot_index != 0:
        return bot

    @bot.event
    async def on_message(message: discord.Message):
        if message.author.bot or not message.guild:
            return
        for (g, v), pm in list(vc_panels.items()):
            if g != message.guild.id or not isinstance(pm, dict):
                continue
            if message.channel.id == pm.get("channel_id"):
                await repost_panel(g, v, bot)
        await bot.process_commands(message)

    @bot.tree.command(name="register",
                      description="Link your account for cross-VC tracking")
    @app_commands.describe(in_game_name="Your rally name — must match group entries exactly")
    async def register(interaction: discord.Interaction, in_game_name: str):
        uid = interaction.user.id
        registered_players[uid] = in_game_name.strip()
        if interaction.user.voice and interaction.user.voice.channel:
            player_current_vc[uid] = interaction.user.voice.channel.id
        save_data()
        await interaction.response.send_message(
            f"Registered as **{in_game_name.strip()}**.", ephemeral=True)

    @bot.tree.command(name="unregister", description="Unlink your account")
    async def unregister(interaction: discord.Interaction):
        uid = interaction.user.id
        if uid in registered_players:
            del registered_players[uid]
            player_current_vc.pop(uid, None)
            save_data()
            await interaction.response.send_message("Unregistered.", ephemeral=True)
        else:
            await interaction.response.send_message("You aren't registered.", ephemeral=True)

    async def _do_join(interaction: discord.Interaction):
        """Join the user's current VC with the first available bot."""
        try:
            if not interaction.user.voice or not interaction.user.voice.channel:
                return await interaction.response.send_message("Join a VC first.", ephemeral=True)
            vc_ch    = interaction.user.voice.channel
            gid, vid = interaction.guild_id, vc_ch.id
            key      = (gid, vid)

            if bot_for_vc(gid, vid, all_bots_ref):
                return await interaction.response.send_message(
                    f"A bot is already in **{vc_ch.name}**.", ephemeral=True)

            idx, target_bot = free_bot(all_bots_ref)
            if target_bot is None:
                return await interaction.response.send_message(
                    "Both bots are busy. Use /leave_voice to free one first.", ephemeral=True)

            # Defer before any async work — must happen within 3s of interaction
            await interaction.response.defer(ephemeral=True)

            target_guild = target_bot.get_guild(gid)
            if target_guild is None:
                return await interaction.followup.send(
                    "Bot isn't in this server yet — add it via OAuth2 invite.", ephemeral=True)

            # Disconnect any stale connection this bot has in this guild
            for existing_vc in list(target_bot.voice_clients):
                if existing_vc.guild.id == gid:
                    await existing_vc.disconnect(force=True)

            target_vc_ch = target_guild.get_channel(vid)
            if target_vc_ch is None:
                return await interaction.followup.send("Can't find that channel.", ephemeral=True)

            try:
                await asyncio.wait_for(
                    target_vc_ch.connect(timeout=15, reconnect=True),
                    timeout=20)
            except asyncio.TimeoutError:
                return await interaction.followup.send(
                    "Timed out connecting to VC — try again.", ephemeral=True)
            except discord.ClientException as e:
                return await interaction.followup.send(
                    f"Already connected: {e}", ephemeral=True)

            bot_vc_assignment[idx] = key
            vc_groups.setdefault(key, [])
            save_data()
            await interaction.followup.send(f"Joined **{vc_ch.name}**.", ephemeral=True)

        except Exception as e:
            print(f"[join_voice] unhandled error: {e}")
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message(
                        f"Something went wrong: {e}", ephemeral=True)
                else:
                    await interaction.followup.send(
                        f"Something went wrong: {e}", ephemeral=True)
            except Exception:
                pass

    async def _do_leave(interaction: discord.Interaction):
        """Remove the bot from the user's current VC."""
        if not interaction.user.voice or not interaction.user.voice.channel:
            return await interaction.response.send_message("Join a VC first.", ephemeral=True)
        vid, gid = interaction.user.voice.channel.id, interaction.guild_id
        assigned = bot_for_vc(gid, vid, all_bots_ref)
        if not assigned:
            return await interaction.response.send_message(
                "No bot in your VC.", ephemeral=True)

        await interaction.response.defer(ephemeral=True)
        for vc in list(assigned.voice_clients):
            if vc.channel.id == vid:
                await vc.disconnect(force=True)
                break
        for i, b in enumerate(all_bots_ref):
            if b is assigned:
                bot_vc_assignment[i] = None
                _vc_sources.pop(vid, None)
                _vc_started.pop(vid, None)
                vc_text_channel.pop((gid, vid), None)
                break
        await interaction.followup.send("Left.", ephemeral=True)

    @bot.tree.command(name="join_voice",
                      description="Put a bot in your current voice channel")
    async def join_voice(interaction: discord.Interaction):
        await _do_join(interaction)

    @bot.tree.command(name="leave_voice",
                      description="Remove the bot from your current voice channel")
    async def leave_voice(interaction: discord.Interaction):
        await _do_leave(interaction)

    @bot.tree.command(name="rebuild_commands",
                      description="Force re-sync slash commands with Discord (removes stale ones)")
    async def rebuild_commands(interaction: discord.Interaction):
        if not interaction.user.guild_permissions.administrator:
            return await interaction.response.send_message(
                "Admin only.", ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        try:
            # Sync without clearing — the in-memory tree already has all commands.
            # clear_commands() would wipe them from memory before sync, pushing empty list.
            synced = await bot.tree.sync()
            await interaction.followup.send(
                f"Done — {len(synced)} command(s) registered. "
                "Discord may take up to 1 min to reflect changes.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"Sync failed: {e}", ephemeral=True)

    @bot.tree.command(name="status",
                      description="Show which bots are connected and where")
    async def status(interaction: discord.Interaction):
        lines = []
        for i, b in enumerate(all_bots_ref):
            key = bot_vc_assignment.get(i)
            name = b.user.display_name if b.user else f"Bot {i+1}"
            if key:
                gid2, vid2 = key
                ch        = b.get_channel(vid2)
                ch_name   = ch.name if ch else str(vid2)
                grp_count = len(vc_groups.get(key, []))
                lines.append(f"**{name}** → #{ch_name} ({grp_count} group(s))")
            else:
                lines.append(f"**{name}** → *not in any VC*")
        embed = discord.Embed(title="Bot Status", color=discord.Color.blue(),
                              description="\n".join(lines))
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @bot.tree.command(name="add_group",
                      description="Add a rally group to your current voice channel")
    async def add_group(interaction: discord.Interaction):
        await interaction.response.send_modal(AddGroupModal(all_bots_ref))

    @bot.tree.command(name="list_groups",
                      description="List groups with options to remove members or whole groups")
    async def list_groups(interaction: discord.Interaction):
        r = await resolve_user_vc(interaction)
        if not r:
            return
        gid, vid, vc_ch = r
        groups = vc_groups.get((gid, vid), [])
        if not groups:
            return await interaction.response.send_message("No groups yet.", ephemeral=True)

        def make_embed():
            embed = discord.Embed(title=f"Groups — #{vc_ch.name}", color=discord.Color.green())
            embed.description = ""
            for idx, g in enumerate(groups):
                label = _group_label(g)
                name_tag = f" *({label})*" if g.get("name") else ""
                parts = [f"**{n}** ({t}s)" + (" *(lead)*" if i == 0 else "")
                         for i, (n, t) in enumerate(g["members"])]
                embed.description += f"**{idx + 1}**{name_tag}: " + " + ".join(parts) + "\n"
            embed.set_footer(text="Use the dropdowns below to remove members or whole groups.")
            return embed

        class GroupManageView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=60)
                self._build()

            def _build(self):
                self.clear_items()
                # Remove Member select — one option per member across all groups
                member_opts = []
                for g_idx, g in enumerate(groups):
                    for m_idx, (name, t) in enumerate(g["members"]):
                        label_g = _group_label(g)
                        member_opts.append(discord.SelectOption(
                            label=f"{name} ({t}s)",
                            description=f"Group {g_idx+1}: {label_g}",
                            value=f"m:{g_idx}:{m_idx}"))
                if member_opts:
                    rm_member = discord.ui.Select(
                        placeholder="Remove a member…",
                        options=member_opts[:25],   # Discord cap
                        row=0)
                    rm_member.callback = self._rm_member_cb
                    self.add_item(rm_member)

                # Remove Group select
                group_opts = [
                    discord.SelectOption(
                        label=f"Group {g_idx+1}: {_group_label(g)}",
                        description=" + ".join(m[0] for m in g["members"]),
                        value=f"g:{g_idx}")
                    for g_idx, g in enumerate(groups)
                ]
                if group_opts:
                    rm_group = discord.ui.Select(
                        placeholder="Remove a whole group…",
                        options=group_opts[:25],
                        row=1)
                    rm_group.callback = self._rm_group_cb
                    self.add_item(rm_group)

            async def _refresh(self, interaction: discord.Interaction):
                self._build()
                if groups:
                    await interaction.edit_original_response(embed=make_embed(), view=self)
                else:
                    await interaction.edit_original_response(
                        content="All groups removed.", embed=None, view=None)

            async def _rm_member_cb(self, interaction: discord.Interaction):
                await interaction.response.defer()
                _, g_idx_s, m_idx_s = interaction.data["values"][0].split(":")
                g_idx, m_idx = int(g_idx_s), int(m_idx_s)
                if g_idx >= len(groups):
                    return await interaction.followup.send("Group no longer exists.", ephemeral=True)
                group   = groups[g_idx]
                members = group["members"]
                if m_idx >= len(members):
                    return await interaction.followup.send("Member no longer exists.", ephemeral=True)
                removed_name = members[m_idx][0]
                if len(members) == 1:
                    return await interaction.followup.send(
                        f"Can't remove the only member — remove the whole group instead.", ephemeral=True)
                members.pop(m_idx)
                members.sort(key=lambda x: x[1], reverse=True)
                invalidate_group_audio(gid, vid)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, generate_group_audio, gid, vid, groups)
                save_data()
                await _update_panel_after_edit(gid, vid, groups, interaction.guild)
                await self._refresh(interaction)
                await interaction.followup.send(
                    f"Removed **{removed_name}** from group {g_idx+1}.", ephemeral=True)

            async def _rm_group_cb(self, interaction: discord.Interaction):
                await interaction.response.defer()
                _, g_idx_s = interaction.data["values"][0].split(":")
                g_idx = int(g_idx_s)
                if g_idx >= len(groups):
                    return await interaction.followup.send("Group no longer exists.", ephemeral=True)
                removed = groups.pop(g_idx)
                button_cooldowns.get((gid, vid), {}).pop(g_idx, None)
                invalidate_group_audio(gid, vid)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, generate_group_audio, gid, vid, groups)
                save_data()
                await _update_panel_after_edit(gid, vid, groups, interaction.guild)
                await self._refresh(interaction)
                await interaction.followup.send(
                    f"Removed group: **{_group_label(removed)}**.", ephemeral=True)

            async def on_timeout(self):
                pass   # just let it expire silently

        await interaction.response.send_message(
            embed=make_embed(), view=GroupManageView(), ephemeral=True)

    @bot.tree.command(name="remove_group", description="Remove a group by number")
    @app_commands.describe(group_number="From /list_groups")
    async def remove_group(interaction: discord.Interaction, group_number: int):
        r = await resolve_user_vc(interaction)
        if not r:
            return
        gid, vid, vc_ch = r
        key    = (gid, vid)
        groups = vc_groups.get(key, [])
        if not groups:
            return await interaction.response.send_message("No groups.", ephemeral=True)
        if not (1 <= group_number <= len(groups)):
            return await interaction.response.send_message(
                f"Range: 1-{len(groups)}.", ephemeral=True)
        removed = groups.pop(group_number - 1)
        button_cooldowns.get(key, {}).pop(group_number - 1, None)
        invalidate_group_audio(gid, vid)
        await interaction.response.defer(ephemeral=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_group_audio, gid, vid, groups)
        save_data()
        await interaction.followup.send(
            f"Removed Group {group_number}: {removed['members'][0][0]}", ephemeral=True)
        if key in vc_panels:
            try:
                pm   = vc_panels[key]
                ch   = bot.get_channel(pm["channel_id"]) or interaction.channel
                msg  = await ch.fetch_message(pm["msg_id"])
                ab   = bot_for_vc(gid, vid, all_bots_ref) or bot
                view = RallyStartView(groups, gid, vid, ab, all_bots_ref)
                active_views[key] = view
                view._message     = msg
                await msg.edit(embed=build_panel_embed(groups, vc_ch.name), view=view)
            except Exception:
                pass

    @bot.tree.command(name="rename_group", description="Set or clear a group's display name")
    @app_commands.describe(group_number="From /list_groups", name="New name, or leave blank to clear")
    async def rename_group(interaction: discord.Interaction, group_number: int, name: str = ""):
        r = await resolve_user_vc(interaction)
        if not r:
            return
        gid, vid, vc_ch = r
        key    = (gid, vid)
        groups = vc_groups.get(key, [])
        if not (1 <= group_number <= len(groups)):
            return await interaction.response.send_message(
                f"Range: 1-{len(groups)}.", ephemeral=True)
        group = groups[group_number - 1]
        name  = name.strip()[:32]
        if name:
            group["name"] = name
        else:
            group.pop("name", None)
        # Audio cache keys embed group index not name, so no re-synth needed —
        # but _getready and _startrally prompts DO use the name, so invalidate.
        invalidate_group_audio(gid, vid)
        await interaction.response.defer(ephemeral=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_group_audio, gid, vid, groups)
        save_data()
        label = f"**{name}**" if name else f"Group {group_number} (cleared)"
        # Update panel
        if key in active_views and key in vc_panels:
            pm = vc_panels[key]
            for b in all_bots_ref:
                ch = b.get_channel(pm.get("channel_id"))
                if ch:
                    try:
                        msg = await ch.fetch_message(pm["msg_id"])
                        v   = active_views[key]
                        v.groups = groups
                        v.update_buttons()
                        await msg.edit(embed=build_panel_embed(groups, vc_ch.name), view=v)
                    except Exception as e:
                        print(f"[panel] rename_group update failed: {e}")
                    break
        await interaction.followup.send(f"Group {group_number} renamed to {label}.", ephemeral=True)

    async def _update_panel_after_edit(gid, vid, groups, interaction_guild):
        return await _panel_refresh(gid, vid, groups, interaction_guild, all_bots_ref)

    @bot.tree.command(name="add_member",
                      description="Add a member to an existing group")
    @app_commands.describe(group_number="Group number from /list_groups",
                           member_name="Player name",
                           march_time="March time in seconds")
    async def add_member(interaction: discord.Interaction,
                         group_number: int, member_name: str, march_time: int):
        r = await resolve_user_vc(interaction)
        if not r:
            return
        gid, vid, vc_ch = r
        key    = (gid, vid)
        groups = vc_groups.get(key, [])
        if not (1 <= group_number <= len(groups)):
            return await interaction.response.send_message(
                f"Range: 1-{len(groups)}.", ephemeral=True)
        group = groups[group_number - 1]
        if len(group["members"]) >= MAX_MEMBERS:
            return await interaction.response.send_message(
                f"Group {group_number} is full ({MAX_MEMBERS} members max).", ephemeral=True)
        name = member_name.strip()
        if not name:
            return await interaction.response.send_message("Name can't be empty.", ephemeral=True)
        group["members"].append((name, march_time))
        group["members"].sort(key=lambda x: x[1], reverse=True)
        invalidate_group_audio(gid, vid)
        await interaction.response.defer(ephemeral=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_group_audio, gid, vid, groups)
        save_data()
        await _update_panel_after_edit(gid, vid, groups, interaction.guild)
        summary = " + ".join(f"**{n}** ({t}s)" + (" *(lead)*" if i == 0 else "")
                              for i, (n, t) in enumerate(group["members"]))
        await interaction.followup.send(
            f"Group {group_number} updated: {summary}", ephemeral=True)

    @bot.tree.command(name="remove_member",
                      description="Remove a member from a group")
    @app_commands.describe(group_number="Group number from /list_groups",
                           member_name="Exact name of the member to remove")
    async def remove_member(interaction: discord.Interaction,
                            group_number: int, member_name: str):
        r = await resolve_user_vc(interaction)
        if not r:
            return
        gid, vid, vc_ch = r
        key    = (gid, vid)
        groups = vc_groups.get(key, [])
        if not (1 <= group_number <= len(groups)):
            return await interaction.response.send_message(
                f"Range: 1-{len(groups)}.", ephemeral=True)
        group   = groups[group_number - 1]
        members = group["members"]
        name    = member_name.strip()
        match   = next((m for m in members if m[0].lower() == name.lower()), None)
        if not match:
            current = ", ".join(m[0] for m in members)
            return await interaction.response.send_message(
                f"**{name}** not found in group {group_number}. Members: {current}", ephemeral=True)
        if len(members) == 1:
            return await interaction.response.send_message(
                f"Can't remove the only member — use /remove_group {group_number} to delete the group.",
                ephemeral=True)
        members.remove(match)
        # Re-sort in case lead changed (highest time = lead)
        members.sort(key=lambda x: x[1], reverse=True)
        invalidate_group_audio(gid, vid)
        await interaction.response.defer(ephemeral=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generate_group_audio, gid, vid, groups)
        save_data()
        await _update_panel_after_edit(gid, vid, groups, interaction.guild)
        summary = " + ".join(f"**{n}** ({t}s)" + (" *(lead)*" if i == 0 else "")
                              for i, (n, t) in enumerate(members))
        await interaction.followup.send(
            f"Removed **{match[0]}** from group {group_number}: {summary}", ephemeral=True)

    @bot.tree.command(name="clear_groups",
                      description="Clear all groups in your voice channel")
    async def clear_groups(interaction: discord.Interaction):
        r = await resolve_user_vc(interaction)
        if not r:
            return
        gid, vid, vc_ch = r
        key = (gid, vid)
        invalidate_group_audio(gid, vid)
        vc_groups[key] = []
        if key in vc_panels:
            try:
                pm = vc_panels[key]
                ch = bot.get_channel(pm["channel_id"]) or interaction.channel
                m  = await ch.fetch_message(pm["msg_id"])
                await m.delete()
            except Exception:
                pass
            del vc_panels[key]
        active_views.pop(key, None)
        button_cooldowns.pop(key, None)
        save_data()
        await interaction.response.send_message(
            f"Groups cleared for **{vc_ch.name}**.", ephemeral=True)

    @bot.tree.command(name="reset_panel",
                      description="Clear the dead panel and post a fresh one")
    async def reset_panel(interaction: discord.Interaction):
        if not interaction.user.voice or not interaction.user.voice.channel:
            return await interaction.response.send_message("Join a VC first.", ephemeral=True)
        vc_ch    = interaction.user.voice.channel
        gid, vid = interaction.guild_id, vc_ch.id
        key      = (gid, vid)
        assigned = bot_for_vc(gid, vid, all_bots_ref)
        if not assigned:
            return await interaction.response.send_message("Use /join_voice first.", ephemeral=True)
        groups = vc_groups.get(key, [])
        if not groups:
            return await interaction.response.send_message("No groups. Use /add_group.", ephemeral=True)

        # Kill the old panel message and wipe state
        if key in vc_panels:
            pm = vc_panels[key]
            for b in all_bots_ref:
                ch = b.get_channel(pm.get("channel_id"))
                if ch:
                    try:
                        old_msg = await ch.fetch_message(pm["msg_id"])
                        await old_msg.delete()
                    except (discord.NotFound, discord.HTTPException):
                        pass
                    break
        active_views.pop(key, None)
        vc_panels.pop(key, None)

        # Post a clean panel in the current text channel (intentionally moves it here)
        view           = RallyStartView(groups, gid, vid, assigned, all_bots_ref)
        active_views[key] = view
        embed          = build_panel_embed(groups, vc_ch.name)
        await interaction.response.defer(ephemeral=True)
        msg            = await interaction.channel.send(embed=embed, view=view)
        view._message  = msg
        vc_panels[key]       = {"msg_id": msg.id, "channel_id": interaction.channel.id}
        vc_text_channel[key] = interaction.channel.id
        save_data()
        await interaction.followup.send("Panel reset.", ephemeral=True)
        asyncio.create_task(_bump_panel(interaction.channel))

    def _wrong_channel_msg(key, interaction):
        """Return an error string if interaction is in the wrong text channel, else None."""
        expected_cid = vc_text_channel.get(key)
        if expected_cid and interaction.channel_id != expected_cid:
            ch = interaction.guild.get_channel(expected_cid)
            name = f"<#{expected_cid}>" if ch else "its designated text channel"
            return (f"This VC's panel is in {name}. "
                    f"Run commands there, or use `/reset_panel` here to move it.")
        return None

    @bot.tree.command(name="start_rallies",
                      description="Post the rally panel for your voice channel")
    async def start_rallies(interaction: discord.Interaction):
        if not interaction.user.voice or not interaction.user.voice.channel:
            return await interaction.response.send_message("Join a VC first.", ephemeral=True)
        vc_ch    = interaction.user.voice.channel
        gid, vid = interaction.guild_id, vc_ch.id
        key      = (gid, vid)
        assigned = bot_for_vc(gid, vid, all_bots_ref)
        if not assigned:
            return await interaction.response.send_message("Use /join_voice first.", ephemeral=True)
        groups = vc_groups.get(key, [])
        if not groups:
            return await interaction.response.send_message(
                "No groups. Use /add_group.", ephemeral=True)
        # Warn if panel already exists in a different text channel
        err = _wrong_channel_msg(key, interaction)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        view  = RallyStartView(groups, gid, vid, assigned, all_bots_ref)
        active_views[key] = view
        embed = build_panel_embed(groups, vc_ch.name)
        if key in vc_panels:
            try:
                pm  = vc_panels[key]
                msg = await interaction.channel.fetch_message(pm["msg_id"])
                await msg.edit(embed=embed, view=view)
                view._message  = msg
                vc_panels[key] = {"msg_id": msg.id, "channel_id": interaction.channel.id}
                vc_text_channel[key] = interaction.channel.id
                save_data()
                return await interaction.response.send_message("Panel updated.", ephemeral=True)
            except discord.NotFound:
                pass
        msg = await interaction.channel.send(embed=embed, view=view)
        view._message  = msg
        vc_panels[key] = {"msg_id": msg.id, "channel_id": interaction.channel.id}
        vc_text_channel[key] = interaction.channel.id
        save_data()
        await interaction.response.send_message("Panel posted.", ephemeral=True)
        asyncio.create_task(_bump_panel(interaction.channel))

    return bot

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    generate_static_audio()
    load_data()
    all_bots = []
    for i, token in enumerate(TOKENS):
        if not token:
            raise ValueError(f"BOT_TOKEN_{i + 1} not set in environment")
        all_bots.append(make_bot(i, all_bots))
    await asyncio.gather(*(b.start(t) for b, t in zip(all_bots, TOKENS)))


if __name__ == "__main__":
    asyncio.run(main())