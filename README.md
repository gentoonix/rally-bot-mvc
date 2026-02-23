# rally-bot-mvc

A Discord bot for coordinating multiple simultaneous rally events within Voice Channels (VC). Built with Python using an MVC (Model-View-Controller) architecture for clean separation of concerns and easy extensibility.

---

## Features

- Coordinate and manage multiple rallies within Discord voice channels simultaneously
- Simple command-based interface for rally creation and management
- Lightweight and easy to self-host
- Docker support for quick, containerized deployment

---

## Prerequisites

- A [Discord Developer Portal](https://discord.com/developers/applications) account
- A Discord Bot Token with the appropriate permissions
- Python 3.10+ **or** Docker (recommended)

---

## Configuration

Before running the bot, you need to set your Discord bot token. Create a `.env` file in the project root:

```env
DISCORD_TOKEN=your_discord_bot_token_here
```

> **Never commit your `.env` file or token to version control.**

---

## Installation

### Option 1 — Docker (Recommended)

Docker is the easiest way to get up and running without managing Python dependencies manually.

#### 1. Clone the repository

```bash
git clone https://github.com/gentoonix/rally-bot-mvc.git
cd rally-bot-mvc
```

#### 2. Create your `.env` file

```bash
cp .env.example .env   # or create it manually
# Edit .env and set your DISCORD_TOKEN
```

#### 3. Build the Docker image

```bash
docker build -t rally-bot-mvc .
```

#### 4. Run the container

```bash
docker run -d \
  --name rally-bot \
  --env-file .env \
  --restart unless-stopped \
  rally-bot-mvc
```

The `--restart unless-stopped` flag ensures the bot automatically restarts after a system reboot or crash.

#### Useful Docker commands

```bash
# View live logs
docker logs -f rally-bot

# Stop the bot
docker stop rally-bot

# Restart the bot
docker restart rally-bot

# Remove the container
docker rm -f rally-bot
```

---

### Option 2 — Docker Compose

If you prefer Docker Compose, create a `docker-compose.yml` in the project root:

```yaml
version: "3.8"

services:
  rally-bot:
    build: .
    container_name: rally-bot
    env_file:
      - .env
    restart: unless-stopped
```

Then start it with:

```bash
docker compose up -d --build
```

---

### Option 3 — Manual (Python)

#### 1. Clone the repository

```bash
git clone https://github.com/gentoonix/rally-bot-mvc.git
cd rally-bot-mvc
```

#### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate.bat      # Windows
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Set your token and run

```bash
export DISCORD_TOKEN=your_discord_bot_token_here   # Linux/macOS
set DISCORD_TOKEN=your_discord_bot_token_here       # Windows CMD

python bot.py
```

---

## Discord Bot Permissions

When inviting the bot to your server, make sure it has the following permissions:

- **Read Messages / View Channels**
- **Send Messages**
- **Connect** (Voice)
- **Speak** (Voice, if applicable)
- **Manage Channels** (if rally channels are created dynamically)

You can generate an invite URL from the [Discord Developer Portal](https://discord.com/developers/applications) under **OAuth2 → URL Generator**.

---

## Project Structure

```
rally-bot-mvc/
├── bot.py              # Entry point — initializes and runs the bot
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build instructions
├── .env                # Your local secrets (not committed)
└── LICENSE
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## License

See [LICENSE](LICENSE) for details.
