# PaperFlow - arXiv Paper Agent

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/aritra741/paper-flow)

An automated agent that monitors arXiv for papers on **any topics you choose**. It uses a local LLM using ollama to filter papers for relevance and sends email notifications when new relevant papers are found. Fully customizable - search for papers on machine learning, physics, mathematics, or any field you're interested in!

## Features

- 🔍 Searches arXiv for papers on any topics you configure
- 🤖 Uses an LLM locally to intelligently filter papers for relevance
- 📧 Sends email notifications for new relevant papers
- 💾 Caches previously seen papers to avoid duplicate notifications
- ⏰ Can run automatically daily via macOS LaunchAgent

## Prerequisites

Before setting up, make sure you have:

1. **Python 3** installed (check with `python3 --version`)
2. **Ollama** installed and running
   - Install from: https://ollama.ai
   - The agent uses the `gemma2:2b` model by default
   - Pull the model: `ollama pull gemma2:2b`
3. **macOS** (for automatic scheduling via LaunchAgent)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/aritra741/paper-flow.git
cd paper-flow
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Email (Optional)

If you want email notifications:

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` with your email settings:
   ```bash
   # Open .env in your editor
   nano .env  # or use your preferred editor
   ```

3. Fill in your email details:
   - `USER_EMAIL`: Your email address
   - `SMTP_USERNAME`: Usually the same as USER_EMAIL
   - `SMTP_PASSWORD`: For Gmail, use an [App Password](https://myaccount.google.com/apppasswords), not your regular password
   - `SMTP_SERVER`: Defaults to `smtp.gmail.com` (change if using a different provider)
   - `SMTP_PORT`: Defaults to `587` (TLS)

**Note:** The agent will work without email configuration, but won't send notifications.

### 5. Test Run

Run the agent manually to test:

```bash
source .venv/bin/activate  # If not already activated
python3 main.py
```

This will:
- Search arXiv for papers from the last 7 days
- Filter them using Ollama
- Display relevant papers
- Send email notifications if configured

## Automatic Daily Runs (macOS)

To set up the agent to run automatically every day at 9:00 AM:

```bash
./setup.sh
```

This will:
- Create a macOS LaunchAgent
- Schedule daily runs at 9:00 AM
- Configure logging to `agent.log`

### Managing the Scheduled Agent

- **Check status**: `launchctl list | grep arxiv`
- **Run manually**: `launchctl start com.arxiv.agent`
- **View logs**: `cat agent.log`
- **Uninstall**: `./uninstall.sh`

### Changing the Scheduled Time

The agent runs daily at 9:00 AM by default. To change this:

**Steps to change the scheduled time** 
1. Edit `setup.sh` and change lines 64-66:
   ```bash
   <key>Hour</key>
   <integer>9</integer>  # Change 9 to your desired hour (0-23)
   <key>Minute</key>
   <integer>0</integer>  # Change 0 to your desired minute (0-59)
   ```
2. Run `./setup.sh` again

## Configuration

The agent is fully customizable! Edit `main.py` to search for papers on **any topics you want**:

- **Topics**: Modify the `TOPICS` list (line 12-18) - change it to search for papers on machine learning, quantum computing, astrophysics, or anything else!
- **Max Results**: Change `MAX_RESULTS` (line 19) - how many papers to fetch per topic
- **LLM Model**: Change `MODEL` (line 20) - must be installed in Ollama
- **Days Back**: Change `DAYS_BACK` (line 21) - how far back to search (default: 7 days)
- **Verbose Mode**: Set `VERBOSE` to `False` (line 22) for less output

**Example:** To search for papers on "quantum computing" and "neural networks", simply change the `TOPICS` list to:
```python
TOPICS = [
    "quantum computing",
    "neural networks",
]
```

## How It Works

1. **Search**: Queries arXiv API for papers matching your configured topics
2. **Filter**: Uses Ollama LLM to intelligently determine if papers are relevant to your interests
3. **Cache**: Stores relevant paper IDs in `relevant_papers.txt` to avoid duplicate notifications
4. **Notify**: Sends email with new relevant papers (if configured)

## Troubleshooting

### Ollama Not Found

If you see errors about Ollama:
- Make sure Ollama is installed: `ollama --version`
- Make sure Ollama is running: `ollama list`
- Pull the required model: `ollama pull gemma2:2b`

### Email Not Sending

- Check that `.env` file exists and has correct values
- For Gmail, make sure you're using an App Password, not your regular password
- Check `agent.log` for error messages
- Test email settings manually if needed

### LaunchAgent Not Running

- Check status: `launchctl list | grep arxiv`
- Check logs: `cat agent.log`
- Reload: `launchctl unload ~/Library/LaunchAgents/com.arxiv.agent.plist && launchctl load ~/Library/LaunchAgents/com.arxiv.agent.plist`

### Virtual Environment Issues

If you get import errors:
- Make sure virtual environment is activated: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## Files

- `main.py` - Main agent script
- `requirements.txt` - Python dependencies
- `setup.sh` - macOS LaunchAgent setup script
- `uninstall.sh` - Remove LaunchAgent
- `env.example` - Example environment configuration
- `.env` - Your actual configuration (create from env.example)
- `relevant_papers.txt` - Cache of previously seen papers
- `agent.log` - Log file (created when running)

## License

See LICENSE file for details.
