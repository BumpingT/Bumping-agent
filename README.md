# Bumping Agent

**Your personal AI. Your data. Your rules.**

Bumping is a private, customizable AI assistant that runs entirely on your machine.

**Make it yours:**
- **Bring your own model** — Connect your own API key (DeepSeek or OpenAI)
- **Build your own knowledge base** — Upload docs, PDFs, Excel, emails, chat logs
- **Set your own personality** — Change the system prompt to make Bumping sound like you
- **Keep your data private** — Everything stays on your machine, zero servers

Download it, run it, customize it. Everyone gets their own personal AI.

## Quick Start

[![Download ZIP](https://img.shields.io/badge/Download%20ZIP-blue?style=for-the-badge&logo=github)](https://github.com/BumpingT/Bumping-agent/archive/refs/heads/main.zip)

> **No terminal setup needed.** Just run the app and enter your API key in the browser.

```bash
# 1. Download the ZIP and extract
# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python chat_app.py

# 4. Open http://localhost:5000
#    The app will prompt you to enter your API key in the browser.
```

Get an API key: [DeepSeek](https://platform.deepseek.com/) (recommended, cheaper) or [OpenAI](https://platform.openai.com/)

```bash
# Or set via environment variable (for server deployments):
# Windows:    $env:DEEPSEEK_API_KEY = "sk-your-key"
# macOS/Linux: export DEEPSEEK_API_KEY="sk-your-key"
```

art

[![Download ZIP](https://img.shields.io/badge/Download%20ZIP-blue?style=for-the-badge&logo=github)](https://github.com/BumpingT/Bumping-agent/archive/refs/heads/main.zip)

> **No terminal needed. Just run and enter your API key in the browser.**

```bash
# 1. Download the ZIP and extract
# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python chat_app.py

# 4. Open http://localhost:5000
#    The app will prompt you to enter your API key in the browser.
```

Get an API key: [DeepSeek](https://platform.deepseek.com/) (recommended, cheaper) or [OpenAI](https://platform.openai.com/)

```bash
# Or set via environment variable (for server deployments):
# Windows:    $env:DEEPSEEK_API_KEY = "sk-your-key"
# macOS/Linux: export DEEPSEEK_API_KEY="sk-your-key"
```

## Usage

- **Sidebar (left):** Upload documents, view chat history, start a new conversation
- **Header:** Language switcher (中文 / EN) — the assistant responds in the selected language
- **Chat input:** Type your question — Bumping responds warmly like a friend
- **Files:** Upload .txt/.md/.pdf files, then ask questions about their content
- **Images:** Upload images and ask Bumping to describe them

The conversation history auto-saves. Click "New Chat" to start a fresh session — old chats appear at the top of the sidebar, newest first.

## Personality

Bumping is designed to feel like a real person, not a robot:

- **Warm and patient** — like a kind older sister who listens
- **No bullet points** — all replies are natural flowing paragraphs
- **No AI disclaimers** — never says "as an AI" or "I'm a robot"
- **Gentle tone** — uses casual language, comfort when needed
- **Language-aware** — responds in the language you choose (Chinese / English)

Post-processing filters ensure the output stays clean and human-like, regardless of the underlying model's habits.

## Configuration

Copy `.env.example` to `.env` and set your API key (only need one):

```
DEEPSEEK_API_KEY=sk-your-key-here
```

The app auto-detects which key you set. If both are set, OpenAI takes priority.

| Variable | Model | Notes |
|----------|-------|-------|
| `DEEPSEEK_API_KEY` | `deepseek-chat` | Cheaper, recommended for daily use |
| `OPENAI_API_KEY` | `gpt-4o-mini` | Supports vision (image analysis) |

## Architecture

```
chat_app.py            Flask server + LangGraph agent + RAG + post-processing
templates/index.html   Web UI (HTML, CSS, vanilla JavaScript)
knowledge_docs/        Uploaded documents for RAG
knowledge_index/       TF-IDF index cache (auto-generated)
uploads/images/        Uploaded images for vision analysis
```

The agent uses LangGraph's `create_react_agent` with 10 tools (including image analysis and webpage fetch).
Document retrieval uses TF-IDF vectorization (scikit-learn) over 500-character chunks.

## Tech Stack

- **Backend:** Python, Flask, LangGraph, LangChain, scikit-learn, DeepSeek API
- **Frontend:** Vanilla JavaScript, CSS
- **Storage:** Browser localStorage for conversations, local filesystem for documents

## License

MIT
