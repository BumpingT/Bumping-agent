# Bumping Agent

A warm, emotional AI companion — powered by LangGraph + DeepSeek, with built-in RAG knowledge base, web search, stock quotes, weather, image analysis, and language switching. All wrapped in a clean web UI.

Bumping speaks like a caring older sister (知心姐姐): patient, warm, and reassuring. No robotic lists, no "I'm an AI" disclaimers — just natural, flowing conversation.

## Features

| Feature | Description |
|---------|-------------|
| ❤️ **Emotional Chat** | Warm, human-like conversation. Never bullet points or robotic replies |
| 🌤 **Weather** | Real-time weather for any city (via wttr.in) |
| 📍 **Location** | IP-based geolocation |
| 🧮 **Calculator** | Math expression evaluation |
| 🕐 **Time** | Current date and time |
| 📄 **Document RAG** | Upload .txt/.md/.pdf files and ask questions |
| 📈 **Stock Price** | A-shares (China) and US stocks |
| 🌐 **Web Search** | Bing search integration |
| 🖼 **Image Analysis** | Analyze uploaded images (via GPT-4o-mini vision) |
| 📰 **Webpage Fetch** | Extract and summarize webpage content |

## Quick Start

> **You need an API key to use Bumping.** The app won't start without one.
> Get a key from [DeepSeek](https://platform.deepseek.com/) (cheaper, recommended) or [OpenAI](https://platform.openai.com/).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (pick one)
# Windows PowerShell:
#   $env:DEEPSEEK_API_KEY = "sk-your-deepseek-key"
# macOS / Linux:
#   export DEEPSEEK_API_KEY="sk-your-deepseek-key"

# Or use OpenAI instead:
#   $env:OPENAI_API_KEY = "sk-your-openai-key"

# 3. Run
python chat_app.py

# 4. Open http://localhost:5000
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
