# Bumping Agent

A versatile AI assistant powered by LangGraph + DeepSeek, with a built-in RAG knowledge base, web search, stock quotes, weather, and language switching — all in a clean web UI.

## Features

| Tool | Description |
|------|-------------|
| Weather | Real-time weather for any city (via wttr.in) |
| Location | IP-based geolocation |
| Calculator | Math expression evaluation |
| Time | Current date and time |
| Document RAG | Upload .txt/.md files and ask questions |
| Stock Price | A-shares (China) and US stocks |
| Web Search | Bing search integration |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your DeepSeek or OpenAI API key
# Windows PowerShell:
#   $env:DEEPSEEK_API_KEY = "sk-your-key"
# macOS / Linux:
#   export DEEPSEEK_API_KEY="sk-your-key"

# 3. Run
python chat_app.py

# 4. Open http://localhost:5000
```

## Usage

- **Sidebar (left):** Upload documents, view chat history, start a new conversation
- **Header:** Language switcher (中文 / EN) — the assistant responds in the selected language
- **Chat input:** Type your question, the assistant uses tools to answer
- **Files:** Upload .txt or .md files, then ask questions about their content

The conversation history auto-saves. Click "New Chat" to start a fresh session — old chats appear in the sidebar.

## Configuration

Copy `.env.example` to `.env` and set your API key:

```
DEEPSEEK_API_KEY=sk-your-key-here
```

The app supports both DeepSeek (`deepseek-chat`) and OpenAI models.

## Architecture

```
chat_app.py            Flask server + LangGraph agent + RAG
templates/index.html   Web UI (HTML, CSS, JavaScript)
knowledge_docs/        Uploaded documents
knowledge_index/       TF-IDF index cache
```

The agent uses LangGraph's `create_react_agent` with 7 tools.
Document retrieval uses TF-IDF vectorization (scikit-learn).

## Tech Stack

- **Backend:** Python, Flask, LangGraph, LangChain, scikit-learn, DeepSeek API
- **Frontend:** Vanilla JavaScript, CSS
- **Storage:** Local file system for documents, browser localStorage for conversations

## License

MIT