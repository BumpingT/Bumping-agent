# Multi-Tool AI Agent

A versatile AI assistant powered by LangGraph + DeepSeek, with a built-in RAG knowledge base, web search, stock quotes, weather, and more ? all in a clean web UI.

## Features

| Tool | Description |
|------|-------------|
| ?? Weather | Real-time weather for any city (via wttr.in) |
| ?? Location | IP-based geolocation |
| ?? Calculator | Math expression evaluation |
| ?? Time | Current date & time |
| ?? Document RAG | Upload .txt/.md files and ask questions |
| ?? Stock Price | A-shares (China) & US stocks |
| ?? Web Search | Bing search integration |

## Quick Start

`ash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (DeepSeek or OpenAI)
# Windows PowerShell:
#   ='sk-your-key'
# macOS/Linux:
#   export DEEPSEEK_API_KEY='sk-your-key'

# 3. Run
python chat_app.py

# 4. Open http://localhost:5000
`

## Screenshots

### Chat Interface
- Sidebar: document management & chat history
- Header: language switcher (?? / EN)
- Streaming responses with tool-call visibility
- Auto-saved conversation history

### Knowledge Base
- Drag-and-drop file upload (.txt, .md)
- TF-IDF based semantic search
- Persistent storage across restarts

## Configuration

Copy .env.example to .env and set your API key:

`
DEEPSEEK_API_KEY=sk-your-key-here
`

The app supports both DeepSeek (deepseek-chat) and OpenAI models.
Set either DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable.

## Architecture

`
chat_app.py          ? Flask server + LangGraph agent + RAG
templates/
  index.html         ? Web UI (HTML/CSS/JS)
knowledge_docs/      ? Uploaded documents
knowledge_index/     ? TF-IDF index cache
`

The agent uses LangGraph's create_react_agent with 7 tools.
Document retrieval uses TF-IDF vectorization (scikit-learn).

## Tech Stack

**Backend:** Python, Flask, LangGraph, LangChain, scikit-learn, DeepSeek API

**Frontend:** Vanilla JS, CSS, Server-Sent Events for streaming

## License

MIT
