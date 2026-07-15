---
name: bumping
description: "Bumping Agent project knowledge: a Flask + LangGraph + DeepSeek AI assistant with RAG knowledge base, web search, stock quotes, weather, image analysis, and webpage fetching."
---

# Bumping Agent

Multi-tool AI assistant at `C:\Users\Administrator\Documents\ai agent`.

GitHub: https://github.com/BumpingT/bumping-agent

## Core Files

| File | Purpose |
|------|---------|
| `chat_app.py` | Flask server + LangGraph agent + all tools + upload API |
| `templates/index.html` | Full web UI (vanilla JS, no framework) |
| `tests.py` | 6 basic tests |
| `requirements.txt` | Flask, langchain, langgraph, scikit-learn |
| `.env.example` | API key template |
| `knowledge_docs/` | Uploaded documents for RAG |
| `uploads/images/` | Uploaded images for vision analysis |

## Tools (9)

- `get_weather(city)` — wttr.in
- `detect_location()` — ip-api.com  
- `calculate(expr)` — safe math eval
- `get_current_time()` — datetime.now
- `search_knowledge(query)` — TF-IDF over uploaded docs
- `query_stock_price(symbol)` — A-shares + US stocks
- `search_web(query)` — Bing + BS4
- `analyze_image(filename)` — GPT-4o-mini vision (needs OPENAI_API_KEY)
- `fetch_webpage(url)` — extract webpage text

## API Keys

- `DEEPSEEK_API_KEY` → deepseek-chat (default)
- `OPENAI_API_KEY` → gpt-4o-mini (vision support)
- Both can be set; OpenAI takes priority.

## Key Config

- LLM: `langchain_openai.ChatOpenAI` with dynamic `base_url`
- Agent: `langgraph.prebuilt.create_react_agent`
- RAG: TF-IDF (scikit-learn), 500-char chunks, 100 overlap
- Frontend: vanilla JS, multi-language (zh/en)
- Port 5000, Flask dev server
- Template loaded from `templates/index.html`, `_get_html(force=True)` per request
- Chat auto-saves to localStorage; old chats under `chat_TIMESTAMP`
- Images shown as horizontal strip; PDFs auto-extracted to text