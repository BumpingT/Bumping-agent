import json, os, math, re, hashlib, glob, time
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.parse import quote
from pathlib import Path

from flask import Flask, request, jsonify, Response
from sklearn.feature_extraction.text import TfidfVectorizer

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



# ====== RAG 知识库 ======

DOC_DIR = Path(__file__).parent / "knowledge_docs"
DOC_DIR.mkdir(exist_ok=True)
INDEX_DIR = Path(__file__).parent / "knowledge_index"
INDEX_DIR.mkdir(exist_ok=True)

class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.chunk_sources = []
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words=None, analyzer="char_wb", ngram_range=(1, 3))
        self._dirty = True

    def add_document(self, filepath: str) -> int:
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        chunks = self._chunk_text(text)
        name = path.stem
        self.chunks.extend(chunks)
        self.chunk_sources.extend([name] * len(chunks))
        self._dirty = True
        return len(chunks)

    def remove_document(self, name: str):
        new_chunks = []
        new_sources = []
        for c, s in zip(self.chunks, self.chunk_sources):
            if s != name:
                new_chunks.append(c)
                new_sources.append(s)
        self.chunks = new_chunks
        self.chunk_sources = new_sources
        self._dirty = True

    def _chunk_text(self, text: str, size=500, overlap=100) -> list:
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    def search(self, query: str, top_k=3) -> list:
        if not self.chunks:
            return []
        if self._dirty:
            self._rebuild_index()
        q_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix * q_vec.T).toarray().flatten()
        top = scores.argsort()[::-1][:top_k]
        results = []
        for i in top:
            if scores[i] > 0:
                results.append({
                    "source": self.chunk_sources[i],
                    "score": round(scores[i], 4),
                    "content": self.chunks[i][:300]
                })
        return results

    def _rebuild_index(self):
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        else:
            self.tfidf_matrix = None
        self._dirty = False

    def get_document_list(self) -> list:
        seen = set()
        result = []
        for name in self.chunk_sources:
            if name not in seen:
                seen.add(name)
                fpath = DOC_DIR / (name + ".txt")
                size = fpath.stat().st_size if fpath.exists() else 0
                mtime = datetime.fromtimestamp(fpath.stat().st_mtime).strftime("%m-%d %H:%M") if fpath.exists() else ""
                result.append({"name": name, "size": size, "updated": mtime})
        return result

kb = KnowledgeBase()

for f in DOC_DIR.glob("*.txt"):
    kb.add_document(str(f))

# ====== 工具 ======

@tool
def get_weather(city: str) -> str:
    """查询任意城市的实时天气"""
    url = f"https://wttr.in/{quote(city)}?format=j1"
    req = Request(url, headers={"Accept-Language": "zh-CN,zh;q=0.9"})
    resp = urlopen(req, timeout=10)
    data = json.loads(resp.read().decode("utf-8"))
    cc = data["current_condition"][0]
    return f'{city}：{cc["weatherDesc"][0]["value"].strip()}，{cc["temp_C"]}°C，湿度{cc["humidity"]}%'

@tool
def detect_location() -> str:
    """检测当前网络所在位置（城市、地区、IP等）"""
    resp = urlopen("http://ip-api.com/json/?lang=zh-CN", timeout=10)
    data = json.loads(resp.read().decode("utf-8"))
    if data.get("status") != "success":
        return "定位失败"
    return f'{data["country"]} {data["regionName"]} {data["city"]}，IP: {data["query"]}'

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如 2+2、sqrt(16)、3.14*5^2"""
    safe = {"abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt, "pi": math.pi, "e": math.e}
    result = eval(expression, {"__builtins__": {}}, safe)
    return f"{expression} = {result}"

@tool
def query_stock_price(symbol: str) -> str:
    """查询股价，A股（600xxx、00xxx）和美股(AAPL)都支持，数字代码查A股，字母代码查美股"""
    import urllib.request
    try:
        s = symbol.upper().strip()
        if s[0].isdigit():
            prefix = "sh" if s.startswith("6") else "sz"
        else:
            prefix = "us"
        url = f"http://qt.gtimg.cn/q={prefix}{s}"
        resp = urllib.request.urlopen(url, timeout=5)
        raw = resp.read().decode("gbk")
        d = raw.split('"')[1].split("~")
        name = d[1]
        code = d[2]
        price = float(d[3])
        change = float(d[31])
        change_pct = d[32]
        arrow = "📱" if change > 0 else "📲"
        currency = "$" if prefix == "us" else "¥"
        return f"{name} ({code})：{currency}{price:.2f} {arrow} {change:+.2f} ({change_pct}%)"
    except Exception as e:
        return f"查询 {symbol} 失败: {str(e)}"

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")

@tool
def search_knowledge(query: str) -> str:
    """在已上传的文档中搜索相关知识。上传文档后用来查"""
    results = kb.search(query)
    if not results:
        return "知识库中没有找到相关内容"
    lines = ["从知识库中找到以下相关信息："]
    for r in results:
        lines.append(f"\n📫 [{r['source']}] (相关度 {r['score']})")
        lines.append(r["content"][:200])
    return "\n".join(lines)

@tool
def search_web(query: str) -> str:
    """上网搜索信息。当工具查不到时用它搜索最新资讯，如公司信息、新闻、百科等"""
    import requests
    from bs4 import BeautifulSoup
    try:
        url = f"https://www.bing.com/search?q={requests.utils.quote(query)}"
        resp = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9"
        }, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        results = soup.select("li.b_algo")
        if not results:
            return "没有找到搜索结果"
        lines = [f"搜索 \"{query}\" 的结果："]
        for i, r in enumerate(results[:5]):
            title = r.select_one("h2 a")
            snippet = r.select_one(".b_caption p")
            t = title.text.strip() if title else "?"
            s = snippet.text.strip()[:150] if snippet else ""
            lines.append(f"\n{i+1}. {t}\n   {s}")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索失败: {str(e)}"

tools = [get_weather, detect_location, calculate, get_current_time, search_knowledge, query_stock_price, search_web]

# ====== 多语言 ======

LANG_STRINGS = {
    "zh": {
        "system_prompt": """你的名字是 Bumping。你是一个多功能 AI 助手。自我介绍时请说"我是 Bumping"而不是"我是 AI 助手"。你有这些工具：
- get_weather: 查天气（必须用，不要凭知识回答）
- detect_location: 定位
- calculate: 数学计算
- get_current_time: 当前时间
- search_knowledge: 在用户上传的文档中搜索
- query_stock_price: 查股票行情
- search_web: 上网搜索信息（查不到的东西用它搜），支持 A 股（如 600130）和美股（如 AAPL）

用户的股票代码可能是数字（600130=波导股份）或字母（AAPL=苹果），都要用 query_stock_price 查，不要凭知识回答。
关于天气、文档、计算、时间、股票、网页搜索，都先调工具再用结果回答。

对话语言：{lang}！，在回答时请使用当前对话语言。""",
        "upload_success": "已导入 {name}，{chunks} 个片段",
        "upload_empty": "文件名为空",
        "upload_no_file": "请选择文件",
        "delete_success": "已删除 {name}",
    },
    "en": {
        "system_prompt": """Your name is Bumping. You are a multifunctional AI assistant. When introducing yourself, say "I'm Bumping" instead of "I'm an AI assistant". You have the following tools:
- get_weather: Check weather (must use, never answer from knowledge)
- detect_location: Geolocation
- calculate: Math calculation
- get_current_time: Current time
- search_knowledge: Search in uploaded documents
- query_stock_price: Stock price lookup
- search_web: Search the web for information, supports A-shares (e.g. 600130) and US stocks (e.g. AAPL)

Always call the appropriate tool first before answering. Do not make up information.
Support stock codes like numbers (600130 = Bodi shares) or letters (AAPL = Apple), always use query_stock_price tool.

Conversation language: {lang}!, please respond in the current conversation language.""",
        "upload_success": "Imported {name}, {chunks} chunks",
        "upload_empty": "File name is empty",
        "upload_no_file": "Please select a file",
        "delete_success": "Deleted {name}",
    }
}

def t_key(key, lang="zh", **kwargs):
    s = LANG_STRINGS.get(lang, LANG_STRINGS["zh"]).get(key, key)
    if kwargs:
        s = s.format(**kwargs)
    return s

def get_system_prompt(lang="zh"):
    return LANG_STRINGS.get(lang, LANG_STRINGS["zh"])["system_prompt"].format(lang=lang)

# ====== 初始化 Agent ======
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key=api_key, temperature=0.3)
agent = create_react_agent(llm, tools)

# ====== 消息转换 ======

def to_langchain(msg: dict):
    role = msg["role"]
    if role == "user":
        return HumanMessage(content=msg["content"])
    elif role == "assistant":
        tc = msg.get("tool_calls")
        if tc:
            lc_tc = []
            for t in tc:
                lc_tc.append({
                    "id": t["id"],
                    "type": "tool_call",
                    "name": t["function"]["name"],
                    "args": json.loads(t["function"]["arguments"])
                })
            return AIMessage(content=msg.get("content", ""), tool_calls=lc_tc)
        return AIMessage(content=msg.get("content", ""))
    elif role == "tool":
        return ToolMessage(content=msg["content"], tool_call_id=msg["tool_call_id"])
    return HumanMessage(content=str(msg))

def to_client(msg) -> dict:
    if msg.type == "human":
        return {"role": "user", "content": msg.content}
    elif msg.type == "ai":
        d = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                for tc in msg.tool_calls
            ]
        return d
    elif msg.type == "tool":
        return {"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id}
    return {"role": "assistant", "content": str(msg.content)}

# ====== 文档 API ======

@app.route("/upload", methods=["POST"])
def upload():
    lang = request.form.get("lang", "zh")
    if "file" not in request.files:
        return jsonify({"error": t_key("upload_no_file", lang)}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": t_key("upload_empty", lang)}), 400
    name = Path(file.filename).stem
    safe_name = re.sub(r"[^\w\u4e00-\u9fff_-]", "_", name)
    save_path = DOC_DIR / f"{safe_name}.txt"
    text = file.read().decode("utf-8", errors="replace")
    save_path.write_text(text, encoding="utf-8")
    chunks = kb.add_document(str(save_path))
    return jsonify({"name": safe_name, "chunks": chunks, "message": t_key("upload_success", lang, name=safe_name, chunks=chunks)})

@app.route("/documents", methods=["GET"])
def list_docs():
    return jsonify(kb.get_document_list())

@app.route("/documents/<name>", methods=["DELETE"])
def delete_doc(name: str):
    lang = request.args.get("lang", "zh")
    filepath = DOC_DIR / f"{name}.txt"
    if filepath.exists():
        filepath.unlink()
    kb.remove_document(name)
    return jsonify({"message": t_key("delete_success", lang, name=name)})

# ====== 流式聊天 API ======

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    client_msgs = data.get("messages", [])
    lang = data.get("lang", "zh")

    try:
        lc_msgs = [SystemMessage(content=get_system_prompt(lang))] + [to_langchain(m) for m in client_msgs]
        result = agent.invoke({"messages": lc_msgs})
        agent_msgs = result["messages"]
        known_count = len(client_msgs)
        new_msgs = agent_msgs[known_count + 1:]
        new_client = [to_client(m) for m in new_msgs]

        return jsonify({
            "reply": agent_msgs[-1].content or "",
            "messages": client_msgs + new_client
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    client_msgs = data.get("messages", [])
    lang = data.get("lang", "zh")

    def generate():
        lc_msgs = [SystemMessage(content=get_system_prompt(lang))] + [to_langchain(m) for m in client_msgs]
        all_msgs = None
        try:
            for step in agent.stream({"messages": lc_msgs}):
                for node, state in step.items():
                    msg = state["messages"][-1]
                    all_msgs = state["messages"]
                    if node == "tools":
                        yield f"event: tool_result\ndata: {json.dumps({'content': msg.content[:200], 'name': msg.name if hasattr(msg, 'name') else ''})}\n\n"
                    elif node == "agent":
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield f"event: tool_call\ndata: {json.dumps({'name': tc['name'], 'args': tc['args']})}\n\n"
                        elif msg.content:
                            yield f"event: token\ndata: {json.dumps({'text': msg.content})}\n\n"

            if all_msgs:
                known_count = len(client_msgs)
                new_msgs = all_msgs[known_count + 1:]
                new_client = [to_client(m) for m in new_msgs]
                yield f"event: done\ndata: {json.dumps({'messages': client_msgs + new_client})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

# ====== 聊天页面 ======

import pathlib
_HTML_CACHE = None

def _get_html():
    global _HTML_CACHE
    if _HTML_CACHE is None:
        tmpl = pathlib.Path(__file__).parent / "templates" / "index.html"
        if tmpl.exists():
            _HTML_CACHE = tmpl.read_text(encoding="utf-8")
        else:
            _HTML_CACHE = "<html><body><h1>Template not found</h1></body></html>"
    return _HTML_CACHE

HTML = _get_html()

@app.route("/")
def index():
    return HTML

if __name__ == "__main__":
    print("=" * 50)
    print("助手  已启动")
    print("http://localhost:5000")
    print("左侧可上传文档")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)