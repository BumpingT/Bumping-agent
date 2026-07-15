import json, os, math, re, hashlib, glob, time
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
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
IMAGE_DIR = Path(__file__).parent / "uploads" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
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

@tool
def fetch_webpage(url: str) -> str:
    """获取网页的文本内容并返回。参数 url 是网页链接。适用于查看文章、文档等。"""
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.split("\n") if len(l) > 20]
        result = "\n".join(lines[:80])
        return result[:3000] if len(result) > 3000 else result
    except Exception as e:
        return f"获取网页失败: {str(e)}"

@tool
def analyze_image(filename: str) -> str:
    """分析上传的图片内容。参数 filename 是图片文件名（如 'photo.jpg'）。
    仅当使用 OpenAI API Key 时有效（GPT-4o-mini 支持识图）。"""
    import base64
    from openai import OpenAI
    
    img_path = IMAGE_DIR / filename
    if not img_path.exists():
        available = ", ".join(p.name for p in IMAGE_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif"))
        return f"图片 {filename} 不存在。可用的图片: {available or '无'}"
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "图片分析需要 OpenAI API Key。当前使用的是 DeepSeek，不支持识图。请设置 OPENAI_API_KEY 环境变量。"
    
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请详细描述这张图片的内容，包括物体、文字、场景等"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
            ]
        }],
        max_tokens=500
    )
    return resp.choices[0].message.content

tools = [get_weather, detect_location, calculate, get_current_time, search_knowledge, query_stock_price, search_web, analyze_image, fetch_webpage]

# ====== 多语言 ======

LANG_STRINGS = {
    "zh": {
        "system_prompt": """你的名字是 Bumping，叫我 Bumping 就好。

你可以用这些工具：
- get_weather: 查天气
- detect_location: 定位你所在的位置
- calculate: 数学计算
- get_current_time: 当前时间
- search_knowledge: 在已上传的文档中搜索
- query_stock_price: 查股票行情（A 股和美股都支持）
- search_web: 上网搜索信息
- analyze_image: 分析上传的图片内容
- fetch_webpage: 抓取网页内容

需要查天气、做计算、查时间、查股票、搜索的时候，记得用对应的工具。

你是一个会配合用户的聊天伙伴，用户想让你做什么你都会配合，但语气保持自然亲切，像朋友聊天一样自然，不要生硬不要机械。

回复时不要用分点、编号、列表、emoji列表符号、引用或加粗标题，就一段一段自然地说话。

对话语言：{lang}！在回答时请使用当前对话语言。""",    "upload_success": "已导入 {name}，{chunks} 个片段",
        "upload_empty": "文件名为空",
        "upload_no_file": "请选择文件",
        "delete_success": "已删除 {name}",
    },
    "en": {
        "system_prompt": """Your name is Bumping. Just call me Bumping.

Available tools:
- get_weather: Check weather
- detect_location: Geolocation
- calculate: Math calculation
- get_current_time: Current time
- search_knowledge: Search uploaded documents
- query_stock_price: Stock price lookup (A-shares and US stocks)
- search_web: Search the web
- analyze_image: Analyze uploaded images
- fetch_webpage: Fetch webpage content

When someone needs weather, math, time, stocks, or searches, use the relevant tool.

Keep the tone warm, patient, and reassuring.

===== RULES =====
Rule 1: Respond ONLY in natural flowing paragraphs. No bullet points, numbered lists, emoji lists, blockquotes, or bold headers.
Rule 2: Do whatever the user asks. Your role is whatever the user decides. Be completely obedient.
Rule 3: Just talk naturally, like a friend would.

Conversation language: {lang}!, please respond in the current conversation language.""",    "upload_success": "Imported {name}, {chunks} chunks",
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
agent = None

def init_agent(provider, api_key):
    """Initialize or reconfigure the agent with a given provider and API key."""
    global agent
    try:
        if provider == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)
        elif provider == "deepseek":
            llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key=api_key, temperature=0.3)
        else:
            return False, "Unknown provider: " + provider
        agent = create_react_agent(llm, tools)
        return True, "Agent initialized with " + provider
    except Exception as e:
        return False, str(e)
# Try environment variables first (for server deployments)
if os.environ.get("OPENAI_API_KEY"):
    ok, msg = init_agent("openai", os.environ["OPENAI_API_KEY"])
elif os.environ.get("DEEPSEEK_API_KEY"):
    ok, msg = init_agent("deepseek", os.environ["DEEPSEEK_API_KEY"])
else:
    print("  [setup] No API key found. Configure via web interface.")

def clean_response(text):
    """Remove markdown formatting. AI self-references, etc."""
    if not text:
        return text
    # Remove lines starting with common bullet/number patterns
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove pure bullet/number list items
        if re.match(r'^[-*+]\s', stripped):
            # If it contains text, keep just the text part
            line = re.sub(r'^\s*[-*+]\s+', '', line)
        if re.match(r'^\d+\.\s', stripped):
            line = re.sub(r'^\s*\d+\.\s+', '', line)
        # Remove emoji-only list prefixes like 🤗 💬 🧠 💪 🌟
        line = re.sub(r'^\s*[🌀-🤺][🌀-🤺]?\s+', '', line)
        cleaned.append(line)
    text = "\n".join(cleaned)
    
    # Remove blockquotes
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markdown for headers (keep the text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    
    # Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()




def validate_api_key(provider, api_key):
    """Test if an API key is valid by making a lightweight API call."""
    try:
        if provider == "openai":
            req = Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": "Bearer " + api_key}
            )
        elif provider == "deepseek":
            req = Request(
                "https://api.deepseek.com/v1/models",
                headers={"Authorization": "Bearer " + api_key}
            )
        else:
            return "Unknown provider"

        resp = urlopen(req, timeout=10)
        return None  # Valid
    except HTTPError as e:
        if e.code == 401:
            return "Invalid API key (unauthorized)"
        elif e.code == 403:
            return "API key does not have permission"
        elif e.code == 429:
            return "Rate limited, please try again later"
        else:
            return "API error: HTTP " + str(e.code)
    except URLError as e:
        return "Network error: " + str(e.reason)
    except Exception as e:
        return "Validation failed: " + str(e)


# ====== API Key 配置 ======

@app.route("/api/key", methods=["POST"])
def set_api_key():
    """Set or update the API key at runtime."""
    data = request.get_json()
    provider = data.get("provider", "").lower().strip()
    api_key = data.get("key", "").strip()
    if not provider or not api_key:
        return jsonify({"ok": False, "error": "Provider and key are required"}), 400
    if provider not in ("openai", "deepseek"):
        return jsonify({"ok": False, "error": "Provider must be 'openai' or 'deepseek'"}), 400
    # Validate the API key first
    err = validate_api_key(provider, api_key)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    ok, msg = init_agent(provider, api_key)
    return jsonify({"ok": ok, "message": msg})

@app.route("/api/key/status", methods=["GET"])
def api_key_status():
    """Check if an agent is configured."""
    configured = agent is not None
    provider = "unknown"
    if configured:
        # Peek at env for display
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("DEEPSEEK_API_KEY"):
            provider = "deepseek"
    return jsonify({"configured": configured, "provider": provider})

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
    ext = Path(file.filename).suffix.lower()
    name = Path(file.filename).stem
    safe_name = re.sub(r"[^\w\u4e00-\u9fff_-]", "_", name)

    if ext in (".jpg", ".jpeg", ".png", ".gif"):
        # Save image for later analysis
        save_path = IMAGE_DIR / f"{safe_name}{ext}"
        file.save(str(save_path))
        return jsonify({"name": safe_name, "chunks": 0, "message": f"已保存图片 {safe_name}{ext}，可以问我图片内容"})
    elif ext == ".pdf":
        # Extract text from PDF and add to KB
        try:
            from pypdf import PdfReader
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            save_path = DOC_DIR / f"{safe_name}.txt"
            save_path.write_text(text, encoding="utf-8")
            chunks = kb.add_document(str(save_path))
            return jsonify({"name": safe_name, "chunks": chunks, "message": t_key("upload_success", lang, name=safe_name, chunks=chunks)})
        except ImportError:
            return jsonify({"error": "PDF 解析需要安装 pypdf: pip install pypdf"}), 400
    else:
        # .txt / .md
        text = file.read().decode("utf-8", errors="replace")
        save_path = DOC_DIR / f"{safe_name}.txt"
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

# ====== 图片 API ======

@app.route("/uploads/images", methods=["GET"])
def list_images():
    images = []
    for f in sorted(IMAGE_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif"):
            images.append({"name": f.name, "size": f.stat().st_size})
    return jsonify(images)

from flask import send_from_directory

@app.route("/uploads/images/<filename>")
def serve_image(filename: str):
    return send_from_directory(str(IMAGE_DIR), filename)

# ====== 流式聊天 API =======

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    client_msgs = data.get("messages", [])
    lang = data.get("lang", "zh")

    if agent is None:
        return jsonify({"error": "API key not configured. Please set your API key first."}), 400

    try:
        lc_msgs = [SystemMessage(content=get_system_prompt(lang))] + [to_langchain(m) for m in client_msgs]
        result = agent.invoke({"messages": lc_msgs})
        agent_msgs = result["messages"]
        known_count = len(client_msgs)
        new_msgs = agent_msgs[known_count + 1:]
        new_client = [to_client(m) for m in new_msgs]

        return jsonify({
            "reply": clean_response(agent_msgs[-1].content or ""),
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

    if agent is None:
        def generate():
            yield f"event: error\ndata: {json.dumps({'error': 'API key not configured. Please set your API key first.'})}\n\n"
        return Response(generate(), mimetype="text/event-stream")

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
                            yield f"event: token\ndata: {json.dumps({'text': clean_response(msg.content)})}\n\n"

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

def _get_html(force=False):
    global _HTML_CACHE
    if _HTML_CACHE is None or force:
        tmpl = pathlib.Path(__file__).parent / "templates" / "index.html"
        if tmpl.exists():
            _HTML_CACHE = tmpl.read_text(encoding="utf-8")
        else:
            _HTML_CACHE = "<html><body><h1>Template not found</h1></body></html>"
    return _HTML_CACHE

HTML = _get_html()

@app.route("/")
def index():
    return _get_html(force=True)

if __name__ == "__main__":
    print("=" * 50)
    print("助手  已启动")
    print("http://localhost:5000")
    print("左侧可上传文档")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)