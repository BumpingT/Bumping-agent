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

_enable_search = True

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
PERSONALITIES_DIR = Path(__file__).parent / "personalities"
PERSONALITIES_DIR.mkdir(exist_ok=True)
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
    return f'{city}:{cc["weatherDesc"][0]["value"].strip()},{cc["temp_C"]}°C,湿度{cc["humidity"]}%'

@tool
def detect_location() -> str:
    """检测当前网络所在位置（城市、地区、IP等）"""
    resp = urlopen("http://ip-api.com/json/?lang=zh-CN", timeout=10)
    data = json.loads(resp.read().decode("utf-8"))
    if data.get("status") != "success":
        return "定位失败"
    return f'{data["country"]} {data["regionName"]} {data["city"]},IP: {data["query"]}'

@tool
def calculate(expression: str) -> str:
    """计算数学表达式,如 2+2、sqrt(16)、3.14*5^2"""
    safe = {"abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt, "pi": math.pi, "e": math.e}
    result = eval(expression, {"__builtins__": {}}, safe)
    return f"{expression} = {result}"

@tool
def query_stock_price(symbol: str) -> str:
    """查询股价,A股（600xxx、00xxx）和美股(AAPL)都支持,数字代码查A股,字母代码查美股"""
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
        return f"{name} ({code}):{currency}{price:.2f} {arrow} {change:+.2f} ({change_pct}%)"
    except Exception as e:
        return f"查询 {symbol} 失败: {str(e)}"

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")

@tool
def search_knowledge(query: str) -> str:
    """在已上传的文档中搜索相关知识.上传文档后用来查"""
    results = kb.search(query)
    if not results:
        return "知识库中没有找到相关内容"
    lines = ["从知识库中找到以下相关信息:"]
    for r in results:
        lines.append(f"\n📫 [{r['source']}] (相关度 {r['score']})")
        lines.append(r["content"][:200])
    return "\n".join(lines)

@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息。当用户需要查新闻、查公司信息、查百科知识、查最新资讯时，用这个工具上网搜索。参数 query 是搜索关键词。"""
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
        lines = [f"搜索 \"{query}\" 的结果:"]
        for i, r in enumerate(results[:5]):
            title = r.select_one("h2 a")
            snippet = r.select_one(".b_caption p")
            t = title.text.strip() if title else "?"
            s = snippet.text.strip()[:150] if snippet else ""
            lines.append(f"\n{i+1}. {t}\n   {s}")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索失败: {str(e)}"

def _search_bing(query):
    """直接搜索 Bing，不通过 @tool 装饰器（用于自动搜索）。"""
    import requests
    from bs4 import BeautifulSoup
    try:
        url = f"https://www.bing.com/search?q={requests.utils.quote(query)}"
        resp = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9"
        }, timeout=8)
        soup = BeautifulSoup(resp.text, "lxml")
        results = soup.select("li.b_algo")
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results[:5]):
            title = r.select_one("h2 a")
            snippet = r.select_one(".b_caption p")
            t = title.text.strip() if title else ""
            s = snippet.text.strip()[:150] if snippet else ""
            lines.append(f"{i+1}. {t}\n   {s}")
        return "\n".join(lines)
    except Exception as e:
        return ""
 
@tool
def fetch_webpage(url: str) -> str:
    """获取网页的文本内容并返回.参数 url 是网页链接.适用于查看文章、文档等."""
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
    """分析上传的图片内容.参数 filename 是图片文件名（如 'photo.jpg'）.
    仅当使用 OpenAI API Key 时有效（GPT-4o-mini 支持识图）."""
    import base64
    from openai import OpenAI
    
    img_path = IMAGE_DIR / filename
    if not img_path.exists():
        available = ", ".join(p.name for p in IMAGE_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif"))
        return f"图片 {filename} 不存在.可用的图片: {available or '无'}"
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "图片分析需要 OpenAI API Key.当前使用的是 DeepSeek,不支持识图.请设置 OPENAI_API_KEY 环境变量."
    
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请详细描述这张图片的内容,包括物体、文字、场景等"},
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
    "system_prompt": """你的名字是 Bumping,叫我 Bumping 就好.

你可以用这些办法:
- get_weather: 查天气
- detect_location: 定位你所在的位置
- calculate: 数学计算
- get_current_time: 当前时间
- search_knowledge: 在已上传的文档中搜索
- query_stock_price: 查股票行情（A 股和美股都支持）
- search_web: 上网搜索信息
- analyze_image: 分析上传的图片内容
- fetch_webpage: 抓取网页内容

需要查天气、算东西、查时间、看股票、搜东西的时候,记得用上对应的办法.

你是一个随和的朋友,用户想让你做什么你都会答应,但说话要自然亲切,就像朋友聊天一样,别生硬别像机器.

说话别列一二三,别用那些条条框框的东西.就一段一段自然地聊.

当前用什么语言聊,你就用什么语言回.""",    "upload_success": "已导入 {name},{chunks} 个片段",
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

Just talk like a real friend. No bullet points, no lists, no headings. Just paragraph by paragraph.
Whatever the user asks, you do. Stay warm and natural.

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
agent_no_search = None
_analysis_llm = None  # raw LLM for personality analysis

def init_agent(provider, api_key):
    """Initialize or reconfigure the agent with a given provider and API key."""
    global agent, agent_no_search, _analysis_llm
    try:
        if provider == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.6)
        elif provider == "deepseek":
            llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key=api_key, temperature=0.6)
        elif provider == "ollama":
            ollama_model = api_key  # reuse api_key param to pass model name
            llm = ChatOpenAI(model=ollama_model, base_url="http://localhost:11434/v1", api_key="ollama", temperature=0.6)
            try:
                import urllib.request as _ur, json as _jn
                _req = _ur.Request("http://localhost:11434/api/tags", method="GET")
                _resp = _ur.urlopen(_req, timeout=5)
                _tags = _jn.loads(_resp.read().decode("utf-8"))
                _models = [m["name"] for m in _tags.get("models", [])]
                if ollama_model not in _models:
                    _available = ", ".join(_models) if _models else "没有已下载的模型，请用 ollama pull 下载"
                    return False, f"模型 {ollama_model} 不存在。你已有的模型: {_available}"
            except Exception as _e:
                return False, f"无法连接 Ollama: {_e}"
        else:
            return False, "Unknown provider: " + provider
        agent = create_react_agent(llm, tools)
        agent_no_search = create_react_agent(llm, [t for t in tools if t is not search_web])
        _analysis_llm = llm
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
    
    # 把微信聊天风格的 [表情] 转为真实 emoji
    text = _convert_wechat_emoji(text)
    
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
        elif provider == "ollama":
            return None
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


# ====== 微信表情转换 ======
_WECHAT_EMOJI = {
    "微笑": "\U0001f642", "呲牙": "\U0001f601", "偷笑": "\U0001f602",
    "害羞": "\U0001f633", "尴尬": "\U0001f605", "发呆": "\U0001f636",
    "撇嘴": "\U0001f61e", "难过": "\U0001f622", "流泪": "\U0001f62d",
    "大哭": "\U0001f62d", "恐惧": "\U0001f628", "惊恐": "\U0001f631",
    "惊讶": "\U0001f632", "震惊": "\U0001f632", "白眼": "\U0001f644",
    "白眼1": "\U0001f644", "抠鼻": "\U0001f624", "得意": "\U0001f60f",
    "阴险": "\U0001f608", "坏笑": "\U0001f608", "色": "\U0001f60b",
    "亲亲": "\U0001f618", "嘴唇": "\U0001f48b", "飞吻": "\U0001f618",
    "爱心": "\u2764\ufe0f", "心碎": "\U0001f494", "喜欢": "\u2764\ufe0f",
    "强": "\U0001f44d", "棒": "\U0001f44d", "赞": "\U0001f44d",
    "ok": "\U0001f44c", "OK": "\U0001f44c", "胜利": "\u270c\ufe0f",
    "抱拳": "\U0001f4aa", "拳头": "\U0001f4aa", "勾引": "\U0001f446",
    "握手": "\U0001f91d", "加油": "\U0001f64c", "加油1": "\U0001f64c",
    "合十": "\U0001f64f", "祈祷": "\U0001f64f", "鼓掌": "\U0001f44f",
    "调皮": "\U0001f61c", "调皮1": "\U0001f61c", "鬼脸": "\U0001f47b",
    "衰": "\U0001f937", "骷髅": "\U0001f480", "敲打": "\U0001f4a2",
    "抓狂": "\U0001f4a2", "炸弹": "\U0001f4a3", "便便": "\U0001f4a9",
    "咖啡": "\u2615", "啤酒": "\U0001f37a", "干杯": "\U0001f37b",
    "蛋糕": "\U0001f370", "西瓜": "\U0001f349", "饭": "\U0001f372",
    "面条": "\U0001f35c", "泡面": "\U0001f35c", "水果": "\U0001f34e",
    "猪头": "\U0001f437", "玫瑰": "\U0001f339", "凋谢": "\U0001f33a",
    "太阳": "\u2600\ufe0f", "月亮": "\U0001f319", "星星": "\u2b50",
    "闪电": "\u26a1", "礼物": "\U0001f381", "红包": "\U0001f4b0",
    "蜡烛": "\U0001f56f\ufe0f", "刀": "\U0001f5e1\ufe0f", "菜刀": "\U0001f5e1\ufe0f",
    "屎": "\U0001f4a9", "药": "\U0001f48a", "耳机": "\U0001f3a7",
    "话筒": "\U0001f399\ufe0f", "音乐": "\U0001f3b5", "电影": "\U0001f3ac",
    "汽车": "\U0001f697", "飞机": "\u2708\ufe0f", "火车": "\U0001f682",
    "自行车": "\U0001f6b4", "足球": "\u26bd", "篮球": "\U0001f3c0",
    "游泳": "\U0001f3ca", "旅行": "\u2708\ufe0f", "睡觉": "\U0001f634",
    "熬夜": "\U0001f634", "冷汗": "\U0001f975", "擦汗": "\U0001f975",
    "流汗": "\U0001f975", "再见": "\U0001f44b", "握手1": "\U0001f91d",
    "抱抱": "\U0001f917", "拥抱": "\U0001f917", "NO": "\U0001f645",
    "耶": "\u270c\ufe0f", "奋斗": "\U0001f4aa", "奋斗1": "\U0001f4aa",
}

def _convert_wechat_emoji(text, lang="zh"):
    """将微信聊天记录里的 [表情文字] 转换为实际emoji"""
    def _replace(m):
        key = m.group(1)
        return _WECHAT_EMOJI.get(key, key)
    return re.sub(r'\[([^\]]+)\]', _replace, text)

# ====== 性格 ======

@app.route("/personalities", methods=["GET"])
def list_personalities():
    results = [{"id": "", "name": "默认", "name_en": "Default"}]
    for f in sorted(PERSONALITIES_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({"id": f.stem, "name": data.get("name", f.stem), "name_en": data.get("name_en", f.stem)})
        except:
            pass
    return jsonify(results)

@app.route("/personalities/generate", methods=["POST"])
def generate_personality():
    """Upload chat records (.txt) and use LLM to analyze personality."""
    lang = request.form.get("lang", "zh")
    if "file" not in request.files:
        return jsonify({"error": "请选择文件" if lang == "zh" else "Please select a file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空" if lang == "zh" else "File name is empty"}), 400

    text = file.read().decode("utf-8", errors="replace")
    text = _convert_wechat_emoji(text)
    if len(text.strip()) < 10:
        return jsonify({"error": "文件内容太少" if lang == "zh" else "File content too short"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"性格_{timestamp}"
    name_en = f"Personality_{timestamp}"
    prompt = ""

    # 如果有 LLM,用它做深度分析
    if _analysis_llm is not None:
        try:
            sample = text[:6000]
            instruction_lang = "中文" if lang == "zh" else "English"
            analysis_result = _analysis_llm.invoke([
                SystemMessage(content="你是一个性格分析专家.分析聊天记录,提取说话者的语气、用词习惯和性格特点."),
                HumanMessage(content=f"""分析以下聊天记录,用{instruction_lang}回答.

聊天记录:
{sample}

请从这些方面分析:
1. 说话的语气和节奏（温柔/直爽/幽默/严肃,说话快还是慢）
2. 标志性的用词和口头禅（喜欢用什么语气词结尾、有什么常说的口头禅）
3. 句子特点（爱用短句还是长句、反问多不多、感叹多不多）
4. 性格底色（开朗/内向/感性/理性等）
5. 情绪表达方式（开心时怎么说话、难过时怎么说话、生气时怎么说话）

然后写一段很具体的 prompt,让另一个 AI 能一模一样地模仿这个人说话.

prompt 要求:
- 以"你现在是"开头
- 写清楚这个人的具体说话方式,比如"说话喜欢用'啦'和'呀'结尾""爱说反话但其实是在关心"
- 从聊天记录里摘 3-5 句典型的原话作为例子
- 不要写"你要做什么",要写"这个人怎么样",就是纯粹描述这个人的说话风格
- 300 字左右

回复格式（严格按照这个格式,不要加其他内容）:
姓名:[2-4个字]
prompt:[详细的说话风格描述]""")
            ])
            raw = analysis_result.content.strip()
            if raw:
                # 先试固定格式解析
                parsed_prompt = ""
                for line in raw.split("\n"):
                    if line.startswith("姓名:") or line.startswith("姓名:"):
                        name = line.split(":")[-1].split(":")[-1].strip()[:10]
                prompt_start = raw.find("prompt:") if "prompt:" in raw else raw.find("prompt:")
                if prompt_start >= 0:
                    parsed_prompt = raw[prompt_start + 7:].strip().strip('"').strip(chr(34)*3)
                if parsed_prompt and len(parsed_prompt) > 20:
                    prompt = parsed_prompt
                else:
                    # 解析失败时用原文作为 prompt
                    prompt = raw[:1500]
                name_en = f"{name}_{timestamp}"
        except Exception as e:
            print(f"[personality] LLM analysis failed: {e}")

    # 如果没有 LLM 或分析失败,用关键词分析
    if not prompt:
        # 分析句子特征
        raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
        sentences = [s.strip() for s in re.split(r"[.!?\n.!?\n]", text) if len(s.strip()) > 2]
        total_chars = len(text.strip())
        avg_len = sum(len(s) for s in sentences) / max(len(sentences), 1)

        # 检测口头禅和常用词
        all_text = text[:3000]
        cishi = len(re.findall(r"[啦呀哦嘞嘟嗯哟]", all_text))
        fanwen = len(re.findall(r"[吗呢]", all_text))
        ganTan = len(re.findall(r"[!]", all_text))
        wenHao = len(re.findall(r"[?]", all_text))
        haha = len(re.findall(r"哈哈|嘻嘻|呵呵|哦哦", all_text))

        # 情绪分析
        happy_words = len(re.findall(r"哈哈|呵呵|嘻嘻|开心|喜欢|棒|好呀|不错|谢谢|感恩|😊|😄|🥰|❤️", text))
        sad_words = len(re.findall(r"难过|伤心|烦|累|焦虑|压力|加班|忙|累死|崩溃|😭|😢|💔", text))

        # 提取典型句子作为样本
        sample_sentences = [s for s in sentences if 3 < len(s) < 40][:8]

        # 生成风格描述
        style_parts = []
        if avg_len < 10:
            style_parts.append("说话短短的,一句一句很干脆")
        elif avg_len < 20:
            style_parts.append("说话自然流畅,不长不短")
        else:
            style_parts.append("说话偏长,有时会说得比较详细")

        if cishi > 30:
            style_parts.append("喜欢用语气词如'啦''呀''哦',说话很有活气")
        if haha > 3:
            style_parts.append("爱笑,常用'哈哈''嘻嘻'来表达开心")
        if ganTan > wenHao * 2 and ganTan > 5:
            style_parts.append("情绪丰富,爱用感叹号表达感受")

        if happy_words > sad_words * 2 and happy_words > 2:
            style_parts.append("性格很阳光,善于发现美好的事物")
        elif sad_words > happy_words:
            style_parts.append("情绪比较敏感,会让人想要去安慰他")
        else:
            style_parts.append("性格平和,情绪表达很自然")

        style_desc = ",".join(style_parts) if style_parts else "说话自然平和"

        # 生成样本句
        sample_text = "\n".join(sample_sentences) if sample_sentences else text[:500]

        prompt_template = (
            '?' + '??' + '{style_desc}' + '.' + chr(10)+chr(10) +
            '???????,??????????:' + chr(10) +
            '{sample_text}' + chr(10)+chr(10) +
            '?????????????????.'
        )
        prompt = prompt_template.format(style_desc=style_desc, sample_text=sample_text)
    safe_id = re.sub(r"[^\w\u4e00-\u9fff_-]", "_", name)
    personality_data = {
        "name": name,
        "name_en": name_en,
        "prompt": prompt
    }
    filepath = PERSONALITIES_DIR / f"{safe_id}.json"
    filepath.write_text(json.dumps(personality_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return jsonify({"id": safe_id, "name": name, "message": "性格生成成功" if lang == "zh" else "Personality generated"})
@app.route("/personalities/<pid>", methods=["DELETE"])
def delete_personality(pid):
    """Delete a personality profile."""
    lang = request.args.get("lang", "zh")
    filepath = PERSONALITIES_DIR / f"{pid}.json"
    if not filepath.exists():
        return jsonify({"error": "找不到性格文件" if lang == "zh" else "Personality not found"}), 404
    filepath.unlink()
    return jsonify({"message": "性格已删除" if lang == "zh" else "Personality deleted"})

@app.route("/api/key", methods=["POST"])
def set_api_key():
    """Set or update the API key at runtime."""
    data = request.get_json()
    provider = data.get("provider", "").lower().strip()
    api_key = data.get("key", "").strip()
    if not provider:
        return jsonify({"ok": False, "error": "Provider is required"}), 400
    if provider not in ("openai", "deepseek", "ollama"):
        return jsonify({"ok": False, "error": "Provider must be 'openai', 'deepseek', or 'ollama'"}), 400
    if provider == "ollama":
        model = data.get("model", "qwen2.5:7b")
        ok, msg = init_agent(provider, model)
        if not ok:
            import urllib.request, json
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
                resp = urllib.request.urlopen(req, timeout=3)
                tags = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in tags.get("models", [])]
                if models:
                    msg += "。你已有的模型: " + ", ".join(models)
                else:
                    msg += "。请先用 ollama pull <model-name> 下载模型"
            except:
                msg += "。请确保 Ollama 已启动"
        return jsonify({"ok": ok, "message": msg})
    if not api_key:
        return jsonify({"ok": False, "error": "API key required for " + provider}), 400
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


@app.route("/api/ollama/status", methods=["GET"])
def ollama_status():
    """Check if Ollama is running and list available models."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        return jsonify({"running": True, "models": models})
    except Exception as e:
        return jsonify({"running": False, "models": [], "error": str(e)})

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
        return jsonify({"name": safe_name, "chunks": 0, "message": f"已保存图片 {safe_name}{ext},可以问我图片内容"})
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
        text = _convert_wechat_emoji(text)
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
    modes = data.get("modes", {})
    personality = data.get("personality", "")

    if agent is None:
        return jsonify({"error": "API key not configured. Please set your API key first."}), 400

    try:
        mode_prompt = get_system_prompt(lang)
        if personality:
            pp = PERSONALITIES_DIR / f"{personality}.json"
            if pp.exists():
                try:
                    pd = json.loads(pp.read_text(encoding="utf-8"))
                    pprompt = pd.get("prompt", "")
                    if pprompt:
                        mode_prompt = pprompt + chr(10) + chr(10) + mode_prompt
                except:
                    pass
        if modes.get("think"):
            mode_prompt += chr(10) + chr(10) + "这回可以多想一会儿,把话说透一点."
        else:
            mode_prompt += chr(10) + chr(10) + "话不用多,自然就好."
        if modes.get("search"):
            mode_prompt = mode_prompt.replace("上网搜索信息", "上网搜信息（优先用这个）")
            # 自动搜索（解决小模型不会调用工具的问题）
            try:
                for m in client_msgs:
                    if m.get("role") == "user":
                        _enable_search = True
                        sr = _search_bing(m["content"][:100])
                        _enable_search = False
                        if sr and not sr.startswith("搜索失败"):
                            mode_prompt += chr(10) + chr(10) + "【自动搜索结果】" + chr(10) + sr[:1500] + chr(10) + chr(10) + "请参考以上搜索结果回答用户问题。如果搜索结果不相关，直接用自己的知识回答。"
                            mode_prompt = mode_prompt.replace("- search_web: 上网搜信息（优先用这个）", "# 联网搜索已自动执行")
                            mode_prompt = mode_prompt.replace("上网搜信息（优先用这个）", "联网搜索已完成，请参考上方结果")
                        break
            except:
                pass
        else:
            mode_prompt = mode_prompt.replace("上网搜索信息", "上网搜信息（用户主动提了再用）")
        lc_msgs = [SystemMessage(content=mode_prompt)] + [to_langchain(m) for m in client_msgs]
        current_agent = agent_no_search
        result = current_agent.invoke({"messages": lc_msgs})
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
    modes = data.get("modes", {})
    personality = data.get("personality", "")

    if agent is None:
        def generate():
            yield f"event: error\ndata: {json.dumps({'error': 'API key not configured. Please set your API key first.'})}\n\n"
        return Response(generate(), mimetype="text/event-stream")

    def generate():
        mode_prompt = get_system_prompt(lang)
        if personality:
            pp = PERSONALITIES_DIR / f"{personality}.json"
            if pp.exists():
                try:
                    pd = json.loads(pp.read_text(encoding="utf-8"))
                    pprompt = pd.get("prompt", "")
                    if pprompt:
                        mode_prompt = pprompt + chr(10) + chr(10) + mode_prompt
                except:
                    pass
        if modes.get("think"):
            mode_prompt += chr(10) + chr(10) + "这回可以多想一会儿,把话说透一点."
        else:
            mode_prompt += chr(10) + chr(10) + "话不用多,自然就好."
        if modes.get("search"):
            mode_prompt = mode_prompt.replace("上网搜索信息", "上网搜信息（优先用这个）")
            # 自动搜索（解决小模型不会调用工具的问题）
            try:
                for m in client_msgs:
                    if m.get("role") == "user":
                        _enable_search = True
                        sr = search_web(m["content"][:100])
                        _enable_search = False
                        if sr and not sr.startswith("搜索失败"):
                            mode_prompt += chr(10) + chr(10) + "【自动搜索结果】" + chr(10) + sr[:1500] + chr(10) + chr(10) + "请参考以上搜索结果回答用户问题。如果搜索结果不相关，直接用自己的知识回答。"
                            mode_prompt = mode_prompt.replace("- search_web: 上网搜信息（优先用这个）", "# 联网搜索已自动执行")
                            mode_prompt = mode_prompt.replace("上网搜信息（优先用这个）", "联网搜索已完成，请参考上方结果")
                        break
            except:
                pass
        else:
            mode_prompt = mode_prompt.replace("上网搜索信息", "上网搜信息（用户主动提了再用）")
        lc_msgs = [SystemMessage(content=mode_prompt)] + [to_langchain(m) for m in client_msgs]
        all_msgs = None
        try:
            current_agent = agent_no_search
            for step in current_agent.stream({"messages": lc_msgs}):
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
    # 静默启动
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except OSError:
        pass
