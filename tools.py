# ====== AI 工具函数 ======
import json, os, math, re, hashlib, glob, time
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote

from langchain_core.tools import tool

from config import IMAGE_DIR
from kb import kb
import state

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
- github_list_issues / github_get_issue: 查看 GitHub Issues
- github_get_file: 读取 GitHub 仓库文件内容
- github_search_code: 搜索 GitHub 仓库代码
- github_list_repos: 列出 GitHub 用户的仓库
- github_list_branches: 列出仓库的分支
- github_list_pull_requests: 查看 Pull Requests

需要查天气、算东西、查时间、看股票、搜东西的时候,记得用上对应的办法.

你是一个随和的朋友,用户想让你做什么你都会答应,但说话要自然亲切,就像朋友聊天一样,别生硬别像机器.

说话别列一二三,别用那些条条框框的东西.就一段一段自然地聊.



常见场景的工作流程（遇到这些情况直接做，不用问用户确认）：
- 用户说"导入文件"/"把文件导入数据库"：先 read_uploaded_file 读原始内容，再用 db_import_csv 导入到新表。表名用文件名去掉后缀和多余文字。
- 用户说"查一下XX表"/"看看XX表的数据"/"展示XX表"：用 db_query 从数据库查真实数据展示。绝对禁止凭记忆重新生成表格。
- 用户说"帮我读这个文件并生成表格"/"读取这个文件并做成表格"/"把文件内容展示出来"：严格三步走——先 read_uploaded_file 读原始内容，再 db_import_csv 导入数据库，最后 db_query 从数据库查出展示。缺一步都不行，绝不允许用记忆直接编表格。
- 用户说"加一条数据到XX表"：用 db_insert。
- 用户说"导出"：查出数据后用户会看到"导出 Excel"按钮。
- 用户问仓库/代码/PR/README：用 github_list_repos / github_get_file / github_search_code / github_list_pull_requests。
- 用户说"搜索"但没指定搜什么：默认用 search_knowledge 搜已上传的文档。
- 用户说"帮我看看这个文件"/"读一下文件"：用 read_uploaded_file 读文件内容并严格按原文展示。

⚠ 严禁编造数据的铁律：任何时候要展示数据、生成表格、查询结果，都必须走 read_uploaded_file → db_import_csv → db_query 的完整数据库链路，绝对不允许用自身对数据的记忆或理解来"重新生成"或"概括"数据表格。即使你刚读完文件内容，也必须导入数据库再查询展示。如果用户发现数据对不上，任务直接判定失败。

记住：用户要的是省心，所有能自动完成的步骤直接做，不要问用户。
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




@tool
def wecom_send_message(user_id: str, content: str) -> str:
    """Wecom Send Message."""
    import state
    if not state.wecom_corpid or not state.wecom_corpsecret:
        return "WeCom not configured"
    import urllib.request, json as _json
    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={state.wecom_corpid}&corpsecret={state.wecom_corpsecret}"
    resp = urllib.request.urlopen(url, timeout=10)
    data = _json.loads(resp.read())
    if "access_token" not in data:
        return f"Token failed: {data.get('errmsg', '')}"
    token = data["access_token"]
    msg_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    body = _json.dumps({"touser": user_id, "msgtype": "text", "text": {"content": content}}).encode()
    req = urllib.request.Request(msg_url, data=body, headers={"Content-Type": "application/json"})
    resp2 = urllib.request.urlopen(req, timeout=10)
    result = _json.loads(resp2.read())
    if result.get("errcode") == 0:
        return "Sent"
    return f"Failed: {result.get('errmsg', '')}"

@tool
def wecom_list_users() -> str:
    """Wecom List Users."""
    import state
    if not state.wecom_corpid or not state.wecom_corpsecret:
        return "WeCom not configured"
    import urllib.request, json as _json
    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={state.wecom_corpid}&corpsecret={state.wecom_corpsecret}"
    resp = urllib.request.urlopen(url, timeout=10)
    data = _json.loads(resp.read())
    if "access_token" not in data:
        return f"Token failed: {data.get('errmsg', '')}"
    token = data["access_token"]
    user_url = f"https://qyapi.weixin.qq.com/cgi-bin/user/list?access_token={token}&department_id=1"
    resp2 = urllib.request.urlopen(user_url, timeout=10)
    users = _json.loads(resp2.read())
    if users.get("errcode") != 0:
        return f"Failed: {users.get('errmsg', '')}"
    members = users.get("userlist", [])
    if not members:
        return "No members"
    names = [f"{u['name']}({u['userid']})" for u in members]
    return "\n".join(names)

@tool
def dingtalk_send_message(user_id: str, content: str) -> str:
    """Dingtalk Send Message."""
    import state
    if not state.dingtalk_app_key or not state.dingtalk_app_secret:
        return "DingTalk not configured"
    import urllib.request, json as _json, time
    url = "https://oapi.dingtalk.com/gettoken"
    params = f"appkey={state.dingtalk_app_key}&appsecret={state.dingtalk_app_secret}"
    req_url = f"{url}?{params}"
    resp = urllib.request.urlopen(req_url, timeout=10)
    data = _json.loads(resp.read())
    if "access_token" not in data:
        return f"Token failed: {data.get('errmsg', '')}"
    token = data["access_token"]
    msg_url = f"https://oapi.dingtalk.com/topapi/message/corpconversation/asyncsend_v2?access_token={token}"
    body = _json.dumps({
        "agent_id": state.dingtalk_app_key,
        "userid_list": user_id,
        "msg": {"msgtype": "text", "text": {"content": content}}
    }).encode()
    req = urllib.request.Request(msg_url, data=body, headers={"Content-Type": "application/json"})
    resp2 = urllib.request.urlopen(req, timeout=10)
    result = _json.loads(resp2.read())
    if result.get("errcode") == 0:
        return "Sent"
    return f"Failed: {result.get('errmsg', '')}"

@tool
def github_list_issues(repo: str, state_filter: str = "open") -> str:
    """Github List Issues."""
    import state
    if not state.github_token:
        return "GitHub not configured"
    import urllib.request, json as _json
    url = f"https://api.github.com/repos/{repo}/issues?state={state_filter}&per_page=10"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {state.github_token}", "Accept": "application/vnd.github.v3+json", "User-Agent": "BumpingAgent"})
    resp = urllib.request.urlopen(req, timeout=15)
    issues = _json.loads(resp.read())
    if isinstance(issues, dict) and "message" in issues:
        return f"Failed: {issues['message']}"
    if not issues:
        return "No issues"
    result = []
    for i in issues:
        labels = ", ".join([l["name"] for l in i.get("labels", [])])
        result.append(f"#{i['number']} {i['title']} [{i['state']}] {labels}\n  {i['html_url']}")
    return "\n---\n".join(result)

@tool
def github_get_issue(repo: str, issue_number: int) -> str:
    """Github Get Issue."""
    import state
    if not state.github_token:
        return "GitHub not configured"
    import urllib.request, json as _json
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {state.github_token}", "Accept": "application/vnd.github.v3+json", "User-Agent": "BumpingAgent"})
    resp = urllib.request.urlopen(req, timeout=15)
    issue = _json.loads(resp.read())
    if "message" in issue:
        return f"Failed: {issue['message']}"
    labels = ", ".join([l["name"] for l in issue.get("labels", [])])
    body = (issue.get("body") or "")[:500]
    return f"#{issue['number']} {issue['title']} [{issue['state']}]\nLabels: {labels}\nCreated: {issue['created_at']}\n\n{body}\n\n{issue['html_url']}"

@tool
def feishu_query(app_token: str, table_id: str, query: str = "") -> str:
    """Feishu Query."""
    import state, urllib.request, json as _json
    if not state.feishu_app_id or not state.feishu_app_secret:
        return "Feishu not configured"
    token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    body = _json.dumps({"app_id": state.feishu_app_id, "app_secret": state.feishu_app_secret}).encode()
    req = urllib.request.Request(token_url, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=10)
    data = _json.loads(resp.read())
    if "tenant_access_token" not in data:
        return "Token failed"
    token = data["tenant_access_token"]
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/search"
    body2 = _json.dumps({}) if not query else query
    req2 = urllib.request.Request(url, data=body2.encode() if body2 else None, headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"}, method="POST")
    resp2 = urllib.request.urlopen(req2, timeout=15)
    data2 = _json.loads(resp2.read())
    if data2.get("code") != 0:
        return f"Query failed: {data2.get('msg', '')}"
    records = data2.get("data", {}).get("items", [])
    if not records:
        return "No records"
    result = []
    for r in records[:20]:
        fields = r.get("fields", {})
        result.append(_json.dumps(fields, ensure_ascii=False))
    return "\n---\n".join(result)

@tool
def feishu_add(app_token: str, table_id: str, fields: str) -> str:
    """Feishu Add."""
    import state, urllib.request, json as _json
    if not state.feishu_app_id or not state.feishu_app_secret:
        return "Feishu not configured"
    token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    body = _json.dumps({"app_id": state.feishu_app_id, "app_secret": state.feishu_app_secret}).encode()
    req = urllib.request.Request(token_url, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=10)
    data = _json.loads(resp.read())
    if "tenant_access_token" not in data:
        return "Token failed"
    token = data["tenant_access_token"]
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
    fields_dict = _json.loads(fields) if isinstance(fields, str) else fields
    body2 = _json.dumps({"fields": fields_dict})
    req2 = urllib.request.Request(url, data=body2.encode(), headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"}, method="POST")
    resp2 = urllib.request.urlopen(req2, timeout=15)
    data2 = _json.loads(resp2.read())
    if data2.get("code") != 0:
        return f"Add failed: {data2.get('msg', '')}"
    return "Record added: " + data2.get("data", {}).get("record", {}).get("record_id", "")

@tool
def db_create_table(table_name: str, columns: str) -> str:
    """Db Create Table."""
    import sqlite3, json as _json, os
    os.makedirs(os.path.dirname(state.db_path), exist_ok=True)
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        cols = _json.loads(columns)
        col_sql = ", ".join([f'"{col["name"]}" TEXT' for col in cols])
        c.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_sql})')
        conn.commit()
        return f'Table "{table_name}" created with {len(cols)} fields'
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()

@tool
def db_insert(table_name: str, data: str) -> str:
    """Db Insert."""
    import sqlite3, json as _json
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        d = _json.loads(data)
        cols = ", ".join([f'"{k}"' for k in d.keys()])
        vals = ", ".join(["?" for _ in d])
        c.execute(f'INSERT INTO "{table_name}" ({cols}) VALUES ({vals})', list(d.values()))
        conn.commit()
        return f"Inserted record ID: {c.lastrowid}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()
@tool
def db_insert_batch(table_name: str, data: str) -> str:
    """Db Insert Batch - 批量插入多条数据。data 为 JSON 数组，如 [{"col1":"val1","col2":"val2"},{"col1":"val3","col2":"val4"}]。"""
    import sqlite3, json as _json
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        rows = _json.loads(data)
        if not isinstance(rows, list):
            return "Error: data 必须是 JSON 数组"
        if not rows:
            return "Error: 数据为空"
        keys = list(rows[0].keys())
        cols = ", ".join([f'"{k}"' for k in keys])
        vals = ", ".join(["?" for _ in keys])
        count = 0
        for row in rows:
            c.execute(f'INSERT INTO "{table_name}" ({cols}) VALUES ({vals})', [row.get(k) for k in keys])
            count += 1
        conn.commit()
        return f"成功插入 {count} 条记录"
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()
@tool
def db_import_csv(table_name: str, csv_text: str) -> str:
    """直接从 CSV 文本导入数据到数据库表。自动识别表头作为列名，自动建表并插入所有行。csv_text 为 CSV 格式文本，第一行为列名，后续为数据行，逗号分隔。"""
    import sqlite3, json as _json, csv, io
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        # 自动找到 CSV 数据区域（跳过标题、说明等非 CSV 行）
        lines = csv_text.split('\n')
        csv_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('=') or stripped.startswith('-') or stripped.startswith('#'):
                continue
            if ',' in stripped and (not csv_lines or stripped.count(',') == csv_lines[0].count(',')):
                csv_lines.append(stripped)
        if not csv_lines:
            return "未找到 CSV 数据"
        reader = csv.DictReader(io.StringIO('\n'.join(csv_lines)))
        rows = list(reader)
        if not rows:
            return "数据为空"
        cols = list(rows[0].keys())
        col_defs = ", ".join([f'"{k}" TEXT' for k in cols])
        c.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})')
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join([f'"{k}"' for k in cols])
        count = 0
        for row in rows:
            values = [str(row.get(k, "")).strip() for k in cols]
            c.execute(f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})', values)
            count += 1
        conn.commit()
        return f"成功创建表 {table_name}，导入 {count} 条记录。列: {', '.join(cols)}"
    except Exception as e:
        return f"导入失败: {e}"
    finally:
        conn.close()
@tool
def read_uploaded_file(name: str) -> str:
    """读取已上传文件的内容。name 为文件名（支持模糊匹配）。返回文件原文，可用于 db_import_csv 导入。"""
    import os, glob
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_docs")
    matches = []
    for f in os.listdir(docs_dir):
        if name.lower() in f.lower():
            matches.append(f)
    if not matches:
        return f"未找到文件（含 '{name}'）。已有文件: {', '.join(os.listdir(docs_dir))}"
    fpath = os.path.join(docs_dir, matches[0])
    with open(fpath, 'r', encoding='utf-8') as f:
        return f.read()




@tool
def db_query(table_name: str, conditions: str = "") -> str:
    """Db Query."""
    import sqlite3, json as _json
    conn = sqlite3.connect(state.db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    try:
        sql = f'SELECT * FROM "{table_name}"'
        if conditions:
            sql += f" WHERE {conditions}"
        sql += " LIMIT 50"
        c.execute(sql)
        rows = c.fetchall()
        if not rows:
            return "No records"
        cols = [d[0] for d in c.description]
        result = [{col: r[col] for col in cols} for r in rows]
        return _json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()

@tool
def db_update(table_name: str, record_id: int, data: str) -> str:
    """Db Update."""
    import sqlite3, json as _json
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        d = _json.loads(data)
        sets = ", ".join([f'"{k}" = ?' for k in d.keys()])
        c.execute(f'UPDATE "{table_name}" SET {sets} WHERE id = ?', list(d.values()) + [record_id])
        conn.commit()
        if c.rowcount > 0:
            return f"Updated record ID {record_id}"
        return f"No record found with ID {record_id}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()

@tool
def db_delete(table_name: str, record_id: int) -> str:
    """Db Delete."""
    import sqlite3
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        c.execute(f'DELETE FROM "{table_name}" WHERE id = ?', [record_id])
        conn.commit()
        if c.rowcount > 0:
            return f"Deleted record ID {record_id}"
        return f"No record found with ID {record_id}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()
@tool
def db_drop_table(table_name: str) -> str:
    """删除（DROP）整个数据库表，包括所有数据和表结构。谨慎使用，不可恢复。"""
    import sqlite3
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        c.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.commit()
        return f"表 {table_name} 已删除"
    except Exception as e:
        return f"删除失败: {e}"
    finally:
        conn.close()


@tool
def db_list_tables() -> str:
    """Db List Tables."""
    import sqlite3
    conn = sqlite3.connect(state.db_path)
    c = conn.cursor()
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = c.fetchall()
        if not tables:
            return "No tables yet. Use db_create_table to create one."
        return "\n".join([t[0] for t in tables])
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


@tool
def github_get_file(repo: str, path: str, branch: str = "") -> str:
    """GitHub 获取仓库文件内容。repo 格式为 owner/repo，如 octocat/Hello-World。path 为文件路径，如 README.md 或 src/main.py。branch 可选，默认主分支。"""
    import state, urllib.request, json as _json, base64
    if not state.github_token:
        return "GitHub not configured"






@tool
def github_get_file(repo: str, path: str, branch: str = "") -> str:
    """GitHub 获取仓库文件内容。repo 格式为 owner/repo，如 octocat/Hello-World。path 为文件路径，如 README.md 或 src/main.py。branch 可选，默认主分支。"""
    import state, urllib.request, json as _json, base64
    if not state.github_token:
        return "GitHub not configured"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    if branch:
        url += f"?ref={branch}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {state.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BumpingAgent"
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = _json.loads(resp.read())
        if isinstance(data, list):
            items = "\n".join([f"  \U0001f4c1 {i['name']}/" if i['type'] == 'dir' else f"  \U0001f4c4 {i['name']}" for i in data])
            return f"目录 {path} 的内容:\n{items}"
        if data.get("encoding") == "base64" and data.get("content"):
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            size = data.get("size", 0)
            if size > 10000:
                content = content[:10000] + "\n\n... (文件过大，仅显示前 10000 字符)"
            return f"文件 {path} ({size} 字节):\n\n{content}"
        return _json.dumps(data, ensure_ascii=False, indent=2)[:2000]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"文件 {path} 不存在于 {repo}"
        return f"GitHub API 错误: {e.code} - {e.read().decode()[:200]}"
    except Exception as e:
        return f"请求失败: {e}"

@tool
def github_search_code(query: str, owner: str, repo: str) -> str:
    """GitHub 搜索仓库代码。搜索指定仓库中的代码，返回匹配的文件和行。"""
    import state, urllib.request, json as _json
    if not state.github_token:
        return "GitHub not configured"
    url = f"https://api.github.com/search/code?q={query}+repo:{owner}/{repo}&per_page=10"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {state.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BumpingAgent"
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = _json.loads(resp.read())
        if not data.get("items"):
            return "未找到匹配的代码"
        results = []
        for item in data["items"][:10]:
            results.append(f"  {item['path']} - {item['html_url']}")
        return f"找到 {data['total_count']} 个结果，显示前 {len(results)} 个:\n" + "\n".join(results)
    except urllib.error.HTTPError as e:
        return f"GitHub API 错误: {e.code}"
    except Exception as e:
        return f"请求失败: {e}"

@tool
def github_list_repos(username: str, type_filter: str = "all") -> str:
    """GitHub 列出用户的仓库。username 为 GitHub 用户名。type_filter 可选: all, owner, member。"""
    import state, urllib.request, json as _json
    if not state.github_token:
        return "GitHub not configured"
    url = f"https://api.github.com/users/{username}/repos?type={type_filter}&per_page=20&sort=updated"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {state.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BumpingAgent"
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        repos = _json.loads(resp.read())
        if not repos:
            return "未找到仓库"
        lines = []
        for r in repos:
            desc = (r.get("description") or "")[:60]
            lines.append(f"  {r['name']} {'⭐' + str(r['stargazers_count']) if r['stargazers_count'] else ''} - {desc}")
        return "仓库列表:\n" + "\n".join(lines)
    except urllib.error.HTTPError as e:
        return f"GitHub API 错误: {e.code}"
    except Exception as e:
        return f"请求失败: {e}"

@tool
def github_list_branches(repo: str) -> str:
    """GitHub 列出仓库的分支。repo 格式为 owner/repo。"""
    import state, urllib.request, json as _json
    if not state.github_token:
        return "GitHub not configured"
    url = f"https://api.github.com/repos/{repo}/branches?per_page=30"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {state.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BumpingAgent"
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        branches = _json.loads(resp.read())
        if not branches:
            return "未找到分支"
        lines = [f"  {b['name']}" for b in branches]
        return "分支列表:\n" + "\n".join(lines)
    except urllib.error.HTTPError as e:
        return f"GitHub API 错误: {e.code}"
    except Exception as e:
        return f"请求失败: {e}"

@tool
def github_list_pull_requests(repo: str, state_filter: str = "open") -> str:
    """GitHub 列出仓库的 Pull Requests。repo 格式为 owner/repo。state_filter 可选: open, closed, all。"""
    import state, urllib.request, json as _json
    if not state.github_token:
        return "GitHub not configured"
    url = f"https://api.github.com/repos/{repo}/pulls?state={state_filter}&per_page=10"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {state.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BumpingAgent"
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        prs = _json.loads(resp.read())
        if isinstance(prs, dict) and "message" in prs:
            return f"Failed: {prs['message']}"
        if not prs:
            return "没有 Pull Requests"
        lines = []
        for pr in prs:
            lines.append(f"  #{pr['number']} {pr['title']} [{pr['state']}] - {pr['user']['login']}\n     {pr['html_url']}")
        return "Pull Requests:\n" + "\n".join(lines)
    except urllib.error.HTTPError as e:
        return f"GitHub API 错误: {e.code} - {e.read().decode()[:200]}"
    except Exception as e:
        return f"请求失败: {e}"



tools = [get_weather, detect_location, calculate, get_current_time,
         search_knowledge, query_stock_price, search_web, analyze_image, fetch_webpage,
         feishu_query, feishu_add,
         wecom_send_message, wecom_list_users,
         dingtalk_send_message,
         github_list_issues, github_get_issue, github_get_file, github_search_code,
         github_list_repos, github_list_branches, github_list_pull_requests,
         db_create_table, db_insert, db_insert_batch, db_import_csv, db_query, read_uploaded_file,
         db_update, db_delete, db_drop_table, db_list_tables]

