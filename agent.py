# ====== Agent 初始化与管理 ======
import os
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from config import (
    DEFAULT_OPENAI_MODEL, DEFAULT_DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL, OLLAMA_BASE_URL, TEMPERATURE
)
from tools import tools, search_web
from state import _enable_search, _provider, _api_key, _model

# ====== 全局 Agent 实例 ======
agent = None          # 完整工具集（含 search_web）
agent_no_search = None  # 无搜索工具
_analysis_llm = None   # 用于性格分析的 LLM 实例


def init_agent(provider, api_key):
    """初始化或重新配置 Agent。"""
    global agent, agent_no_search, _analysis_llm, _provider, _api_key, _model
    _provider = provider
    _api_key = api_key
    try:
        if provider == "openai":
            llm = ChatOpenAI(model=DEFAULT_OPENAI_MODEL, api_key=api_key, temperature=TEMPERATURE)
        elif provider == "deepseek":
            llm = ChatOpenAI(model=DEFAULT_DEEPSEEK_MODEL, base_url=DEEPSEEK_BASE_URL,
                             api_key=api_key, temperature=TEMPERATURE)
        elif provider == "ollama":
            _model = api_key  # api_key 参数复用作模型名
            llm = ChatOpenAI(model=_model, base_url=OLLAMA_BASE_URL,
                             api_key="ollama", temperature=TEMPERATURE)
            # 验证模型存在
            import urllib.request as _ur
            _req = _ur.Request("http://localhost:11434/api/tags", method="GET")
            _resp = _ur.urlopen(_req, timeout=5)
            _tags = json.loads(_resp.read().decode("utf-8"))
            _models = [m["name"] for m in _tags.get("models", [])]
            if _model not in _models:
                avail = ", ".join(_models) if _models else "无"
                return False, f"模型 {_model} 不存在。已有的: {avail}"
        else:
            return False, "Unknown provider: " + provider

        agent = create_react_agent(llm, tools)
        agent_no_search = create_react_agent(llm, [t for t in tools if t is not search_web])
        _analysis_llm = llm
        return True, "Agent initialized with " + provider
    except Exception as e:
        return False, str(e)


# 启动时尝试从环境变量初始化
if os.environ.get("OPENAI_API_KEY"):
    ok, msg = init_agent("openai", os.environ["OPENAI_API_KEY"])
elif os.environ.get("DEEPSEEK_API_KEY"):
    ok, msg = init_agent("deepseek", os.environ["DEEPSEEK_API_KEY"])


def validate_api_key(provider, api_key):
    """验证 API Key 是否有效。"""
    try:
        if provider == "openai":
            req = Request("https://api.openai.com/v1/models",
                          headers={"Authorization": "Bearer " + api_key})
        elif provider == "deepseek":
            req = Request("https://api.deepseek.com/v1/models",
                          headers={"Authorization": "Bearer " + api_key})
        elif provider == "ollama":
            return None  # 本地模型无需验证
        else:
            return "Unknown provider"

        resp = urlopen(req, timeout=10)
        return None
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
