# ====== 工具函数 ======
import re
import json

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage


def clean_response(text):
    """清理回复文本：去掉 markdown 格式、列表符号等。"""
    if not text:
        return text
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[-*+]\s', stripped):
            line = re.sub(r'^\s*[-*+]\s+', '', line)
        if re.match(r'^\d+\.\s', stripped):
            line = re.sub(r'^\s*\d+\.\s+', '', line)
        line = re.sub(r'^\s*[\U0001F300-\U0001FAFF][\U0001F300-\U0001FAFF]?\s+', '', line)
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = _convert_wechat_emoji(text)
    return text.strip()


def _convert_wechat_emoji(text, lang="zh"):
    """将微信聊天记录里的 [表情文字] 转换为实际 emoji。"""
    def _replace(m):
        key = m.group(1)
        return _WECHAT_EMOJI.get(key, key)
    return re.sub(r'\[([^\]]+)\]', _replace, text)


# ====== 消息转换 ======

def to_langchain(msg: dict):
    """将前端消息格式转为 LangChain 消息格式。"""
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
    """将 LangChain 消息格式转为前端消息格式。"""
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


# ====== 微信表情映射表 ======
_WECHAT_EMOJI = {
    "微笑": "\U0001f642", "呲牙": "\U0001f601", "偷笑": "\U0001f602",
    "害羞": "\U0001f633", "尴尬": "\U0001f605", "发呆": "\U0001f636",
    "撇嘴": "\U0001f61e", "难过": "\U0001f622", "流泪": "\U0001f62d",
    "大哭": "\U0001f62d", "恐惧": "\U0001f628", "惊恐": "\U0001f631",
    "惊讶": "\U0001f632", "震惊": "\U0001f632", "白眼": "\U0001f644",
    "抠鼻": "\U0001f624", "得意": "\U0001f60f", "阴险": "\U0001f608",
    "坏笑": "\U0001f608", "色": "\U0001f60b", "亲亲": "\U0001f618",
    "嘴唇": "\U0001f48b", "飞吻": "\U0001f618", "爱心": "\u2764\ufe0f",
    "心碎": "\U0001f494", "强": "\U0001f44d", "赞": "\U0001f44d",
    "抱拳": "\U0001f4aa", "拳头": "\U0001f4aa", "握手": "\U0001f91d",
    "加油": "\U0001f64c", "合十": "\U0001f64f", "鼓掌": "\U0001f44f",
    "调皮": "\U0001f61c", "勾引": "\U0001f446", "衰": "\U0001f937",
    "骷髅": "\U0001f480", "敲打": "\U0001f4a2", "抓狂": "\U0001f4a2",
    "炸弹": "\U0001f4a3", "便便": "\U0001f4a9", "咖啡": "\u2615",
    "啤酒": "\U0001f37a", "蛋糕": "\U0001f370", "西瓜": "\U0001f349",
    "饭": "\U0001f372", "猪头": "\U0001f437", "玫瑰": "\U0001f339",
    "凋谢": "\U0001f33a", "太阳": "\u2600\ufe0f", "月亮": "\U0001f319",
    "星星": "\u2b50", "闪电": "\u26a1", "礼物": "\U0001f381",
    "红包": "\U0001f4b0", "菜刀": "\U0001f5e1\ufe0f", "OK": "\U0001f44c",
    "ok": "\U0001f44c", "胜利": "\u270c\ufe0f", "NO": "\U0001f645",
    "再见": "\U0001f44b", "抱抱": "\U0001f917", "耶": "\u270c\ufe0f",
    "奋斗": "\U0001f4aa",
}
