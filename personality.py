# ====== 性格分析与生成 ======
import re
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from config import PERSONALITIES_DIR
from agent import _analysis_llm


def analyze_and_generate(text, lang="zh"):
    """分析聊天记录，生成性格 prompt。
    如果有 LLM 可用则用 LLM 分析，否则用关键词兜底。
    返回 (name, name_en, prompt) 三元组。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"性格_{timestamp}"
    name_en = f"Personality_{timestamp}"
    prompt = ""

    # 尝试 LLM 深度分析
    if _analysis_llm is not None:
        try:
            sample = text[:6000]
            instruction_lang = "中文" if lang == "zh" else "English"
            result = _analysis_llm.invoke([
                SystemMessage(content="你是一个性格分析专家。分析聊天记录，提取说话者的语气、用词习惯和性格特点。"),
                HumanMessage(content=f"""分析以下聊天记录，用{instruction_lang}回答。

聊天记录：
{sample}

请从这些方面分析：
1. 说话的语气和节奏
2. 标志性的用词和口头禅
3. 句子特点
4. 性格底色
5. 情绪表达方式

然后写一段 prompt，以"你现在是"开头，描述这个人的说话风格。300字左右。

回复格式：
姓名：[2-4个字]
prompt：[详细的说话风格描述]""")
            ])
            raw = result.content.strip()
            if raw:
                for line in raw.split("\n"):
                    if line.startswith("姓名：") or line.startswith("姓名:"):
                        name = line.split("：")[-1].split(":")[-1].strip()[:10]
                prompt_start = raw.find("prompt：")
                if prompt_start < 0:
                    prompt_start = raw.find("prompt:")
                if prompt_start >= 0:
                    prompt = raw[prompt_start + 7:].strip().strip('"').strip("'")
                name_en = f"{name}_{timestamp}"
        except Exception:
            pass

    # 关键词兜底
    if not prompt:
        sentences = [s.strip() for s in re.split(r'[。！？\n.!?\n]', text) if len(s.strip()) > 2]
        avg_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
        total_chars = len(text.strip())

        # 检测语气词和情绪
        cishi = len(re.findall(r'[啦呀哦嚯哼嗯哟]', text[:3000]))
        haha = len(re.findall(r'哈哈|嘻嘻|呵呵|哦哦', text[:3000]))
        happy = len(re.findall(r'哈哈|嘻嘻|开心|喜欢|棒|好呀|不错|谢谢|感恩', text))
        sad = len(re.findall(r'难过|伤心|烦|累|焦虑|压力|加班|忙|累死|崩溃', text))

        parts = []
        if avg_len < 10:
            parts.append("说话很短，干脆利落")
        elif avg_len < 20:
            parts.append("说话自然流畅")
        else:
            parts.append("说话偏长，喜欢详细表达")
        if cishi > 15:
            parts.append("爱用语气词（啦呀哦）")
        if haha > 3:
            parts.append("性格开朗，爱笑")
        if happy > sad * 2:
            parts.append("积极阳光")
        elif sad > happy:
            parts.append("情绪敏感")

        style = "，".join(parts) if parts else "说话自然平和"
        sample_text = "\n".join([s for s in sentences if 3 < len(s) < 40][:8])

        prompt = f"""你现在{style}。

从聊天记录来看，你说话的习惯是这样的：
{sample_text}

跟别人聊天时就用这种感觉来回答，别列一二三，像真人一样自然地聊就好。"""

    return name, name_en, prompt


def save_personality(name, name_en, prompt):
    """保存性格文件到磁盘。"""
    safe_id = re.sub(r"[^\w\u4e00-\u9fff_-]", "_", name)
    data = {"name": name, "name_en": name_en, "prompt": prompt}
    filepath = PERSONALITIES_DIR / f"{safe_id}.json"
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return safe_id


def list_personalities():
    """列出所有已保存的性格。"""
    results = [{"id": "", "name": "默认", "name_en": "Default"}]
    for f in sorted(PERSONALITIES_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "id": f.stem,
                "name": data.get("name", f.stem),
                "name_en": data.get("name_en", f.stem)
            })
        except Exception:
            pass
    return results


def delete_personality(pid):
    """删除指定性格。"""
    filepath = PERSONALITIES_DIR / f"{pid}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def load_personality_prompt(pid):
    """加载指定性格的 prompt。"""
    filepath = PERSONALITIES_DIR / f"{pid}.json"
    if filepath.exists():
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return data.get("prompt", "")
        except Exception:
            return ""
    return ""
