from pathlib import Path

# ====== 应用路径配置 ======
BASE_DIR = Path(__file__).parent
DOC_DIR = BASE_DIR / "knowledge_docs"
IMAGE_DIR = BASE_DIR / "uploads" / "images"
PERSONALITIES_DIR = BASE_DIR / "personalities"
INDEX_DIR = BASE_DIR / "knowledge_index"

# 确保目录存在
for _d in [DOC_DIR, IMAGE_DIR, PERSONALITIES_DIR, INDEX_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# 模型默认值
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
TEMPERATURE = 0.6
