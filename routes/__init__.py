# ====== 路由蓝图注册 ======
from .chat import chat_bp
from .api import api_bp
from .personas import personas_bp
from .docs import docs_bp
from .main import main_bp

__all__ = ["chat_bp", "api_bp", "personas_bp", "docs_bp", "main_bp"]
