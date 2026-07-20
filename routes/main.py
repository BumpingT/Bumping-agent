# ====== 首页路由 ======
import pathlib
from flask import Blueprint

main_bp = Blueprint("main", __name__)

_HTML_CACHE = None


def _get_html(force=False):
    """读取并缓存 HTML 模板。"""
    global _HTML_CACHE
    if _HTML_CACHE is None or force:
        tmpl = pathlib.Path(__file__).parent.parent / "templates" / "index.html"
        if tmpl.exists():
            _HTML_CACHE = tmpl.read_text(encoding="utf-8")
        else:
            _HTML_CACHE = "<html><body><h1>Template not found</h1></body></html>"
    return _HTML_CACHE


_get_html()  # 预热缓存


@main_bp.route("/")
def index():
    return _get_html(force=True)
