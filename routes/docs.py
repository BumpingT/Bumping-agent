# ====== 文档管理路由 ======
import re
from pathlib import Path
from flask import Blueprint, request, jsonify, send_from_directory

from config import DOC_DIR, IMAGE_DIR
from kb import kb
from lang import t_key
from utils import _convert_wechat_emoji

docs_bp = Blueprint("docs", __name__)


@docs_bp.route("/upload", methods=["POST"])
def upload():
    """上传文件（文本、PDF、图片）。"""
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
        save_path = IMAGE_DIR / f"{safe_name}{ext}"
        file.save(str(save_path))
        return jsonify({"name": safe_name, "chunks": 0,
                        "message": f"已保存图片 {safe_name}{ext}，可以问我图片内容"})

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file)
            text = "".join(page.extract_text() + "\n" for page in reader.pages)
            save_path = DOC_DIR / f"{safe_name}.txt"
            save_path.write_text(text, encoding="utf-8")
            chunks = kb.add_document(str(save_path))
            return jsonify({"name": safe_name, "chunks": chunks,
                            "message": t_key("upload_success", lang, name=safe_name, chunks=chunks)})
        except ImportError:
            return jsonify({"error": "PDF 解析需要安装 pypdf: pip install pypdf"}), 400
    else:
        text = file.read().decode("utf-8", errors="replace")
        text = _convert_wechat_emoji(text)
        save_path = DOC_DIR / f"{safe_name}.txt"
        save_path.write_text(text, encoding="utf-8")
        chunks = kb.add_document(str(save_path))
        return jsonify({"name": safe_name, "chunks": chunks,
                        "message": t_key("upload_success", lang, name=safe_name, chunks=chunks)})


@docs_bp.route("/documents", methods=["GET"])
def list_docs():
    """列出已上传的文档。"""
    return jsonify(kb.get_document_list())


@docs_bp.route("/documents/<name>", methods=["DELETE"])
def delete_doc(name: str):
    """删除指定文档。"""
    lang = request.args.get("lang", "zh")
    filepath = DOC_DIR / f"{name}.txt"
    if filepath.exists():
        filepath.unlink()
    kb.remove_document(name)
    return jsonify({"message": t_key("delete_success", lang, name=name)})


@docs_bp.route("/uploads/images", methods=["GET"])
def list_images():
    """列出已上传的图片。"""
    images = []
    for f in sorted(IMAGE_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif"):
            images.append({"name": f.name, "size": f.stat().st_size})
    return jsonify(images)


@docs_bp.route("/uploads/images/<filename>")
def serve_image(filename: str):
    """提供图片文件。"""
    return send_from_directory(str(IMAGE_DIR), filename)
