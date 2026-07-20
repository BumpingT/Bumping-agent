# ====== 性格管理路由 ======
from flask import Blueprint, request, jsonify

from personality import (
    analyze_and_generate, save_personality,
    list_personalities, delete_personality
)

personas_bp = Blueprint("personas", __name__)


@personas_bp.route("/personalities", methods=["GET"])
def list_personas():
    """列出所有性格。"""
    return jsonify(list_personalities())


@personas_bp.route("/personalities/generate", methods=["POST"])
def generate_persona():
    """上传聊天记录生成性格。"""
    lang = request.form.get("lang", "zh")
    if "file" not in request.files:
        return jsonify({"error": "请选择文件" if lang == "zh" else "Please select a file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空" if lang == "zh" else "File name is empty"}), 400

    text = file.read().decode("utf-8", errors="replace")
    from utils import _convert_wechat_emoji
    text = _convert_wechat_emoji(text)

    if len(text.strip()) < 10:
        return jsonify({"error": "文件内容太少" if lang == "zh" else "File content too short"}), 400

    name, name_en, prompt = analyze_and_generate(text, lang)
    if prompt:
        safe_id = save_personality(name, name_en, prompt)
        return jsonify({"id": safe_id, "name": name,
                        "message": "性格生成成功" if lang == "zh" else "Personality generated"})

    return jsonify({"error": "性格生成失败" if lang == "zh" else "Failed to generate personality"}), 500


@personas_bp.route("/personalities/<pid>", methods=["DELETE"])
def delete_persona(pid: str):
    """删除指定性格。"""
    lang = request.args.get("lang", "zh")
    if delete_personality(pid):
        return jsonify({"message": "性格已删除" if lang == "zh" else "Personality deleted"})
    return jsonify({"error": "找不到性格文件" if lang == "zh" else "Personality not found"}), 404
