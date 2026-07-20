import json
import os
import state
import agent
from flask import Blueprint, request, jsonify, Response
from urllib.request import Request as UrlRequest, urlopen

api_bp = Blueprint("api", __name__)


@api_bp.route("/api/key", methods=["POST"])
def set_api_key():
    """Save API key and configure agent."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "无效的请求数据"}), 400
    provider = data.get("provider", "deepseek")
    api_key = data.get("key", "").strip()
    model = data.get("model", "")
    if not api_key and provider != "ollama":
        return jsonify({"ok": False, "error": "请选择AI提供商"}), 400
    if not api_key and provider != "ollama":
        return jsonify({"ok": False, "error": "请输入API Key"}), 400
    if model:
        state._model = model
    ok, msg = agent.init_agent(provider, api_key)
    if ok:
        return jsonify({"ok": True, "message": "API Key 已保存"})
    return jsonify({"ok": False, "error": f"初始化失败: {msg}"})


@api_bp.route("/api/key/status", methods=["GET"])
def get_api_key_status():
    configured = agent.agent is not None
    model_name = ""
    provider = ""
    if configured:
        import state as _s
        provider = _s._provider or ""
        model_name = _s._model or ""
        if hasattr(agent, '_analysis_llm') and agent._analysis_llm:
            try:
                model_name = getattr(agent._analysis_llm, 'model_name', '') or model_name
            except:
                pass
    return jsonify({
        "configured": configured,
        "provider": provider,
        "model": model_name
    })


@api_bp.route("/api/ollama/status", methods=["GET"])
def ollama_status():
    try:
        req = UrlRequest("http://localhost:11434/api/tags", method="GET")
        resp = urlopen(req, timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        return jsonify({"running": True, "models": models})
    except Exception as e:
        return jsonify({"running": False, "models": [], "error": str(e)})


@api_bp.route("/api/ollama/pull", methods=["POST"])
def ollama_pull():
    data = request.get_json()
    model = data.get("model", "")
    if not model:
        return jsonify({"ok": False, "error": "请指定模型名称"}), 400
    import subprocess, threading
    def _pull():
        subprocess.run(["ollama", "pull", model], capture_output=True, timeout=600)
    t = threading.Thread(target=_pull, daemon=True)
    t.start()
    return jsonify({"ok": True, "status": "downloading"})


@api_bp.route("/api/ollama/pull/status", methods=["GET"])
def ollama_pull_status():
    model = request.args.get("model", "")
    if not model:
        return jsonify({"status": "not_found"})
    try:
        req = UrlRequest("http://localhost:11434/api/tags", method="GET")
        resp = urlopen(req, timeout=3)
        data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        if model in models:
            return jsonify({"status": "done"})
    except:
        pass
    return jsonify({"status": "not_found"})


@api_bp.route("/api/feishu/config", methods=["POST"])
def set_feishu_config():
    data = request.get_json()
    app_id = data.get("app_id", "").strip()
    app_secret = data.get("app_secret", "").strip()
    if not app_id or not app_secret:
        return jsonify({"ok": False, "error": "APP_ID 和 APP_Secret 不能为空"}), 400
    state.feishu_app_id = app_id
    state.feishu_app_secret = app_secret
    return jsonify({"ok": True, "message": "飞书配置已保存"})


@api_bp.route("/api/feishu/status", methods=["GET"])
def feishu_status():
    configured = bool(state.feishu_app_id and state.feishu_app_secret)
    return jsonify({"configured": configured})


@api_bp.route("/api/wecom/config", methods=["POST"])
def set_wecom_config():
    data = request.get_json()
    corpid = data.get("corpid", "").strip()
    corpsecret = data.get("corpsecret", "").strip()
    if not corpid or not corpsecret:
        return jsonify({"ok": False, "error": "请填写企业ID和Secret"}), 400
    state.wecom_corpid = corpid
    state.wecom_corpsecret = corpsecret
    return jsonify({"ok": True, "message": "企业微信配置已保存"})


@api_bp.route("/api/wecom/status", methods=["GET"])
def wecom_status():
    configured = bool(state.wecom_corpid and state.wecom_corpsecret)
    return jsonify({"configured": configured})


@api_bp.route("/api/dingtalk/config", methods=["POST"])
def set_dingtalk_config():
    data = request.get_json()
    app_key = data.get("app_key", "").strip()
    app_secret = data.get("app_secret", "").strip()
    if not app_key or not app_secret:
        return jsonify({"ok": False, "error": "请填写AppKey和AppSecret"}), 400
    state.dingtalk_app_key = app_key
    state.dingtalk_app_secret = app_secret
    return jsonify({"ok": True, "message": "钉钉配置已保存"})


@api_bp.route("/api/dingtalk/status", methods=["GET"])
def dingtalk_status():
    configured = bool(state.dingtalk_app_key and state.dingtalk_app_secret)
    return jsonify({"configured": configured})


@api_bp.route("/api/github/config", methods=["POST"])
def set_github_config():
    data = request.get_json()
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"ok": False, "error": "请填写GitHub Token"}), 400
    state.github_token = token
    return jsonify({"ok": True, "message": "GitHub配置已保存"})


@api_bp.route("/api/github/status", methods=["GET"])
def github_status():
    configured = bool(state.github_token)
    return jsonify({"configured": configured})


@api_bp.route("/api/db/config", methods=["POST"])
def set_db_config():
    data = request.get_json()
    new_path = data.get("db_path", "").strip()
    if not new_path:
        return jsonify({"ok": False, "error": "请填写数据库路径"}), 400
    old_path = state.db_path
    state.db_path = new_path
    new_dir = os.path.dirname(new_path)
    os.makedirs(new_dir, exist_ok=True)
    if os.path.exists(old_path) and old_path != new_path:
        try:
            import shutil
            shutil.copy2(old_path, new_path)
        except Exception as e:
            return jsonify({"ok": True, "warning": f"路径已保存，但旧数据库未能复制: {e}", "path": new_path})
    return jsonify({"ok": True, "message": "数据库路径已设置", "path": new_path})


@api_bp.route("/api/db/status", methods=["GET"])
def db_status():
    exists = os.path.exists(os.path.dirname(state.db_path))
    return jsonify({"path": state.db_path, "dir_exists": exists})


@api_bp.route("/api/dialog/select-folder", methods=["POST", "OPTIONS"])
def select_folder():
    """Open native Windows folder picker dialog and return selected path."""
    if request.method == "OPTIONS":
        return jsonify({}), 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS"
        }
    data = request.get_json(silent=True) or {}
    title = data.get("title", "选择数据库保存文件夹")
    ps_script = f'''
Add-Type -AssemblyName System.Windows.Forms
$folder = New-Object System.Windows.Forms.FolderBrowserDialog
$folder.Description = "{title}"
$folder.ShowNewFolderButton = $true
if ($folder.ShowDialog() -eq "OK") {{
    Write-Output $folder.SelectedPath
}}
'''
    import subprocess
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=120
        )
        path = result.stdout.strip()
        if path:
            return jsonify({"ok": True, "path": path})
        return jsonify({"ok": False, "error": "未选择文件夹"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.route("/api/db/tables", methods=["GET"])
def db_list_tables_api():
    """List all database tables."""
    try:
        import sqlite3
        conn = sqlite3.connect(state.db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in c.fetchall()]
        conn.close()
        return jsonify({"ok": True, "tables": tables})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.route("/api/db/export/<table_name>", methods=["GET"])
def db_export(table_name):
    fmt = request.args.get("format", "csv")
    try:
        conn = __import__("sqlite3").connect(state.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM "' + table_name + '"')
        rows = c.fetchall()
        if not rows:
            return jsonify({"ok": False, "error": "empty"}), 404
        cols = [d[0] for d in c.description]
        data = [dict(zip(cols, row)) for row in rows]
        conn.close()
        if fmt == "csv":
            out = __import__("io").StringIO()
            w = __import__("csv").writer(out)
            w.writerow(cols)
            for row in rows:
                w.writerow(row)
            return out.getvalue(), 200, {"Content-Type": "text/csv;charset=utf-8", "Content-Disposition": "attachment; filename=export.csv"}
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.route("/api/db/drop-table", methods=["POST"])
def db_drop_table_api():
    """Delete (DROP) a database table."""
    data = request.get_json(silent=True) or {}
    table_name = data.get("table_name", "").strip()
    if not table_name:
        return jsonify({"ok": False, "error": "请指定表名"}), 400
    try:
        import sqlite3
        conn = sqlite3.connect(state.db_path)
        c = conn.cursor()
        c.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.commit()
        conn.close()
        return jsonify({"ok": True, "message": f"表 {table_name} 已删除"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
