# ====== 聊天路由（含流式接口与自动搜索）=====







import json







from flask import Blueprint, request, jsonify, Response















from langchain_core.messages import SystemMessage















import agent







from lang import get_system_prompt







from tools import _search_bing, tools







from config import PERSONALITIES_DIR







from utils import clean_response, to_langchain, to_client







from state import _enable_search















chat_bp = Blueprint("chat", __name__)























def _build_prompt(lang, personality, modes):







    """构建包含性格和模式的系统提示词。"""







    prompt = get_system_prompt(lang)















    if personality:







        pp = PERSONALITIES_DIR / f"{personality}.json"







        if pp.exists():







            try:







                data = json.loads(pp.read_text(encoding="utf-8"))







                pprompt = data.get("prompt", "")







                if pprompt:







                    prompt = pprompt + chr(10) + chr(10) + prompt







            except Exception:







                pass















    if modes.get("think"):







        prompt += chr(10) + chr(10) + "这回可以多想一会儿，把话说透一点。"







    else:







        prompt += chr(10) + chr(10) + "话不用多，自然就好。"















    if modes.get("search"):







        prompt = prompt.replace("上网搜索信息", "上网搜信息（优先用这个）")







    else:







        prompt = prompt.replace("上网搜索信息", "上网搜信息（用户主动提了再用）")















    return prompt























def _auto_search(client_msgs, prompt):







    """开启搜索模式时自动搜索用户问题并注入结果。"""







    if not client_msgs:







        return prompt







    for m in client_msgs:







        if m.get("role") == "user":







            try:







                sr = _search_bing(m["content"][:100])







                if sr:







                    prompt += (chr(10) + chr(10) +







                               "【自动搜索结果】" + chr(10) + sr[:1500] +







                               chr(10) + chr(10) +







                               "注意：以上是联网搜索到的结果。当用户问新闻、最新信息时，你必须基于这些搜索结果来回答。")







                    prompt = prompt.replace(







                        "- search_web: 上网搜信息（优先用这个）",







                        "# 联网搜索已自动执行")







                    prompt = prompt.replace(







                        "上网搜信息（优先用这个）",







                        "联网搜索已完成，请参考上方结果")







                else:







                    prompt += (chr(10) + chr(10) +







                               "【联网搜索】已尝试搜索但未返回结果。直接用你的知识回答，不要说你不能搜索。")







            except Exception:







                prompt += (chr(10) + chr(10) +







                           "【联网搜索】搜索时出错。直接用你的知识回答，不要说你不能搜索。")







            break







    return prompt























@chat_bp.route("/chat", methods=["POST"])







def chat():







    """非流式聊天接口。"""







    data = request.get_json()







    client_msgs = data.get("messages", [])







    lang = data.get("lang", "zh")







    modes = data.get("modes", {})







    personality = data.get("personality", "")















    if agent.agent is None:







        return jsonify({"error": "API key not configured. Please set your API key first."}), 400















    try:







        mode_prompt = _build_prompt(lang, personality, modes)







        if getattr(agent, "_provider", "") == "ollama":







            if modes.get("search"):

                mode_prompt = _auto_search(client_msgs, mode_prompt)















        lc_msgs = [SystemMessage(content=mode_prompt)] + [to_langchain(m) for m in client_msgs]







        current_agent = agent.agent if modes.get("search") else agent.agent_no_search







        result = current_agent.invoke({"messages": lc_msgs})







        agent_msgs = result["messages"]







        known_count = len(client_msgs)







        new_msgs = agent_msgs[known_count + 1:]







        new_client = [to_client(m) for m in new_msgs]















        return jsonify({







            "reply": clean_response(agent_msgs[-1].content or ""),







            "messages": client_msgs + new_client







        })







    except Exception as e:







        import traceback







        traceback.print_exc()







        return jsonify({"error": str(e)}), 500























@chat_bp.route("/chat/stream", methods=["POST"])







def chat_stream():







    """流式聊天接口（SSE）。"""







    data = request.get_json()







    client_msgs = data.get("messages", [])







    lang = data.get("lang", "zh")







    modes = data.get("modes", {})







    personality = data.get("personality", "")















    if agent.agent is None:







        def gen_err():







            yield f"event: error\ndata: {json.dumps({'error': 'API key not configured. Please set your API key first.'})}\n\n"







        return Response(gen_err(), mimetype="text/event-stream")















    def generate():







        mode_prompt = _build_prompt(lang, personality, modes)







        if getattr(agent, "_provider", "") == "ollama":







            if modes.get("search"):

                mode_prompt = _auto_search(client_msgs, mode_prompt)















        lc_msgs = [SystemMessage(content=mode_prompt)] + [to_langchain(m) for m in client_msgs]







        all_msgs = None







        try:







            current_agent = agent.agent if modes.get("search") else agent.agent_no_search







            for step in current_agent.stream({"messages": lc_msgs}):







                for node, state in step.items():







                    msg = state["messages"][-1]







                    all_msgs = state["messages"]







                    if node == "tools":







                        yield f"event: tool_result\ndata: {json.dumps({'content': msg.content[:200], 'name': msg.name if hasattr(msg, 'name') else ''})}\n\n"







                    elif node == "agent":







                        if hasattr(msg, "tool_calls") and msg.tool_calls:







                            for tc in msg.tool_calls:







                                yield f"event: tool_call\ndata: {json.dumps({'name': tc['name'], 'args': tc['args']})}\n\n"







                        elif msg.content:







                            yield f"event: token\ndata: {json.dumps({'text': clean_response(msg.content)})}\n\n"















            if all_msgs:







                known_count = len(client_msgs)







                new_msgs = all_msgs[known_count + 1:]







                new_client = [to_client(m) for m in new_msgs]







                yield f"event: done\ndata: {json.dumps({'messages': client_msgs + new_client})}\n\n"







        except Exception as e:







            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"















    return Response(generate(), mimetype="text/event-stream")







