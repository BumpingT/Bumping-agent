"""Basic tests for the AI Agent application."""
import os
os.environ["DEEPSEEK_API_KEY"] = "test-key"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    from flask import Flask
    assert True

def test_chunk_text():
    from chat_app import KnowledgeBase
    kb = KnowledgeBase()
    chunks = kb._chunk_text("Hello world " * 200, size=100, overlap=20)
    assert len(chunks) > 1

def test_search_empty():
    from chat_app import KnowledgeBase
    kb = KnowledgeBase()
    results = kb.search("test", top_k=3)
    assert results == []

def test_message_conversion():
    from chat_app import to_langchain, to_client
    msg = {"role": "user", "content": "hello"}
    lc_msg = to_langchain(msg)
    assert lc_msg.type == "human"
    assert lc_msg.content == "hello"
    back = to_client(lc_msg)
    assert back["role"] == "user"

def test_lang_strings():
    from chat_app import t_key, get_system_prompt
    zh = get_system_prompt("zh")
    en = get_system_prompt("en")
    assert "weather" in en.lower() or "tool" in en.lower()
    # Translation found when key != result
    assert t_key("upload_no_file", "zh") != "upload_no_file"
    assert t_key("upload_no_file", "en") != "upload_no_file"

def test_html_template():
    from chat_app import app
    with app.test_client() as c:
        resp = c.get("/")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "switchLang" in html
        assert "lang-btn" in html

if __name__ == "__main__":
    test_imports()
    test_chunk_text()
    test_search_empty()
    test_message_conversion()
    test_lang_strings()
    test_html_template()
    print("All 6 tests passed!")
