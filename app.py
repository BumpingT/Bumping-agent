# ====== Bumping Agent - 搴旂敤鍏ュ彛 ======
import logging
import os
import sys

from flask import Flask

# 鎶戝埗 Flask 鏃ュ織
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)


@app.after_request
def add_header(response):
    """Add cache control headers."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ====== 璺敱钃濆浘娉ㄥ唽 ======
from routes import chat_bp, api_bp, personas_bp, docs_bp, main_bp

app.register_blueprint(chat_bp)
app.register_blueprint(api_bp)
app.register_blueprint(personas_bp)
app.register_blueprint(docs_bp)
app.register_blueprint(main_bp)


if __name__ == "__main__":
    print("  Bumping Agent - Started")
    print("  http://localhost:5000 (or 5001 if busy)")
    print("  http://localhost:5001")
    try:
        app.run(host="0.0.0.0", port=5001, debug=False)
    except OSError as e:
        print(f"  Error: {e}")
        print("  Port 5001 also in use. Please close other servers.")




