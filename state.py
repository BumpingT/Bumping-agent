# ====== 全局状态变量 ======
# 控制联网搜索开关，tools.search_web() 会读取
# routes/chat.py 中的聊天端点会在请求处理时设置
_enable_search = True

# 存储当前使用的 provider 和 model，供 personality 分析使用
_provider = None
_api_key = None
_model = None

# ====== 飞书凭证 ======
feishu_app_id = ""
feishu_app_secret = ""

# ====== 企业微信 ======
wecom_corpid = ""
wecom_corpsecret = ""

# ====== 钉钉 ======
dingtalk_app_key = ""
dingtalk_app_secret = ""

# ====== GitHub ======
github_token = ""

# ====== 本地数据库 ======
db_path = "data/database/bumping.db"
db_dir = "data/database"
