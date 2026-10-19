"""
M0 第一个脚本：和大模型说上话
运行方式（在项目根目录）：uv run python -m course_rag.hello_llm
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

# 1. 从 .env 文件加载环境变量（密钥不硬编码在代码里）
load_dotenv()
api_key = os.getenv("SILICONFLOW_API_KEY")

if not api_key or api_key.startswith("sk-在这里"):
    raise SystemExit("未检测到密钥：请先在项目根目录创建 .env 文件，填入 SILICONFLOW_API_KEY")

# 2. 创建客户端。硅基流动提供 OpenAI 兼容接口，只需把 base_url 换成它的地址
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
)

# 3. 发送一次对话请求（messages 就是聊天记录，按时间顺序排列）
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",  # 免费模型；若报错提示模型不存在，先查 /v1/models
    messages=[
        {"role": "system", "content": "你是一个乐于助人的AI助教。"},
        {"role": "user", "content": "你好！请用一句话介绍你自己。"},
    ],
)

# 4. 取出并打印模型的回复
print(response.choices[0].message.content)
