"""
M0 小升级：终端问答机（支持多轮对话）
运行方式：uv run python -m course_rag.chat
输入问题回车即答；输入 exit 退出。
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# 多轮对话的秘密：用一个列表保存完整聊天记录，每次请求都把历史整包发给模型
messages = [
    {"role": "system", "content": "你是一个乐于助人的AI助教。"},
]

print("=== 终端问答机已启动（输入问题回车提问；输入 exit 退出）===")

while True:
    question = input("\n你：").strip()
    if not question:
        continue
    if question.lower() in ("exit", "quit", "退出"):
        print("再见！")
        break

    # 把用户本轮提问追加进历史
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,  # 传入完整历史，模型因此"记得"上文
    )
    answer = response.choices[0].message.content

    # 把模型回复也追加进历史，下一轮它才知道自己说过什么
    messages.append({"role": "assistant", "content": answer})

    print("AI：" + answer)
