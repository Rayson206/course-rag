# course-rag · 课程讲义智能问答助手

基于 RAG（检索增强生成）的课程讲义问答助手：上传课程 PDF / PPT 讲义，针对讲义内容提问，模型只依据讲义回答，并标注来源出处。

## 技术栈

- Python 3.13 + uv（包与虚拟环境管理）
- OpenAI 兼容接口（硅基流动 SiliconFlow）
- 对话模型：Qwen2.5-7B-Instruct；向量模型：BAAI/bge-large-zh-v1.5（M1 接入）

## 快速开始

```bash
# 1. 安装依赖（uv 会自动创建虚拟环境）
uv sync

# 2. 配置密钥：复制 .env.example 为 .env，填入 SILICONFLOW_API_KEY

# 3. 运行
uv run python -m course_rag.hello_llm
```

## 开发进度

- [x] **M0 环境搭建**：项目骨架、依赖管理、密钥规范、大模型首次对话
- [ ] **M1 最小闭环**：文档切块、向量化、ChromaDB 检索、命令行问答
- [x] **M2 产品化**：PDF/PPT 解析、Gradio 网页、流式输出、来源标注、阈值拒答
- [ ] **M3 简历亮点**：BM25+向量混合检索、Rerank、评测集、Docker 部署
