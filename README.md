# course-rag · 课程讲义智能问答助手

上传课程 PDF 讲义，针对讲义内容提问，系统**只依据讲义原文回答、标注来源页码，讲义里没有的内容会明确拒答**——避免大模型"一本正经地胡说八道"。

本项目从零实现了一套完整的 RAG（检索增强生成）系统，并通过**自建 35 题评测集**驱动迭代：纯向量检索基线 77.1% → 引入 BM25 混合检索 82.9% → 加入 Rerank 精排后 **100%**。

- 🧠 混合检索（向量 + BM25 双路召回，RRF 融合）+ Cross-Encoder 重排序
- 📄 PDF 解析、流式输出、来源页码标注、讲义外问题拒答
- 📊 35 题评测集驱动优化，检索准确率 77.1% → 100%，全流程可复现
- 🚀 FastAPI 服务化 + Docker 容器化，一条命令启动，自带在线接口文档

---

## 在线演示

> 待部署（M3.5）：部署后补充公网访问链接。

---

## 效果：评测集驱动的三次迭代

基于 `evals/questions.jsonl`（30 道讲义内题，标注正确页码；5 道讲义外题，应拒答）：

| 检索方案 | Top1 准确率 | Recall@5 | 讲义外拒答率 | 综合正确率 |
| --- | --- | --- | --- | --- |
| 纯向量检索（基线） | 76.7% (23/30) | 100% | 80% (4/5) | 77.1% |
| 向量 + BM25，RRF 融合 | 83.3% (25/30) | 100% | 80% (4/5) | 82.9% |
| **混合召回 + Rerank 精排** | **100% (30/30)** | **100%** | **100% (5/5)** | **100%** |

**每一步优化都来自对错误的分析，而不是盲目堆模型：**

1. 基线的 7 道排序错题**全部是术语/关键词型**：如"银行家算法用来干什么"被含"算法"二字的调度页抢走；"多个进程互相等待都无法推进"（没出现"死锁"二字）没命中死锁页——这是纯向量语义检索的固有短板。
2. 加入 BM25 补上**字面精确匹配**，修好 2 道（+6.6pct）；两路用 RRF 倒数排名融合，无需人工调权重。
3. Recall@5 始终 100%，说明粗召回已充分、瓶颈只在排序——这是引入 Rerank 的前提判断。
4. Rerank（Cross-Encoder）把问题与每个候选块直接配对打分，精度远高于"向量空间算距离"，Top1 拉到 100%。
5. **拒答双阈值**：Rerank 置信度实测讲义内 > 0.75、讲义外 < 0.001，取阈值 0.1，间隔极大、判据可靠；无 Rerank 时自动退回向量距离阈值 0.55。

---

## 系统架构

```mermaid
flowchart LR
    Q[用户提问] --> C{混合粗召回<br/>top 10}
    C -->|BM25<br/>jieba 分词 + 关键词| V1[词法匹配候选]
    C -->|BGE 向量<br/>ChromaDB 余弦| V2[语义相似候选]
    V1 --> R[RRF 倒数排名融合<br/>免权重调参]
    V2 --> R
    R --> RR[BGE Reranker<br/>Cross-Encoder 精排]
    RR --> T{最高置信度<br/>> 0.1 ?}
    T -->|否| X[拒答：讲义中没有相关内容]
    T -->|是| L[拼接上下文<br/>Qwen2.5-7B 生成]
    L --> A[答案 + 来源页码<br/>+ rerank 分数]
```

**关键设计：**

- **粗召回（Recall）与精排（Precision）解耦**：先用便宜的向量 + BM25 多路召回保证不漏，再用贵但准的 Cross-Encoder 只对 10 个候选精排，兼顾效果与成本。
- **拒答机制**：不是所有问题都该回答。检索置信度低于阈值直接拒答，把"幻觉"挡在生成之前。
- **RRF 融合**：`score = Σ 1/(k+rank)`，两路分数尺度不同也无需归一化、无需调权重（k=60）。

---

## 快速开始

### 方式一：Docker（推荐，一条命令）

```bash
# 构建镜像
docker build -t course-rag:local .

# 运行（密钥在运行时通过环境变量注入，不打进镜像）
docker run -d --name course-rag -p 8000:8000 --env-file .env course-rag:local

# 容器首次启动会自动索引 data/ 目录下的 PDF，约 20 秒后访问：
#   http://localhost:8000/docs  （在线接口文档）
```

### 方式二：本地运行（uv 管理环境）

```bash
# 1. 安装依赖（uv 自动创建虚拟环境）
uv sync

# 2. 配置密钥
cp .env.example .env
#   在 .env 中填入 SILICONFLOW_API_KEY（硅基流动免费申请）

# 3a. 启动 FastAPI 接口服务（启动时自动建索引）
uv run uvicorn course_rag.api:app --host 0.0.0.0 --port 8000

# 3b. 或启动 Gradio 网页界面
uv run python -m course_rag.app
#   访问 http://127.0.0.1:7860
```

> 模型均通过硅基流动 SiliconFlow 的 OpenAI 兼容接口调用：对话 `Qwen/Qwen2.5-7B-Instruct`、向量 `BAAI/bge-large-zh-v1.5`、重排序 `BAAI/bge-reranker-v2-m3`。

---

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查，返回服务状态与已索引块数 |
| POST | `/api/ask` | 非流式问答，返回完整答案 + 来源页码 + rerank 分数 |
| POST | `/api/ask/stream` | 流式问答，逐字返回（打字机效果） |
| POST | `/api/index` | 上传 PDF 建立索引（multipart/form-data） |

`POST /api/ask` 响应示例：

```json
{
  "answer": "产生死锁的四个必要条件是：1. 互斥条件 2. 请求并保持条件 3. 不可剥夺条件 4. 循环等待条件",
  "sources": [
    {"source": "lecture.pdf", "page": 7, "rerank_score": 0.9984}
  ]
}
```

---

## 复现评测

```bash
uv run python -m course_rag.evaluate
```

输出三种检索方案（纯向量 / BM25 混合 / Rerank 精排）在 35 题上的 Top1、Recall@5、拒答率与综合正确率，结果记录于 `evals/benchmark.md`。

---

## 项目结构

```
course-rag/
├── src/course_rag/
│   ├── rag_engine.py      # 核心引擎：混合检索 + RRF 融合 + Rerank + 拒答
│   ├── chunking.py        # 讲义切块（300 字滑窗）
│   ├── document_parser.py # pdfplumber 解析 PDF、保留页码
│   ├── index.py           # 向量化入库 ChromaDB
│   ├── qa.py / chat.py    # 命令行问答 / 多轮对话
│   ├── app.py             # Gradio 网页（流式、来源标注、拒答、上传 PDF）
│   ├── api.py             # FastAPI 四个接口
│   └── evaluate.py        # 三方案对比评测
├── evals/
│   ├── questions.jsonl    # 35 题标注集
│   └── benchmark.md       # 评测结果与错误分析
├── data/                  # 示例讲义（7 页 PDF）
├── Dockerfile             # 容器化（分层缓存、健康检查、数据卷）
└── pyproject.toml
```

---

## 技术栈

- **语言/环境**：Python 3.13、uv
- **检索**：ChromaDB（向量库）、BGE-large-zh（向量）、jieba + rank-bm25（词法）、BGE-reranker-v2-m3（重排序）
- **生成**：Qwen2.5-7B-Instruct（硅基流动 OpenAI 兼容接口）
- **服务**：FastAPI、Uvicorn、Gradio
- **部署**：Docker、Docker Desktop（WSL2）
- **解析**：pdfplumber

---

## 开发历程

- [x] **M0 环境搭建**：项目骨架、uv 依赖管理、密钥规范、大模型首次对话
- [x] **M1 最小闭环**：文档切块、向量化、ChromaDB 检索、命令行 RAG 问答
- [x] **M2 产品化**：PDF 解析、抽出 RAGEngine、Gradio 网页、流式输出、来源标注、阈值拒答
- [x] **M3 效果优化与工程化**：
  - [x] 35 题评测集 + 纯向量基线（77.1%）
  - [x] BM25 + 向量混合检索、RRF 融合（82.9%）
  - [x] Rerank 重排序精排（100%）
  - [x] FastAPI 服务化 + Docker 容器化
  - [ ] 公网部署
