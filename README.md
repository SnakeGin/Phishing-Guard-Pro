# 🛡️ Phishing Guard Pro | 多特征钓鱼邮件检测与防御系统

**Phishing Guard Pro** 是一个集检测、取证、解释与防御于一体的下一代邮件安全平台。面对日益智能化的网络攻击，本系统采用了创新的**双流神经网络架构 (FMPED)**，结合 BERT 语义分析与硬规则取证技术，不仅能精准识别传统钓鱼邮件，更能有效防御由 LLM 生成的社会工程学诈骗邮件。

-----

## ✨ 核心功能 (Key Features)

  * **🧠 双流融合检测引擎**：
    * **语义流**：利用预训练 **BERT** 模型提取高维语义特征，识别语气急迫、诱导性强的 AI 生成文本。
    * **特征流**：基于正则与 DOM 解析的硬规则引擎，精准捕获 IP 直连、隐藏 Iframe、密码表单等技术指纹。
  * **🛡️ 对抗训练增强**：模型训练阶段引入 **FGSM 对抗算法**，显著提升模型面对逃逸攻击（Evasion Attacks）时的鲁棒性。
  * **🔍 深度取证可视化**：
    * 不再是“黑盒”打分，系统提供详细的**取证日志 (Forensic Logs)**。
    * 自动提取并高亮展示恶意 URL、IP 地址及敏感诱导词。
  * **🤖 LLM 智能防御建议**：
    * 集成 **DeepSeekR1-0528-Qwen3-8B** 大语言模型，根据检测结果实时生成通俗易懂的防御操作指南。
  * **📦 安全沙箱预览**：
    * 内置 `sandbox` 隔离环境，支持无害化预览恶意邮件源码，防止脚本自动执行。
  * **📊 现代化情报看板**：
    * 基于 Vue 3 + ECharts 构建的响应式仪表盘，动态展示风险评分、特征占比及历史趋势。

-----

## 🏗️ 系统架构 (System Architecture)

系统采用前后端分离架构：

  * **后端**：FastAPI (Python) 负责数据解析、深度学习推理及 API 服务。
  * **前端**：Vue.js + Tailwind CSS 构建单页应用 (SPA)，无需构建工具，开箱即用。
  * **模型**：FMPED (Feature-Fused Multimodal Phishing Email Detection) 模型。
  * **数据**：SQLite + SQLAlchemy 进行持久化存储。

-----

## 🛠️ 技术栈 (Tech Stack)

| 模块         | 技术选型                                      |
| :----------- | :-------------------------------------------- |
| **深度学习** | PyTorch, Hugging Face Transformers (BERT)     |
| **后端框架** | FastAPI, Uvicorn, Pydantic                    |
| **数据处理** | BeautifulSoup4, Numpy, Pandas, Email (Stdlib) |
| **前端框架** | Vue.js 3 (CDN), Tailwind CSS (CDN)            |
| **可视化**   | Apache ECharts                                |
| **外部 API** | SiliconFlow (Qwen-2.5-7B-Instruct)            |
| **数据库**   | SQLite (SQLAlchemy ORM)                       |

-----

## 🚀 快速开始 (Quick Start)

### 1\. 环境准备

确保您的环境中已安装 Python 3.8+ 和 CUDA（如果需要 GPU 加速）。

### 2\. 依赖安装

```bash
# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # 根据你的 CUDA 版本调整
pip install fastapi uvicorn sqlalchemy requests beautifulsoup4 transformers scikit-learn pandas httpx

# 或者安装我们提供的环境requirements.txt
pip install -r requirements.txt
```

### 3\. 配置模型与 API Key

1. **模型权重**：确保项目根目录下存在训练好的模型权重文件 `fmped_model.pth`。

     * *如果没有，请运行 `train.py` 进行训练。*

2. **API Key**：打开 `main.py`，配置您的 LLM API Key：

   ```python
   # main.py
   LLM_API_KEY = "sk-your-real-api-key-here"
   ```

### 4\. 启动后端服务

```bash
python main.py
```

  * 服务将启动在 `http://0.0.0.0:8000`。
  * 数据库文件 `phishing_logs.db` 会自动生成。

### 5\. 启动前端

由于前端采用 CDN 引入方式，**无需 npm install**。
直接在浏览器中打开 `index2.html` 文件即可。

  * 推荐使用 VS Code 的 "Live Server" 插件打开，或直接双击文件。

-----

## 📂 项目结构 (Project Structure)

```text
phishing-guard-pro/
├── main.py                 # 后端入口：FastAPI app, 数据库定义, 业务逻辑
├── model_architecture.py   # 模型定义：FMPEDModel 双流网络结构
├── feature_extractor.py    # 特征工程：BERT处理与硬规则提取类
├── train.py           # 训练脚本：数据加载、对抗训练循环
├── index2.html              # 前端入口：Vue3 + Tailwind 单页应用
├── fmped_model.pth         # [生成] 训练好的模型权重
├── phishing_logs.db        # [生成] SQLite 数据库文件
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明文档
```

-----

## 🖥️ 使用指南 (Usage)

1.  **上传分析**：
      * 在首页点击或拖拽 `.eml` 格式的邮件文件到上传区。
2.  **查看报告**：
      * **仪表盘**：查看综合风险评分及 LLM/传统特征的贡献度。
      * **防御建议**：阅读由 AI 生成的针对性安全建议。
3.  **取证分析**：
      * 切换到“**详细取证数据**” Tab，查看系统提取的具体恶意链接和敏感词。
4.  **安全预览**：
      * 切换到“**邮件原文预览**” Tab，在沙箱环境中查看邮件原始排版。

-----

## 📸 系统截图展示 (System Screenshots)

### 1. 初始页面 (Initial View)

系统启动后的上传界面，支持拖拽上传 EML 文件。
![初始页面图](pic/屏幕截图%202025-12-04%20205238.png)

### 2. 分析结果展示 (Analysis Dashboard)

双流检测引擎的评分结果，包含威胁指数仪表盘与特征占比。
![分析结果展示](pic/屏幕截图%202025-12-04%20210416.png)

### 3. 详细取证数据 (Forensic Data)

后端提取的详细攻击指纹，如恶意链接、IP直连等证据。
![详细取证预览](pic/1.png)

### 4. 邮件原文预览 (Sandboxed Preview)

在安全沙箱中无害化预览邮件原始排版。
![邮件原文预览](pic/屏幕截图%202025-12-04%20210358.png)


## ⚠️ 注意事项

  * **数据库重置**：如果你修改了 `main.py` 中的数据库表结构（如增加了字段），请务必删除目录下的 `phishing_logs.db` 文件，重启服务以重新建表。
  * **网络连接**：由于使用了 Hugging Face 的 BERT 模型和 CDN 前端资源，请确保运行环境能够访问互联网（或配置了相应的代理/镜像）。

-----

## 🤝 贡献与致谢

本项目是基于对当前网络安全态势的深入分析而构建的实践项目。

  * 数据集来源：[Phishing Dataset (Hugging Face)](https://huggingface.co/datasets/ealvaradob/phishing-dataset)
  * 大模型支持：[SiliconFlow-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://siliconflow.cn)

-----
