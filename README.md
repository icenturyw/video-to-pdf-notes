# video-to-pdf-notes

这是一个把 `wdkns/wdkns-skills` 服务化后的 Web 服务。它支持把视频链接或上传视频转换成可浏览的 HTML / PDF 讲义，并提供登录、任务队列、缓存复用和用户级模型配置。

主要能力：

- 浏览器访问的 Web 界面
- 用户注册 / 登录
- 每个用户独立保存模型 API 配置
- 粘贴 YouTube / Bilibili 链接后创建后台转换任务
- 上传本地视频文件后创建后台转换任务
- 任务完成后直接在浏览器内查看 HTML / PDF
- 支持 PDF 库、任务重命名、删除、取消和重跑

## 当前实现范围

当前版本已经包含：

- `yt-dlp` 抓取视频元数据
- 尝试抓取字幕，失败后回退到 Groq Whisper API 转写
- 分段生成讲义，支持长视频
- 关键帧抽取、去重、黑屏过滤和按章节插图
- 调用用户配置的 OpenAI 兼容接口生成结构化中文讲义
- 生成 HTML 和 PDF，并提供浏览器内预览
- 任务状态、日志、取消、删除、复用重生成和完整重跑
- PDF 库与多种排序方式

与原始 skill 的设计目标相比，仍然可以继续补：

- OCR / VLM 驱动的更智能选图
- 更细粒度的章节与时间轴对齐
- 更强的管理后台与配额系统
- 外部 API 与对象存储

## 运行方式

```bash
cd /root/wdkns-skills-web
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run.py
```

默认监听：

```text
http://0.0.0.0:8000
```

生产环境建议：

```bash
export APP_SECRET_KEY='replace-this-in-production'
export APP_HOST='0.0.0.0'
export APP_PORT='8765'
export APP_DEBUG='false'
export GROQ_API_KEY='your-groq-api-key'
python3 run.py
```

## systemd 自启动

仓库中已经包含：

- `start.sh`
- `wdkns-skills-web.service`

默认端口为 `8765`，避免和现有 `8000` 端口冲突。

生产环境建议通过 `/etc/default/video-to-pdf-notes` 注入密钥与代理配置，例如：

```bash
APP_SECRET_KEY=replace-this-in-production
GROQ_API_KEY=your-groq-api-key
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
NO_PROXY=127.0.0.1,localhost,::1
```

## 配置说明

登录后进入“用户配置”页面，可保存：

- `API Base URL`，例如 `https://api.openai.com/v1`
- `模型名`
- `API Key`
- 自定义 `System Prompt`

`API Key` 会使用应用密钥派生出的 Fernet key 加密后保存到本地 SQLite。

## 数据目录

运行时数据默认保存在：

- 数据库：`data/app.db`
- 每个任务的产物：`data/jobs/<user_id>/<job_id>/`

其中通常会包含：

- `metadata.json`
- `transcript.txt`
- `notes.html`
- `notes.pdf`

`data/` 属于运行时目录，不建议提交到 Git。

## 后续扩展建议

如果你要把它变成真正的生产级服务，下一步建议按这个顺序补：

1. 将 Web 与 worker 彻底拆进程。
2. 加对象存储，把 PDF 与截图文件移出本地磁盘。
3. 接入 OCR / VLM 进一步提升图片选择质量。
4. 增加管理后台、配额统计和告警。
5. 提供稳定的外部 API 和分享链接能力。
