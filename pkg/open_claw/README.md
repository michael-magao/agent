# OpenClaw Go

## 亚马逊商品分析（新增）

基于 **Amazon Product Advertising API 5.0**，抓取商品信息并分析，输出可开发投入市场的产品建议。

### 配置

环境变量：
- `AMAZON_PAAPI_ACCESS_KEY` - PA-API Access Key
- `AMAZON_PAAPI_SECRET_KEY` - PA-API Secret Key
- `AMAZON_PAAPI_PARTNER_TAG` - Partner Tag (Associate ID)
- `AMAZON_PAAPI_COUNTRY` - 站点，默认 US（可选 UK/JP 等）

### 使用方式

**Go CLI：**
```bash
./openclaw amazon analyze -k "蓝牙耳机"
./openclaw agent -m "分析蓝牙耳机市场"  # 自动注入亚马逊数据后调用 LLM
```

**Python Agent：** 工具 `amazon_search_products`、`amazon_analyze_for_product_dev` 已注册，Agent 可自动调用。

```bash
pip install amazon-paapi5  # 或 pip install -r pkg/agentic/requirements-amazon.txt
```

### 输出结构

1. 市场概览（类目规模、竞争密度）
2. 竞品分析（价格带、评分、销量信号）
3. 用户痛点（从评价推断的需求）
4. 产品开发建议（差异化方向、价格带、功能建议）
5. 投入产出评估（竞争难度、预估机会、风险）

---

# OpenClaw Go

Golang 实现的 [OpenClaw](https://github.com/openclaw/openclaw) 核心功能，对标开源项目架构。

## 功能对照

| 功能 | OpenClaw (TS) | open_claw (Go) |
|------|---------------|----------------|
| Gateway WebSocket | ✓ | ✓ ws://127.0.0.1:18789 |
| Session 管理 | ✓ | ✓ |
| Agent 运行时 | Pi agent | LLM Agent (OpenAI/Ollama) |
| CLI: gateway | ✓ | ✓ |
| CLI: agent | ✓ | ✓ |
| CLI: onboard | ✓ | ✓ |
| CLI: doctor | ✓ | ✓ |
| CLI: message send | ✓ | ✓ |
| 配置 ~/.openclaw/openclaw.json | ✓ | ✓ |
| 协议 connect/sessions/agent | ✓ | ✓ |

## 使用

```bash
# 构建
go build -o openclaw ./cmd/openclaw

# 初始化
./openclaw onboard

# 启动 Gateway
./openclaw gateway --port 18789 --verbose

# 发送消息（需设置 OPENAI_API_KEY 或使用 Ollama）
./openclaw agent --message "Hello"

# 诊断
./openclaw doctor
```

## 配置

`~/.openclaw/openclaw.json`:

```json
{
  "agent": { "model": "gpt-4" },
  "gateway": { "port": 18789, "bind": "127.0.0.1" }
}
```

环境变量：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`OPENAI_BASE_URL`（如 Ollama: `http://localhost:11434`）

## 架构

- `config.go` - 配置加载
- `session.go` - Session 模型
- `protocol.go` - WebSocket 帧格式
- `agent.go` - LLM Agent (OpenAI/Ollama)
- `gateway.go` - WebSocket 控制平面
- `cli.go` - Cobra CLI
- `channel.go` - Channel 接口（扩展用）
