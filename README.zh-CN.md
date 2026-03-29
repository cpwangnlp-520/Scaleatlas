# ScaleAtlas

ScaleAtlas 是一个开源的大模型资源规划工具，用于估算参数规模、训练显存和推理容量。

适合回答这些实际问题：

- 一个模型到底有多少参数？
- 训练时能不能放进目标 GPU？
- 哪些 TP / PP / DP / EP 组合是合法的？
- 推理部署需要多少显存和并发余量？

语言版本：

- English: [README.md](./README.md)
- 中文: 当前文件

## 为什么做 ScaleAtlas

很多资源估算流程通常处在两个极端：

- 只靠粗略表格，几乎没有架构感知能力
- 依赖框架内部计算器，结果很难解释和复核

ScaleAtlas 希望做中间层：

- 足够简单，能用于前期规划和沟通
- 足够详细，能解释结果是怎么来的
- 足够直观，适合和工程、产品、算力团队一起评审

## 覆盖的工作流

ScaleAtlas 目前聚焦三类工作流：

- `Parameters`
  查看模型结构、模块级参数量，以及不同精度下的显存占用。
- `Training`
  估算训练峰值显存、吞吐和可行并行策略。
- `Inference`
  估算部署显存、安全并发和服务并行方案。

## 主要特性

- 架构感知的参数拆解
- 支持 Dense、MoE 和多模态模型
- 支持导入 HuggingFace `config.json`
- 内置 `ZH / EN` 双语界面
- 提供基于 Playwright 的浏览器截图回归

## 快速开始

环境要求：

- Node.js 20+
- npm

安装依赖：

```bash
npm install
```

启动本地应用：

```bash
npm run dev
```

生产构建：

```bash
npm run build
```

## 测试

运行逻辑与单元测试：

```bash
npm test
```

运行浏览器截图回归：

```bash
npm run test:e2e
```

更新截图基线：

```bash
npm run test:e2e:update
```

## 仓库结构

```text
src/
  content/               文案字典与文案类型
  engine/                核心估算与规划逻辑
  features/parameter/    参数页模型与渲染
  components/            共享 UI 和各规划页面
  styles/                tokens / layout / planner / parameter 样式
  stores/                Zustand 状态
  parsers/               HuggingFace 配置解析
tests/                   逻辑与结构测试
e2e/                     Playwright 浏览器截图回归
```

## 说明

- `config.toml` 被视为本地机器专用配置，已加入 git ignore。
- `dist/`、`test-results/` 和 Playwright 报告都属于生成物，不应提交。
- 仓库当前维护了以下截图基线：
  - 参数页展开态
  - 训练页结果态
  - 推理页结果态
  - `ZH / EN`
  - 典型响应式断点

## GitHub 展示建议

- 仓库名：`Scaleatlas`
- 描述：
  `Open-source planner for LLM parameters, training VRAM, and inference capacity.`
- Topics：
  `llm`, `gpu`, `vram`, `inference`, `training`, `moe`, `huggingface`, `resource-planning`, `capacity-planning`, `playwright`

## Roadmap

- 更多架构感知预设
- 更完整的多模态拆解
- 接入 CI 的截图回归
- 更完善的结果导出与对比工作流

## 许可证

MIT
