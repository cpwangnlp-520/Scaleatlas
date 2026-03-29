# ScaleAtlas

ScaleAtlas 是一个开源的大模型资源规划工具，用来做参数、训练显存和推理容量的前期估算。

语言版本：

- 英文版：[README.md](./README.md)
- 中文版：当前文件

## 项目概览

ScaleAtlas 主要解决这些实际问题：

- 一个模型到底有多少参数？
- 训练时能不能放进目标 GPU？
- 哪些 TP / PP / DP / EP 组合是合法的？
- 推理部署需要多少显存和并发余量？

项目提供双语界面、架构感知估算，以及基于 Playwright 的浏览器截图回归。

## 功能

- 按模块拆解参数量
- 估算训练显存与吞吐
- 估算推理显存、并发和容量
- 导入 HuggingFace `config.json`
- 支持 Dense、MoE、多模态模型
- 提供浏览器截图回归测试

## 三个工作流

- `Parameters`：查看模型结构、模块级参数量和不同精度下的显存
- `Training`：估算峰值显存、吞吐和推荐并行策略
- `Inference`：估算部署显存、安全并发和推理并行方案

## 快速开始

环境要求：

- Node.js 20+
- npm

安装依赖：

```bash
npm install
```

本地启动：

```bash
npm run dev
```

生产构建：

```bash
npm run build
```

## 测试

运行逻辑测试：

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

## 项目结构

```text
src/
  content/               文案字典与文案类型
  engine/                核心估算与规划逻辑
  features/parameter/    参数页骨架模型与渲染
  components/            共享 UI 和各规划页面
  styles/                tokens / layout / planner / parameter 样式
  stores/                Zustand 状态
  parsers/               HuggingFace 配置解析
tests/                   逻辑与结构测试
e2e/                     Playwright 浏览器截图回归
```

## 说明

- `config.toml` 被视为本地机器专用配置，已加入 git ignore。
- `dist/`、`test-results/` 和 Playwright 报告都被视为生成物，不应提交。
- 当前仓库维护了以下截图基线：
  - 参数页展开态
  - 训练页结果态
  - 推理页结果态
  - `ZH / EN`
  - 部分响应式断点

## 仓库信息建议

- 仓库名：`Scaleatlas`
- 描述：`Open-source planner for LLM parameters, training VRAM, and inference capacity.`

## Roadmap

- 更多架构感知预设
- 更完整的多模态拆解
- 接入 CI 的截图回归
- 更完善的结果导出与对比工作流

## 许可证

MIT
