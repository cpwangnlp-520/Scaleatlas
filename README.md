# ScaleAtlas

ScaleAtlas is an open-source planner for LLM parameters, training VRAM, and inference capacity.

It helps you answer questions like:

- How many parameters does this model really have?
- Will this model fit on my GPUs for training?
- What TP / PP / DP / EP layout is valid?
- How much memory and concurrency do I need for inference?

The UI ships with bilingual `ZH / EN` support and visual regression coverage via Playwright.

## What It Does

ScaleAtlas focuses on three workflows:

- `Parameters`: inspect model structure, parameter breakdown, and memory by precision
- `Training`: estimate memory, throughput, and feasible parallel strategy
- `Inference`: estimate deployment memory, safe concurrency, and serving parallelism

It also supports:

- HuggingFace `config.json` import
- Dense, MoE, and multimodal model structures
- Architecture-aware parameter breakdown
- Visual screenshot baselines for key pages

## Why It Exists

Most model sizing tools stop at one of two extremes:

- a rough spreadsheet with little architecture awareness
- a framework-specific internal calculator that is hard to explain

ScaleAtlas aims for a middle ground:

- simple enough to use as a planning tool
- detailed enough to explain the result
- visual enough to review with engineers and non-specialists together

## Quick Start

Requirements:

- Node.js 20+
- npm

Install dependencies:

```bash
npm install
```

Start the app locally:

```bash
npm run dev
```

Build for production:

```bash
npm run build
```

## Test

Run unit and logic tests:

```bash
npm test
```

Run browser screenshot regression:

```bash
npm run test:e2e
```

Update screenshot baselines:

```bash
npm run test:e2e:update
```

## Project Structure

```text
src/
  content/               Copy dictionaries and copy types
  engine/                Core sizing and planning logic
  features/parameter/    Parameter page model + rendering
  components/            Shared UI and planner pages
  styles/                tokens / layout / planner / parameter styles
  stores/                Zustand state
  parsers/               HuggingFace config parsing
tests/                   Logic and structure tests
e2e/                     Playwright visual regression tests
```

## Notes

- `config.toml` is treated as a local machine-specific file and is ignored by git.
- `dist/`, `test-results/`, and Playwright reports are ignored as generated artifacts.
- Current screenshot baselines are maintained for:
  - parameter page expanded baseline
  - training page result baseline
  - inference page result baseline

## Naming

Product name:

- `ScaleAtlas`

Suggested repository name:

- `scaleatlas`

Suggested GitHub description:

- `Open-source planner for LLM parameters, training VRAM, and inference capacity.`

## Roadmap

- More architecture-aware model presets
- Broader multimodal breakdown support
- CI-backed visual regression
- Better result export and comparison workflows

## 中文说明

ScaleAtlas 是一个开源的大模型资源规划工具，面向：

- 参数量与结构拆解
- 训练显存与并行规划
- 推理显存、并发和部署容量估算

它不是框架级精确模拟器，而是一个可解释、可视化、适合开会和做前期资源决策的工程化估算工具。

## License

MIT
