# ScaleAtlas

ScaleAtlas is an open-source planner for LLM parameters, training VRAM, and inference capacity.

Language:

- English: this file
- Chinese: [README.zh-CN.md](./README.zh-CN.md)

## Overview

ScaleAtlas helps answer practical planning questions before training or deployment:

- How many parameters does this model actually have?
- Will the model fit on the target GPUs for training?
- Which TP / PP / DP / EP layouts are valid?
- How much memory and concurrency are needed for inference?

The application includes bilingual UI support, architecture-aware estimation, and browser screenshot regression coverage.

## Features

- Parameter breakdown by model module
- Training memory and throughput estimation
- Inference memory, concurrency, and capacity estimation
- HuggingFace `config.json` import
- Dense, MoE, and multimodal model support
- Visual regression coverage with Playwright

## Workflows

- `Parameters`: inspect model structure, module-level parameter counts, and memory by precision
- `Training`: estimate peak memory, throughput, and feasible parallel strategy
- `Inference`: estimate deployment memory, safe concurrency, and serving parallelism

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

## Testing

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
  features/parameter/    Parameter-page model and rendering
  components/            Shared UI and planner pages
  styles/                tokens / layout / planner / parameter styles
  stores/                Zustand state
  parsers/               HuggingFace config parsing
tests/                   Logic and structure tests
e2e/                     Playwright visual regression tests
```

## Notes

- `config.toml` is treated as a local machine-specific file and is ignored by git.
- `dist/`, `test-results/`, and Playwright reports are treated as generated artifacts.
- The repository currently maintains screenshot baselines for:
  - parameter page expanded state
  - training result state
  - inference result state
  - `ZH / EN`
  - selected responsive breakpoints

## Suggested Repository Metadata

- Repository name: `Scaleatlas`
- Description: `Open-source planner for LLM parameters, training VRAM, and inference capacity.`

## Roadmap

- More architecture-aware presets
- Broader multimodal breakdown coverage
- CI-backed visual regression
- Better result export and comparison workflows

## License

MIT
