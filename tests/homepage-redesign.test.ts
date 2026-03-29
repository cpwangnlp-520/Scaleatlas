import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';

test('planner copy is centralized in a shared bilingual dictionary', () => {
  const copyPath = new URL('../src/content/plannerCopy.ts', import.meta.url);

  assert.equal(existsSync(copyPath), true);

  const source = readFileSync(copyPath, 'utf8');
  assert.match(source, /APP_CHROME_COPY/);
  assert.match(source, /STEP_FLOW_COPY/);
  assert.match(source, /zh:/);
  assert.match(source, /en:/);
  assert.match(source, /parameter:/);
  assert.match(source, /train:/);
  assert.match(source, /infer:/);
});

test('app shell keeps only the three entries and locale toggle in the top bar', () => {
  const appSource = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');

  assert.match(appSource, /APP_CHROME_COPY/);
  assert.match(appSource, /workspace-nav-pill/);
  assert.match(appSource, /ZH/);
  assert.match(appSource, /EN/);
  assert.doesNotMatch(appSource, /brandSubtitle/);
  assert.doesNotMatch(appSource, /currentView/);
  assert.doesNotMatch(appSource, /workspace-page-note/);
});

test('parameter page shows a collapsible config section and click-to-expand skeleton nodes', () => {
  const source = readFileSync(new URL('../src/features/parameter/components/ParameterSkeleton.tsx', import.meta.url), 'utf8');
  const modelSource = readFileSync(new URL('../src/features/parameter/modelSkeleton.ts', import.meta.url), 'utf8');
  const copySource = readFileSync(new URL('../src/content/plannerCopy.ts', import.meta.url), 'utf8');

  assert.match(source, /parameter-skeleton-shell/);
  assert.match(source, /expandedBlockId/);
  assert.match(modelSource, /Token Embedding/);
  assert.match(modelSource, /RoPE \/ Position/);
  assert.match(modelSource, /Decoder Stack x/);
  assert.match(modelSource, /LM Head \/ Output/);
  assert.match(copySource, /高级/);
  assert.match(copySource, /模型骨架/);
  assert.match(copySource, /点击节点展开内部结构/);
  assert.doesNotMatch(modelSource, /getDecoderInternalLabels/);
});

test('parameter page derives skeleton internals from config and breakdown instead of fixed templates', () => {
  const source = readFileSync(new URL('../src/features/parameter/modelSkeleton.ts', import.meta.url), 'utf8');

  assert.match(source, /buildPositionSkeletonBlock/);
  assert.match(source, /buildDecoderInternalLabels/);
  assert.match(source, /sanitizeModuleLabel/);
  assert.doesNotMatch(source, /getDecoderInternalLabels/);
});

test('config import summary exposes inferred norm, attention, and position metadata', () => {
  const source = readFileSync(new URL('../src/components/shared/ConfigImportPanel.tsx', import.meta.url), 'utf8');

  assert.match(source, /normType/);
  assert.match(source, /attentionType/);
  assert.match(source, /positionEncodingType/);
});

test('training and inference planners forward inferred architecture metadata into derived model configs', () => {
  const trainSource = readFileSync(new URL('../src/components/train/TrainPlanner.tsx', import.meta.url), 'utf8');
  const inferSource = readFileSync(new URL('../src/components/infer/InferencePlanner.tsx', import.meta.url), 'utf8');

  assert.match(trainSource, /normType:\s*model\.normType/);
  assert.match(trainSource, /attentionType:\s*model\.attentionType/);
  assert.match(trainSource, /positionEncodingType:\s*model\.positionEncodingType/);
  assert.match(inferSource, /normType:\s*model\.normType/);
  assert.match(inferSource, /attentionType:\s*model\.attentionType/);
  assert.match(inferSource, /positionEncodingType:\s*model\.positionEncodingType/);
});

test('training planner uses stacked single-page steps with a four-block result summary', () => {
  const source = readFileSync(new URL('../src/components/train/TrainPlanner.tsx', import.meta.url), 'utf8');
  const copySource = readFileSync(new URL('../src/content/plannerCopy.ts', import.meta.url), 'utf8');

  assert.match(source, /data-page-tone="train"/);
  assert.match(source, /data-step="model"/);
  assert.match(source, /data-step="task"/);
  assert.match(source, /data-step="hardware"/);
  assert.match(source, /data-step="result"/);
  assert.match(source, /result-summary-grid/);
  assert.match(copySource, /能不能训/);
  assert.match(copySource, /峰值显存/);
  assert.match(copySource, /推荐并行/);
  assert.match(copySource, /吞吐/);
});

test('inference planner uses stacked single-page steps with a four-block result summary', () => {
  const source = readFileSync(new URL('../src/components/infer/InferencePlanner.tsx', import.meta.url), 'utf8');
  const copySource = readFileSync(new URL('../src/content/plannerCopy.ts', import.meta.url), 'utf8');

  assert.match(source, /data-page-tone="infer"/);
  assert.match(source, /data-step="model"/);
  assert.match(source, /data-step="service"/);
  assert.match(source, /data-step="hardware"/);
  assert.match(source, /data-step="result"/);
  assert.match(source, /result-summary-grid/);
  assert.match(copySource, /能否部署/);
  assert.match(copySource, /安全并发/);
  assert.match(copySource, /峰值显存/);
  assert.match(copySource, /推荐并行/);
});

test('machine split map exposes tone-aware roles instead of same-color cards', () => {
  const source = readFileSync(new URL('../src/components/shared/MachineSplitMap.tsx', import.meta.url), 'utf8');

  assert.match(source, /machine-split-legend/);
  assert.match(source, /machine-split-cell-role/);
  assert.match(source, /weight/);
  assert.match(source, /stage/);
});

test('phase 2 planner result step is explicitly emphasized and uses a primary summary card', () => {
  const trainSource = readFileSync(new URL('../src/components/train/TrainPlanner.tsx', import.meta.url), 'utf8');
  const inferSource = readFileSync(new URL('../src/components/infer/InferencePlanner.tsx', import.meta.url), 'utf8');
  const cssSource = readFileSync(new URL('../src/styles/planner.css', import.meta.url), 'utf8');

  assert.match(trainSource, /planner-step-card planner-step-card-result/);
  assert.match(inferSource, /planner-step-card planner-step-card-result/);
  assert.match(trainSource, /result-summary-card result-summary-card-primary/);
  assert.match(inferSource, /result-summary-card result-summary-card-primary/);
  assert.match(cssSource, /\.planner-step-card-result\s*\{/);
  assert.match(cssSource, /\.result-summary-card-primary\s*\{/);
});

test('phase 1 dashboard refresh defines lighter surface tokens and applies them to core containers', () => {
  const indexSource = readFileSync(new URL('../src/index.css', import.meta.url), 'utf8');
  const tokenSource = readFileSync(new URL('../src/styles/tokens.css', import.meta.url), 'utf8');
  const layoutSource = readFileSync(new URL('../src/styles/layout.css', import.meta.url), 'utf8');
  const plannerSource = readFileSync(new URL('../src/styles/planner.css', import.meta.url), 'utf8');

  assert.match(indexSource, /styles\/tokens\.css/);
  assert.match(indexSource, /styles\/layout\.css/);
  assert.match(indexSource, /styles\/planner\.css/);
  assert.match(indexSource, /styles\/parameter\.css/);
  assert.match(tokenSource, /--surface-base:/);
  assert.match(tokenSource, /--surface-elevated:/);
  assert.match(tokenSource, /--surface-interactive:/);
  assert.match(tokenSource, /--border-soft:/);
  assert.match(tokenSource, /--border-strong:/);
  assert.match(tokenSource, /--focus-ring:/);
  assert.match(layoutSource, /\.workspace-toolbar\s*\{/);
  assert.match(layoutSource, /border:\s*1px solid var\(--border-soft\)/);
  assert.match(layoutSource, /\.workspace-panel\s*\{/);
  assert.match(layoutSource, /\.control-field,\s*\n\.control-select\s*\{/);
  assert.match(plannerSource, /\.planner-step-card\s*\{/);
  assert.match(plannerSource, /\.result-summary-card\s*\{/);
});

test('phase 3 parameter skeleton introduces lane and KPI structure to simplify the default view', () => {
  const pageSource = readFileSync(new URL('../src/features/parameter/components/ParameterSkeleton.tsx', import.meta.url), 'utf8');
  const cssSource = readFileSync(new URL('../src/styles/parameter.css', import.meta.url), 'utf8');

  assert.match(pageSource, /parameter-skeleton-lane parameter-skeleton-lane-main/);
  assert.match(pageSource, /parameter-skeleton-lane parameter-skeleton-lane-side/);
  assert.match(pageSource, /parameter-skeleton-node-header/);
  assert.match(pageSource, /parameter-skeleton-node-kpi/);
  assert.match(cssSource, /\.parameter-skeleton-lane\s*\{/);
  assert.match(cssSource, /\.parameter-skeleton-node-header\s*\{/);
  assert.match(cssSource, /\.parameter-skeleton-node-kpi\s*\{/);
});

test('parameter skeleton uses tone-specific block classes and richer expanded detail stats', () => {
  const pageSource = readFileSync(new URL('../src/features/parameter/components/ParameterSkeleton.tsx', import.meta.url), 'utf8');
  const modelSource = readFileSync(new URL('../src/features/parameter/modelSkeleton.ts', import.meta.url), 'utf8');
  const cssSource = readFileSync(new URL('../src/styles/parameter.css', import.meta.url), 'utf8');

  assert.match(pageSource, /parameter-skeleton-node-tone-/);
  assert.match(modelSource, /tone:\s*'token'/);
  assert.match(pageSource, /parameter-skeleton-node-tone-\$\{block\.tone\}/);
  assert.match(pageSource, /parameter-skeleton-expanded parameter-skeleton-expanded-tone-/);
  assert.match(pageSource, /parameter-skeleton-detail-cell/);
  assert.match(pageSource, /formatDetailShare/);
  assert.match(cssSource, /\.parameter-skeleton-node-tone-token\s*\{/);
  assert.match(cssSource, /\.parameter-skeleton-node-tone-decoder\s*\{/);
  assert.match(cssSource, /\.parameter-skeleton-expanded-tone-token\s*\{/);
  assert.match(cssSource, /\.parameter-skeleton-detail-cell\s*\{/);
});

test('parameter skeleton compresses default notes and aligns expanded detail rows like a compact table', () => {
  const pageSource = readFileSync(new URL('../src/features/parameter/components/ParameterSkeleton.tsx', import.meta.url), 'utf8');
  const cssSource = readFileSync(new URL('../src/styles/parameter.css', import.meta.url), 'utf8');

  assert.match(pageSource, /parameter-skeleton-detail-table-header/);
  assert.match(pageSource, /getDetailToneClass/);
  assert.match(cssSource, /-webkit-line-clamp:\s*2/);
  assert.match(cssSource, /\.parameter-skeleton-detail-table-header\s*\{/);
  assert.match(cssSource, /\.parameter-skeleton-detail-item-tone-router\s*\{/);
  assert.match(cssSource, /\.parameter-skeleton-detail-item-tone-experts\s*\{/);
});

test('parameter skeleton renders expanded detail once per lane instead of nesting it inside narrow block cards', () => {
  const pageSource = readFileSync(new URL('../src/features/parameter/components/ParameterSkeleton.tsx', import.meta.url), 'utf8');

  assert.match(pageSource, /const expandedMainBlock/);
  assert.match(pageSource, /const expandedSideBlock/);
  assert.match(pageSource, /expandedMainBlock && \(/);
  assert.match(pageSource, /expandedSideBlock && \(/);
});

test('parameter page feature files and stylesheet are split out of the monolithic page and root stylesheet', () => {
  const pageSource = readFileSync(new URL('../src/pages/ParameterPage.tsx', import.meta.url), 'utf8');
  const mainSource = readFileSync(new URL('../src/main.tsx', import.meta.url), 'utf8');
  const indexSource = readFileSync(new URL('../src/index.css', import.meta.url), 'utf8');

  assert.match(pageSource, /ParameterSkeleton/);
  assert.match(pageSource, /buildSkeletonBlocks/);
  assert.match(mainSource, /index\.css/);
  assert.match(indexSource, /styles\/tokens\.css/);
  assert.match(indexSource, /styles\/layout\.css/);
  assert.match(indexSource, /styles\/planner\.css/);
  assert.match(indexSource, /styles\/parameter\.css/);
});

test('root stylesheet no longer carries deprecated architecture and parameter-architecture rule sets', () => {
  const cssSource = readFileSync(new URL('../src/styles/planner.css', import.meta.url), 'utf8');

  assert.doesNotMatch(cssSource, /\.architecture-panel\s*\{/);
  assert.doesNotMatch(cssSource, /\.architecture-tooltip\s*\{/);
  assert.doesNotMatch(cssSource, /\.parameter-architecture-shell\s*\{/);
  assert.doesNotMatch(cssSource, /\.parameter-architecture-inspector\s*\{/);
});
