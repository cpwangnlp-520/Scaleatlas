import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { planInference } from '../src/engine/inference.ts';
import { getModelBreakdown } from '../src/engine/model-breakdown.ts';
import { getParallelCompatibilityReport } from '../src/engine/parallel-constraints.ts';
import { getAllRisks } from '../src/engine/risk.ts';
import { parseHFConfig } from '../src/parsers/hf-config.ts';
import { MODEL_PRESETS } from '../src/types/index.ts';
import type { HardwareConfig, InferenceConfig, ModelConfig, ParallelConfig, TrainingConfig } from '../src/types/index.ts';

function createDenseModel(name: string): ModelConfig {
  return {
    sourceType: 'template',
    paramCountTotal: 0,
    ...MODEL_PRESETS[name],
  } as ModelConfig;
}

test('parseHFConfig returns english parse errors when locale is en', () => {
  assert.throws(
    () => parseHFConfig('{', 'en'),
    /Invalid JSON format/,
  );
});

test('parallel compatibility report returns english messages when locale is en', () => {
  const model = createDenseModel('Qwen2.5-72B');
  const report = getParallelCompatibilityReport(model, {
    tpSize: 5,
    ppSize: 3,
    dpSize: 1,
    zeroStage: 'none',
    cpSize: 1,
    epSize: 1,
  }, undefined, undefined, 'en');

  assert.equal(report.isValid, false);
  assert.ok(report.issues.some((issue) => issue.includes('must divide')));
  assert.ok(report.issues.every((issue) => !/[\u4e00-\u9fff]/.test(issue)));
});

test('risk assessment returns english copy when locale is en', () => {
  const model = createDenseModel('Llama-3-70B');
  const training: TrainingConfig = {
    trainingType: 'sft',
    seqLen: 65536,
    globalBatchSize: 64,
    microBatchSize: 8,
    gradAccumSteps: 8,
    computeDtype: 'bf16',
    optimizerType: 'AdamW',
    optimizerOffload: false,
    activationCheckpointing: false,
    flashAttention: true,
    recomputation: 'none',
  };
  const parallel: ParallelConfig = {
    tpSize: 1,
    ppSize: 1,
    dpSize: 1,
    zeroStage: '1',
    cpSize: 1,
    epSize: 1,
  };
  const hardware: HardwareConfig = {
    gpuType: '4090',
    gpuMemoryGb: 24,
    gpusPerNode: 1,
    nodeCount: 1,
    interconnectType: 'PCIe',
  };

  const risks = getAllRisks(model, training, parallel, hardware, undefined, 'en');

  assert.ok(risks.some((risk) => risk.title === 'Out-of-memory risk'));
  assert.ok(risks.some((risk) => risk.suggestion.includes('Enable') || risk.suggestion.includes('Lower')));
  assert.ok(risks.every((risk) => !/[\u4e00-\u9fff]/.test(`${risk.title}${risk.description}${risk.suggestion}`)));
});

test('planInference returns english recommendations and tags when locale is en', () => {
  const model = createDenseModel('Llama-3-70B');
  const hardware: HardwareConfig = {
    gpuType: '4090',
    gpuMemoryGb: 24,
    gpusPerNode: 1,
    nodeCount: 1,
    interconnectType: 'PCIe',
  };
  const inference: InferenceConfig = {
    inferenceMode: 'online',
    engine: 'vLLM',
    inputTokensAvg: 2048,
    inputTokensP95: 4096,
    outputTokensAvg: 512,
    outputTokensP95: 1024,
    targetConcurrency: 16,
    computeDtype: 'bf16',
    weightDtype: 'bf16',
    kvCacheDtype: 'bf16',
    continuousBatching: true,
    pagedKvCache: true,
    speculativeDecoding: false,
  };

  const result = planInference(
    model,
    inference,
    hardware,
    { tpSize: 1, ppSize: 1, dpSize: 1, zeroStage: 'none', cpSize: 1, epSize: 1 },
    undefined,
    'en',
  );

  assert.equal(result.safeConcurrency, 0);
  assert.ok(result.recommendations.some((message) => message.includes('single-GPU memory')));
  assert.ok(result.explanationTags.some((tag) => tag.includes('Memory') || tag.includes('deployment')));
  assert.ok(result.recommendations.every((message) => !/[\u4e00-\u9fff]/.test(message)));
});

test('parameter breakdown notes stay fully english when locale is en', () => {
  const model = createDenseModel('Llama-3-8B');
  const breakdown = getModelBreakdown(model, undefined, undefined, 'en');
  const notes = [
    ...breakdown.sections.flatMap((section) => [section.note, ...section.children.map((child) => child.note)]),
    ...breakdown.auxiliarySections.flatMap((section) => [section.note, ...section.children.map((child) => child.note)]),
  ].filter(Boolean).join(' ');

  assert.equal(/[\u4e00-\u9fff]/.test(notes), false);
});

test('locale is threaded into shared result components and parameter summary fallbacks', () => {
  const trainSource = readFileSync(new URL('../src/components/train/TrainPlanner.tsx', import.meta.url), 'utf8');
  const inferSource = readFileSync(new URL('../src/components/infer/InferencePlanner.tsx', import.meta.url), 'utf8');
  const parameterSource = readFileSync(new URL('../src/pages/ParameterPage.tsx', import.meta.url), 'utf8');
  const sharedRecommendationSource = readFileSync(new URL('../src/components/shared/RecommendationCard.tsx', import.meta.url), 'utf8');
  const machineSplitSource = readFileSync(new URL('../src/components/shared/MachineSplitMap.tsx', import.meta.url), 'utf8');
  const importPanelSource = readFileSync(new URL('../src/components/shared/ConfigImportPanel.tsx', import.meta.url), 'utf8');

  assert.match(trainSource, /locale=\{locale\}/);
  assert.match(inferSource, /locale=\{locale\}/);
  assert.match(parameterSource, /runtimeCopy\.common\.yes/);
  assert.match(parameterSource, /runtimeCopy\.common\.customModel/);
  assert.match(parameterSource, /buildSkeletonBlocks\(breakdown, activeModel, activeMoe, activeMultimodal, locale\)/);
  assert.match(sharedRecommendationSource, /RUNTIME_COPY\[locale\]/);
  assert.match(machineSplitSource, /RUNTIME_COPY\[locale\]/);
  assert.match(importPanelSource, /parseHFConfig\(importedConfigText, locale\)/);
});

test('home page, compare page, and memory bar are locale-aware instead of hardcoding zh copy', () => {
  const homeSource = readFileSync(new URL('../src/pages/HomePage.tsx', import.meta.url), 'utf8');
  const compareSource = readFileSync(new URL('../src/components/compare/ComparePage.tsx', import.meta.url), 'utf8');
  const memoryBarSource = readFileSync(new URL('../src/components/shared/MemoryBar.tsx', import.meta.url), 'utf8');

  assert.match(homeSource, /usePlannerStore/);
  assert.match(homeSource, /locale/);
  assert.match(compareSource, /usePlannerStore/);
  assert.match(compareSource, /locale/);
  assert.match(memoryBarSource, /locale\?: Locale/);
  assert.match(memoryBarSource, /RUNTIME_COPY\[locale\]/);
});

test('planner form labels and action microcopy are centralized instead of hardcoded in page components', () => {
  const copySource = readFileSync(new URL('../src/content/plannerCopy.ts', import.meta.url), 'utf8');
  const trainSource = readFileSync(new URL('../src/components/train/TrainPlanner.tsx', import.meta.url), 'utf8');
  const inferSource = readFileSync(new URL('../src/components/infer/InferencePlanner.tsx', import.meta.url), 'utf8');

  assert.match(copySource, /moeFields/);
  assert.match(copySource, /optimizerOptions/);
  assert.match(copySource, /parallelFields/);
  assert.match(copySource, /optionLabels/);
  assert.match(copySource, /actionCopy/);
  assert.doesNotMatch(trainSource, /label="Experts"/);
  assert.doesNotMatch(trainSource, /label="Vision Hidden Size"/);
  assert.doesNotMatch(trainSource, /label="Flash Attention"/);
  assert.doesNotMatch(trainSource, /label="ZeRO Stage"/);
  assert.doesNotMatch(inferSource, /label="Experts"/);
  assert.doesNotMatch(inferSource, /label="Continuous Batching"/);
  assert.doesNotMatch(inferSource, /locale === 'zh' \? '开始推理计算'/);
});

test('copy dictionaries use explicit shared types instead of Record<Locale, any>', () => {
  const plannerCopySource = readFileSync(new URL('../src/content/plannerCopy.ts', import.meta.url), 'utf8');
  const runtimeCopySource = readFileSync(new URL('../src/content/runtimeCopy.ts', import.meta.url), 'utf8');

  assert.doesNotMatch(plannerCopySource, /Record<Locale,\s*any>/);
  assert.doesNotMatch(runtimeCopySource, /Record<Locale,\s*any>/);
  assert.match(plannerCopySource, /copyTypes/);
  assert.match(runtimeCopySource, /copyTypes/);
});
