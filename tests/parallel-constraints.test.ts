import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { getParallelCompatibilityReport, hasExplicitParallelOverride } from '../src/engine/parallel-constraints.ts';
import { parseHFConfig } from '../src/parsers/hf-config.ts';
import type { ModelConfig, ParallelConfig } from '../src/types/index.ts';

function createQwen35MoeModel(): ModelConfig {
  const parsed = parseHFConfig(JSON.stringify({
    architectures: ['Qwen3_5MoeForConditionalGeneration'],
    model_type: 'qwen3_5_moe',
    tie_word_embeddings: false,
    text_config: {
      hidden_act: 'silu',
      hidden_size: 3072,
      max_position_embeddings: 262144,
      model_type: 'qwen3_5_moe_text',
      moe_intermediate_size: 1024,
      num_attention_heads: 32,
      num_experts: 256,
      num_experts_per_tok: 8,
      num_hidden_layers: 48,
      num_key_value_heads: 2,
      shared_expert_intermediate_size: 1024,
      vocab_size: 248320,
      rope_parameters: {
        rope_theta: 10000000,
      },
    },
    vision_config: {
      depth: 27,
      hidden_size: 1152,
      num_heads: 16,
      patch_size: 16,
    },
  }));

  return {
    modelName: parsed.modelConfig.modelName || 'Qwen3.5 MoE',
    modelFamily: parsed.modelFamily,
    sourceType: 'hf_config_json',
    paramCountTotal: 0,
    hiddenSize: parsed.modelConfig.hiddenSize || 3072,
    numHiddenLayers: parsed.modelConfig.numHiddenLayers || 48,
    numAttentionHeads: parsed.modelConfig.numAttentionHeads || 32,
    numKeyValueHeads: parsed.modelConfig.numKeyValueHeads || 2,
    intermediateSize: parsed.modelConfig.intermediateSize || 1024,
    vocabSize: parsed.modelConfig.vocabSize || 248320,
    maxPositionEmbeddings: parsed.modelConfig.maxPositionEmbeddings || 262144,
    tieWordEmbeddings: parsed.modelConfig.tieWordEmbeddings,
    usesLearnedPositionEmbeddings: parsed.modelConfig.usesLearnedPositionEmbeddings,
    ffnActivation: parsed.modelConfig.ffnActivation || 'swiglu',
  };
}

test('parallel compatibility report explains why tp=5 is invalid for qwen3.5 moe', () => {
  const model = createQwen35MoeModel();
  const report = getParallelCompatibilityReport(model, {
    tpSize: 5,
    ppSize: 2,
    dpSize: 1,
    zeroStage: 'none',
    cpSize: 1,
    epSize: 1,
  });

  assert.equal(report.isValid, false);
  assert.ok(report.issues.some((issue) => issue.includes('TP=5')));
  assert.ok(report.issues.some((issue) => issue.includes('attention heads')));
  assert.ok(report.issues.some((issue) => issue.includes('hidden size')));
});

test('parallel compatibility report accepts tp=2 pp=2 for qwen3.5 moe', () => {
  const model = createQwen35MoeModel();
  const report = getParallelCompatibilityReport(model, {
    tpSize: 2,
    ppSize: 2,
    dpSize: 1,
    zeroStage: 'none',
    cpSize: 1,
    epSize: 1,
  });

  assert.equal(report.isValid, true);
  assert.equal(report.issues.length, 0);
  assert.ok(report.notes.some((note) => note.includes('TP=2')));
  assert.ok(report.notes.some((note) => note.includes('PP=2')));
});

test('explicit parallel override detection includes zero stage changes', () => {
  assert.equal(hasExplicitParallelOverride({
    tpSize: 1,
    ppSize: 1,
    dpSize: 1,
    zeroStage: '1',
    cpSize: 1,
    epSize: 1,
  }), false);

  assert.equal(hasExplicitParallelOverride({
    tpSize: 1,
    ppSize: 1,
    dpSize: 1,
    zeroStage: '3',
    cpSize: 1,
    epSize: 1,
  }), true);
});

test('train page and shared result components expose the simplified single-page planner copy', () => {
  const trainSource = readFileSync(new URL('../src/components/train/TrainPlanner.tsx', import.meta.url), 'utf8');
  const sharedSource = readFileSync(new URL('../src/components/shared/MachineSplitMap.tsx', import.meta.url), 'utf8');
  const copySource = readFileSync(new URL('../src/content/plannerCopy.ts', import.meta.url), 'utf8');

  assert.match(trainSource, /result-summary-grid/);
  assert.match(copySource, /推荐并行/);
  assert.match(sharedSource, /machine-split-legend/);
  assert.match(sharedSource, /stage/);
});
