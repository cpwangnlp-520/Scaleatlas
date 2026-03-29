import assert from 'node:assert/strict';
import test from 'node:test';

import { calculateMoEParamCount, calculateParamCount } from '../src/engine/memory.ts';
import { parseHFConfig } from '../src/parsers/hf-config.ts';
import type { ModelConfig, MoEConfig } from '../src/types/index.ts';

function createDenseModel(overrides: Partial<ModelConfig> = {}): ModelConfig {
  return {
    modelName: 'Dense Test Model',
    modelFamily: 'dense',
    sourceType: 'manual',
    paramCountTotal: 0,
    hiddenSize: 4096,
    numHiddenLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    intermediateSize: 14336,
    vocabSize: 128256,
    maxPositionEmbeddings: 8192,
    ffnActivation: 'swiglu',
    ...overrides,
  };
}

function createMoeModel(overrides: Partial<ModelConfig> = {}): ModelConfig {
  return {
    modelName: 'MoE Test Model',
    modelFamily: 'moe',
    sourceType: 'manual',
    paramCountTotal: 0,
    hiddenSize: 7168,
    numHiddenLayers: 61,
    numAttentionHeads: 128,
    numKeyValueHeads: 128,
    intermediateSize: 18432,
    vocabSize: 129280,
    maxPositionEmbeddings: 163840,
    ffnActivation: 'swiglu',
    ...overrides,
  };
}

function denseDetailedParamCount(model: ModelConfig, options: { learnedPositionalEmbedding: boolean; tieWordEmbeddings: boolean }): number {
  const { hiddenSize: h, numHiddenLayers: L, numAttentionHeads, numKeyValueHeads, intermediateSize: i, vocabSize: v, maxPositionEmbeddings: s, ffnActivation } = model;
  const kvHeads = numKeyValueHeads || numAttentionHeads;
  const headDim = h / numAttentionHeads;
  const attention = 2 * h * h + 2 * h * headDim * kvHeads;
  const feedforward = (ffnActivation === 'swiglu' ? 3 : 2) * h * i;
  const layerNorms = 2 * h;
  const tokenEmbedding = h * v;
  const positionalEmbedding = options.learnedPositionalEmbedding ? h * s : 0;
  const lmHead = options.tieWordEmbeddings ? 0 : h * v;
  const finalNorm = h;

  return tokenEmbedding + positionalEmbedding + L * (attention + feedforward + layerNorms) + lmHead + finalNorm;
}

function moeDetailedParamCount(model: ModelConfig, moe: MoEConfig & { numSharedExperts?: number; firstKDenseReplace?: number; moeLayerFrequency?: number }, options: { learnedPositionalEmbedding: boolean; tieWordEmbeddings: boolean }): number {
  const { hiddenSize: h, numHiddenLayers: L, numAttentionHeads, numKeyValueHeads, intermediateSize: denseI, vocabSize: v, maxPositionEmbeddings: s, ffnActivation } = model;
  const kvHeads = numKeyValueHeads || numAttentionHeads;
  const headDim = h / numAttentionHeads;
  const attention = 2 * h * h + 2 * h * headDim * kvHeads;
  const denseFfn = (ffnActivation === 'swiglu' ? 3 : 2) * h * denseI;
  const expertI = moe.expertIntermediateSize || denseI;
  const expertFfn = (ffnActivation === 'swiglu' ? 3 : 2) * h * expertI;
  const layerNorms = 2 * h;
  const router = h * moe.numLocalExperts;
  const sharedExperts = moe.numSharedExperts || 0;
  const firstDense = moe.firstKDenseReplace || 0;
  const layerFreq = moe.moeLayerFrequency || 1;

  let totalLayerParams = 0;
  for (let layer = 0; layer < L; layer += 1) {
    const isDenseLayer = layer < firstDense || ((layer - firstDense) % layerFreq !== 0);
    if (isDenseLayer) {
      totalLayerParams += attention + denseFfn + layerNorms;
      continue;
    }

    totalLayerParams += attention + layerNorms + router + expertFfn * (moe.numLocalExperts + sharedExperts);
  }

  const tokenEmbedding = h * v;
  const positionalEmbedding = options.learnedPositionalEmbedding ? h * s : 0;
  const lmHead = options.tieWordEmbeddings ? 0 : h * v;
  const finalNorm = h;

  return tokenEmbedding + positionalEmbedding + totalLayerParams + lmHead + finalNorm;
}

test('dense parameter count uses architecture-specific fields when available', () => {
  const model = {
    ...createDenseModel(),
    ropeScaling: { type: 'dynamic', factor: 8 },
    tieWordEmbeddings: false,
    usesLearnedPositionEmbeddings: false,
  } as ModelConfig;

  const expected = denseDetailedParamCount(model, {
    learnedPositionalEmbedding: false,
    tieWordEmbeddings: false,
  });

  assert.equal(calculateParamCount(model), expected);
});

test('moe parameter count respects expert size and dense replacement layers', () => {
  const model = {
    ...createMoeModel(),
    ropeScaling: { type: 'yarn', factor: 40 },
    tieWordEmbeddings: false,
    usesLearnedPositionEmbeddings: false,
  } as ModelConfig;
  const moe = {
    numLocalExperts: 256,
    numExpertsPerTok: 8,
    expertIntermediateSize: 2048,
    numSharedExperts: 1,
    firstKDenseReplace: 3,
    moeLayerFrequency: 1,
  } as MoEConfig;

  const expected = moeDetailedParamCount(model, moe as MoEConfig & { numSharedExperts?: number; firstKDenseReplace?: number; moeLayerFrequency?: number }, {
    learnedPositionalEmbedding: false,
    tieWordEmbeddings: false,
  });

  assert.equal(calculateMoEParamCount(model, moe), expected);
});

test('hf config parser reads deepseek-specific moe fields', () => {
  const parsed = parseHFConfig(JSON.stringify({
    architectures: ['DeepseekV3ForCausalLM'],
    model_type: 'deepseek_v3',
    hidden_size: 7168,
    intermediate_size: 18432,
    moe_intermediate_size: 2048,
    num_hidden_layers: 61,
    num_attention_heads: 128,
    num_key_value_heads: 128,
    n_routed_experts: 256,
    n_shared_experts: 1,
    num_experts_per_tok: 8,
    first_k_dense_replace: 3,
    moe_layer_freq: 1,
    vocab_size: 129280,
    max_position_embeddings: 163840,
    tie_word_embeddings: false,
    rope_scaling: {
      factor: 40,
      type: 'yarn',
    },
  }));

  assert.equal(parsed.modelFamily, 'moe');
  assert.equal(parsed.modelConfig.tieWordEmbeddings, false);
  assert.equal(parsed.modelConfig.usesLearnedPositionEmbeddings, false);
  assert.equal(parsed.moeConfig?.numLocalExperts, 256);
  assert.equal(parsed.moeConfig?.expertIntermediateSize, 2048);
  assert.equal((parsed.moeConfig as MoEConfig & { numSharedExperts?: number })?.numSharedExperts, 1);
  assert.equal((parsed.moeConfig as MoEConfig & { firstKDenseReplace?: number })?.firstKDenseReplace, 3);
});
