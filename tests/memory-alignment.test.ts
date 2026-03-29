import assert from 'node:assert/strict';
import test from 'node:test';

import { calculateActivationMemory, calculateParamCountSimple } from '../src/engine/memory.ts';
import type { ModelConfig, ParallelConfig, TrainingConfig } from '../src/types/index.ts';

function createModel(overrides: Partial<ModelConfig> = {}): ModelConfig {
  return {
    modelName: 'Test Model',
    modelFamily: 'dense',
    sourceType: 'manual',
    paramCountTotal: 0,
    hiddenSize: 4096,
    numHiddenLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 32,
    intermediateSize: 16384,
    vocabSize: 32000,
    maxPositionEmbeddings: 4096,
    ffnActivation: 'relu',
    ...overrides,
  };
}

function createTraining(overrides: Partial<TrainingConfig> = {}): TrainingConfig {
  return {
    trainingType: 'sft',
    seqLen: 2048,
    globalBatchSize: 8,
    microBatchSize: 1,
    gradAccumSteps: 8,
    computeDtype: 'bf16',
    optimizerType: 'AdamW',
    optimizerOffload: false,
    activationCheckpointing: false,
    flashAttention: false,
    recomputation: 'none',
    ...overrides,
  };
}

function createParallel(overrides: Partial<ParallelConfig> = {}): ParallelConfig {
  return {
    tpSize: 1,
    ppSize: 1,
    dpSize: 1,
    zeroStage: 'none',
    cpSize: 1,
    epSize: 1,
    ...overrides,
  };
}

function playbookSelectiveActivationTotal(model: ModelConfig, training: TrainingConfig): number {
  const bytesPerValue = training.computeDtype === 'fp32' ? 4 : 2;
  const { hiddenSize: h, numHiddenLayers: L, intermediateSize: h_ff, vocabSize: v } = model;
  const { seqLen: s, microBatchSize: b } = training;

  const oneLayerAttention = s * b * h * (bytesPerValue * 4 + bytesPerValue + 1);
  const oneLayerFeedforward = s * b * h * bytesPerValue + s * b * h_ff * bytesPerValue + s * b * h;
  const layerNorm = s * b * h * bytesPerValue;
  const inputDropout = s * b * h;
  const outputLayerNorm = s * b * h * bytesPerValue;
  const outputLayerProjection = s * b * h * bytesPerValue;
  const outputCrossEntropy = s * b * v * 4;

  return (
    (oneLayerAttention + oneLayerFeedforward + 2 * layerNorm) * L +
    inputDropout +
    outputLayerNorm +
    outputLayerProjection +
    outputCrossEntropy
  );
}

test('dense parameter count stays aligned with the original playbook formula', () => {
  const hiddenSize = 4096;
  const numLayers = 32;
  const seqLen = 8192;
  const vocabSize = 128256;

  const expected =
    hiddenSize * (vocabSize + seqLen) +
    numLayers * (12 * hiddenSize ** 2 + 13 * hiddenSize) +
    2 * hiddenSize;

  assert.equal(calculateParamCountSimple(hiddenSize, numLayers, seqLen, vocabSize), expected);
});

test('selective checkpointing matches the original playbook activation formula', () => {
  const model = createModel();
  const training = createTraining({ recomputation: 'selective' });
  const parallel = createParallel();

  const result = calculateActivationMemory(model, training, parallel);
  const expected = playbookSelectiveActivationTotal(model, training);

  assert.equal(result.total, expected);
});

test('pipeline parallel scales per-layer activation memory by local stage depth', () => {
  const model = createModel();
  const training = createTraining({ recomputation: 'none' });
  const withoutPp = calculateActivationMemory(model, training, createParallel({ ppSize: 1 }));
  const withPp = calculateActivationMemory(model, training, createParallel({ ppSize: 2 }));

  assert.equal(withPp.attention, withoutPp.attention / 2);
  assert.equal(withPp.feedforward, withoutPp.feedforward / 2);
  assert.equal(withPp.layerNorm, withoutPp.layerNorm / 2);
});
