import assert from 'node:assert/strict';
import test from 'node:test';

import { planTraining, recommendParallelStrategy } from '../src/engine/parallel.ts';
import { parseHFConfig } from '../src/parsers/hf-config.ts';
import { MODEL_PRESETS } from '../src/types/index.ts';
import type { HardwareConfig, ModelConfig, MultimodalConfig, TrainingConfig } from '../src/types/index.ts';

test('recommendParallelStrategy uses the full 64-GPU cluster for a 72B model instead of truncating to 60 GPUs', () => {
  const model = {
    sourceType: 'template',
    paramCountTotal: 0,
    ...MODEL_PRESETS['Qwen2.5-72B'],
  } as ModelConfig;

  const hardware: HardwareConfig = {
    gpuType: 'H100-SXM',
    gpuMemoryGb: 80,
    gpusPerNode: 8,
    nodeCount: 8,
    interconnectType: 'NVLink',
  };

  const training: TrainingConfig = {
    trainingType: 'sft',
    seqLen: 4096,
    globalBatchSize: 64,
    microBatchSize: 4,
    gradAccumSteps: 16,
    computeDtype: 'bf16',
    optimizerType: 'AdamW',
    optimizerOffload: false,
    activationCheckpointing: false,
    flashAttention: true,
    recomputation: 'none',
  };

  const recommended = recommendParallelStrategy(model, hardware, training);
  const recommendedGpuCount = recommended.tpSize * recommended.ppSize * recommended.dpSize;

  assert.equal(recommendedGpuCount, 64);
  assert.equal(64 % (recommended.tpSize * recommended.ppSize), 0);
});

test('recommendParallelStrategy prefers no PP when pure DP plus ZeRO already fits and is faster', () => {
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

  const model = {
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
  } as ModelConfig;

  const hardware: HardwareConfig = {
    gpuType: 'H100-SXM',
    gpuMemoryGb: 80,
    gpusPerNode: 8,
    nodeCount: 4,
    interconnectType: 'NVLink',
  };

  const training: TrainingConfig = {
    trainingType: 'sft',
    seqLen: 4096,
    globalBatchSize: 1,
    microBatchSize: 1,
    gradAccumSteps: 1,
    computeDtype: 'bf16',
    optimizerType: 'AdamW',
    optimizerOffload: false,
    activationCheckpointing: false,
    flashAttention: true,
    recomputation: 'none',
  };

  const recommended = recommendParallelStrategy(
    model,
    hardware,
    training,
    {
      numLocalExperts: parsed.moeConfig?.numLocalExperts || 256,
      numExpertsPerTok: parsed.moeConfig?.numExpertsPerTok || 8,
      numSharedExperts: parsed.moeConfig?.numSharedExperts,
      sharedExpertIntermediateSize: parsed.moeConfig?.sharedExpertIntermediateSize,
      expertIntermediateSize: parsed.moeConfig?.expertIntermediateSize,
    },
    parsed.multimodalConfig as MultimodalConfig,
  );

  assert.equal(recommended.tpSize, 1);
  assert.equal(recommended.ppSize, 1);
  assert.equal(recommended.dpSize, 32);
});

test('planTraining respects manual DP and ZeRO overrides even without TP or PP', () => {
  const model = {
    sourceType: 'template',
    paramCountTotal: 0,
    ...MODEL_PRESETS['Llama-3-8B'],
  } as ModelConfig;

  const hardware: HardwareConfig = {
    gpuType: 'H100-SXM',
    gpuMemoryGb: 80,
    gpusPerNode: 8,
    nodeCount: 4,
    interconnectType: 'NVLink',
  };

  const training: TrainingConfig = {
    trainingType: 'sft',
    seqLen: 4096,
    globalBatchSize: 64,
    microBatchSize: 1,
    gradAccumSteps: 64,
    computeDtype: 'bf16',
    optimizerType: 'AdamW',
    optimizerOffload: false,
    activationCheckpointing: false,
    flashAttention: true,
    recomputation: 'none',
  };

  const manualParallel = {
    tpSize: 1,
    ppSize: 1,
    dpSize: 4,
    zeroStage: '3' as const,
    cpSize: 1,
    epSize: 1,
  };

  const result = planTraining(model, training, hardware, manualParallel);

  assert.deepEqual(result.recommendedParallel, manualParallel);
});
