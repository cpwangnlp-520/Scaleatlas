import assert from 'node:assert/strict';
import test from 'node:test';

import { planInference, recommendInferenceParallel } from '../src/engine/inference.ts';
import { parseHFConfig } from '../src/parsers/hf-config.ts';
import { MODEL_PRESETS } from '../src/types/index.ts';
import type { HardwareConfig, InferenceConfig, ModelConfig } from '../src/types/index.ts';

function createQwen35MoeModel() {
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

  const moe = {
    numLocalExperts: parsed.moeConfig?.numLocalExperts || 256,
    numExpertsPerTok: parsed.moeConfig?.numExpertsPerTok || 8,
    numSharedExperts: parsed.moeConfig?.numSharedExperts,
    sharedExpertIntermediateSize: parsed.moeConfig?.sharedExpertIntermediateSize,
    expertIntermediateSize: parsed.moeConfig?.expertIntermediateSize,
  };

  return { model, moe };
}

function createHardware(): HardwareConfig {
  return {
    gpuType: 'H100-SXM',
    gpuMemoryGb: 80,
    gpusPerNode: 8,
    nodeCount: 2,
    interconnectType: 'NVLink',
  };
}

function createInference(): InferenceConfig {
  return {
    inferenceMode: 'online',
    engine: 'vLLM',
    inputTokensAvg: 1024,
    inputTokensP95: 2048,
    outputTokensAvg: 256,
    outputTokensP95: 512,
    targetConcurrency: 16,
    computeDtype: 'bf16',
    weightDtype: 'bf16',
    kvCacheDtype: 'bf16',
    continuousBatching: true,
    pagedKvCache: true,
    speculativeDecoding: false,
  };
}

test('recommendInferenceParallel avoids impossible tensor splits for qwen3.5 moe multimodal inference', () => {
  const { model, moe } = createQwen35MoeModel();
  const hardware = createHardware();
  const inference = createInference();

  const recommended = recommendInferenceParallel(model, hardware, inference, moe);

  assert.equal(recommended.tpSize, 2);
  assert.equal(recommended.ppSize, 2);
  assert.equal(recommended.dpSize, 1);
  assert.equal(recommended.epSize, 1);
  assert.equal(model.hiddenSize % recommended.tpSize, 0);
  assert.equal(model.intermediateSize % recommended.tpSize, 0);
  assert.equal(model.numAttentionHeads % recommended.tpSize, 0);
  assert.equal(model.numKeyValueHeads % recommended.tpSize, 0);
  assert.equal(model.numHiddenLayers % recommended.ppSize, 0);
});

test('planInference uses pp to reduce per-gpu weight and kv cache memory', () => {
  const { model, moe } = createQwen35MoeModel();
  const hardware = createHardware();
  const inference = createInference();

  const withoutPp = planInference(
    model,
    inference,
    hardware,
    { tpSize: 2, ppSize: 1, dpSize: 1, zeroStage: 'none', cpSize: 1, epSize: 1 },
    moe,
  );

  const withPp = planInference(
    model,
    inference,
    hardware,
    { tpSize: 2, ppSize: 2, dpSize: 1, zeroStage: 'none', cpSize: 1, epSize: 1 },
    moe,
  );

  assert.ok(withPp.memWeightsGb < withoutPp.memWeightsGb);
  assert.ok(withPp.memKvCacheGb < withoutPp.memKvCacheGb);
  assert.ok(withPp.peakMemoryGb < withoutPp.peakMemoryGb);
});

test('planInference returns zero safe concurrency when the model does not fit even at concurrency 1', () => {
  const model = {
    sourceType: 'template',
    paramCountTotal: 0,
    ...MODEL_PRESETS['Llama-3-70B'],
  } as ModelConfig;
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
  );

  assert.equal(result.canDeploy, false);
  assert.equal(result.safeConcurrency, 0);
  assert.ok(result.recommendations.some((message) => message.includes('显存')));
});
