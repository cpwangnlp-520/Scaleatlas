import assert from 'node:assert/strict';
import test from 'node:test';

import { parseHFConfig, estimateParamCount } from '../src/parsers/hf-config.ts';

const QWEN35_MOE_CONFIG = {
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
};

test('qwen3.5 moe multimodal config parses nested text moe fields and vision fields', () => {
  const parsed = parseHFConfig(JSON.stringify(QWEN35_MOE_CONFIG));

  assert.equal(parsed.modelFamily, 'multimodal');
  assert.equal(parsed.modelConfig.hiddenSize, 3072);
  assert.equal(parsed.modelConfig.numHiddenLayers, 48);
  assert.equal(parsed.modelConfig.numAttentionHeads, 32);
  assert.equal(parsed.modelConfig.numKeyValueHeads, 2);
  assert.equal(parsed.modelConfig.vocabSize, 248320);
  assert.equal(parsed.modelConfig.maxPositionEmbeddings, 262144);
  assert.equal(parsed.modelConfig.tieWordEmbeddings, false);
  assert.equal(parsed.modelConfig.usesLearnedPositionEmbeddings, false);
  assert.equal(parsed.modelConfig.positionEncodingType, 'rope');
  assert.equal(parsed.modelConfig.normType, 'rmsnorm');
  assert.equal(parsed.modelConfig.attentionType, 'gqa');
  assert.equal(parsed.moeConfig?.numLocalExperts, 256);
  assert.equal(parsed.moeConfig?.numExpertsPerTok, 8);
  assert.equal(parsed.moeConfig?.expertIntermediateSize, 1024);
  assert.equal(parsed.moeConfig?.numSharedExperts, 1);
  assert.equal(parsed.moeConfig?.sharedExpertIntermediateSize, 1024);
  assert.equal(parsed.multimodalConfig?.visionHiddenSize, 1152);
  assert.equal(parsed.multimodalConfig?.visionNumLayers, 27);
  assert.equal(parsed.multimodalConfig?.imagePatchSize, 16);
});

test('estimateParamCount still returns a positive value for multimodal moe configs', () => {
  const estimated = estimateParamCount(QWEN35_MOE_CONFIG);
  assert.ok(estimated > 0);
});

test('parser infers learned position, layernorm, and mha for bert-like configs', () => {
  const parsed = parseHFConfig(JSON.stringify({
    architectures: ['BertForMaskedLM'],
    model_type: 'bert',
    hidden_act: 'gelu',
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 3072,
    vocab_size: 30522,
    max_position_embeddings: 512,
    layer_norm_eps: 1e-12,
    tie_word_embeddings: true,
  }));

  assert.equal(parsed.modelFamily, 'dense');
  assert.equal(parsed.modelConfig.positionEncodingType, 'learned');
  assert.equal(parsed.modelConfig.usesLearnedPositionEmbeddings, true);
  assert.equal(parsed.modelConfig.normType, 'layernorm');
  assert.equal(parsed.modelConfig.attentionType, 'mha');
  assert.equal(parsed.modelConfig.ffnActivation, 'gelu');
});
