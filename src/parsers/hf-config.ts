import type {
  HFConfig,
  ParsedHFConfig,
  ModelConfig,
  MoEConfig,
  MultimodalConfig,
  ModelFamily,
  FFNActivation,
  AttentionType,
  NormType,
  PositionEncodingType,
  Locale,
} from '../types';
import { calculateMoEParamCount, calculateParamCount } from '../engine/memory.ts';
import { RUNTIME_COPY } from '../content/runtimeCopy.ts';

export function parseHFConfig(jsonString: string, locale: Locale = 'zh'): ParsedHFConfig {
  const copy = RUNTIME_COPY[locale].parser;
  let config: HFConfig;
  
  try {
    config = JSON.parse(jsonString);
  } catch {
    throw new Error(copy.invalidJson);
  }
  
  const warnings: string[] = [];
  const unrecognizedFields: string[] = [];
  const textConfig = getTextConfig(config);
  const visionConfig = config.vision_config;
  
  const knownFields = new Set([
    'model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'intermediate_size', 'vocab_size', 'max_position_embeddings',
    'torch_dtype', 'hidden_act', 'rms_norm_eps', 'layer_norm_eps', 'norm_type', 'architectures', 'tie_word_embeddings', 'rope_scaling', 'rope_theta', 'rope_parameters',
    'num_local_experts', 'n_routed_experts', 'num_experts', 'n_shared_experts', 'num_experts_per_tok',
    'expert_intermediate_size', 'moe_intermediate_size', 'shared_expert_intermediate_size', 'first_k_dense_replace', 'moe_layer_freq',
    'vision_config', 'audio_config', 'text_config',
  ]);
  
  Object.keys(config).forEach(key => {
    if (!knownFields.has(key)) {
      unrecognizedFields.push(key);
    }
  });
  
  const modelFamily = detectModelFamily(config);
  
  const modelConfig: Partial<ModelConfig> = {
    modelName: config.architectures?.[0] || config.model_type || 'Unknown',
    modelFamily,
    sourceType: 'hf_config_json',
    hiddenSize: textConfig.hidden_size,
    numHiddenLayers: textConfig.num_hidden_layers,
    numAttentionHeads: textConfig.num_attention_heads,
    numKeyValueHeads: textConfig.num_key_value_heads || textConfig.num_attention_heads,
    intermediateSize: textConfig.intermediate_size || textConfig.shared_expert_intermediate_size || textConfig.moe_intermediate_size || (textConfig.hidden_size ? textConfig.hidden_size * 4 : undefined),
    vocabSize: textConfig.vocab_size,
    maxPositionEmbeddings: textConfig.max_position_embeddings,
    ropeScaling: textConfig.rope_scaling || config.rope_scaling,
    tieWordEmbeddings: config.tie_word_embeddings,
    usesLearnedPositionEmbeddings: inferUsesLearnedPositionEmbeddings(config),
    normType: detectNormType(config),
    attentionType: detectAttentionType(config),
    positionEncodingType: detectPositionEncodingType(config),
    ffnActivation: detectFFNActivation(config),
  };
  
  if (!modelConfig.hiddenSize) {
    warnings.push(copy.missingHiddenSize);
  }
  if (!modelConfig.numHiddenLayers) {
    warnings.push(copy.missingNumHiddenLayers);
  }
  
  let moeConfig: Partial<MoEConfig> | undefined;
  if (isMoEModel(config)) {
    const hasSharedExpert = typeof textConfig.shared_expert_intermediate_size === 'number' && textConfig.shared_expert_intermediate_size > 0;
    moeConfig = {
      numLocalExperts: textConfig.num_local_experts || textConfig.n_routed_experts || textConfig.num_experts,
      numExpertsPerTok: textConfig.num_experts_per_tok || 2,
      numSharedExperts: textConfig.n_shared_experts || (hasSharedExpert ? 1 : undefined),
      sharedExpertIntermediateSize: textConfig.shared_expert_intermediate_size,
      firstKDenseReplace: textConfig.first_k_dense_replace,
      moeLayerFrequency: textConfig.moe_layer_freq,
      expertIntermediateSize: textConfig.expert_intermediate_size || textConfig.moe_intermediate_size,
    };
    
    if (!moeConfig.numLocalExperts) {
      warnings.push(copy.missingRoutedExperts);
    }
  }
  
  let multimodalConfig: Partial<MultimodalConfig> | undefined;
  if (modelFamily === 'multimodal') {
    multimodalConfig = {
      visionEncoderType: visionConfig?.hidden_size ? 'vision' : undefined,
      visionHiddenSize: visionConfig?.hidden_size,
      visionNumLayers: visionConfig?.num_hidden_layers || visionConfig?.depth,
      imagePatchSize: visionConfig?.patch_size,
    };
    
    warnings.push(copy.multimodalNeedsVision);
  }
  
  return {
    modelFamily,
    modelConfig,
    moeConfig,
    multimodalConfig,
    unrecognizedFields,
    warnings,
  };
}

export function detectModelFamily(config: HFConfig): ModelFamily {
  if (config.vision_config || config.audio_config) {
    return 'multimodal';
  }

  const textConfig = getTextConfig(config);
  if (textConfig.num_local_experts || textConfig.n_routed_experts || textConfig.num_experts || textConfig.num_experts_per_tok || textConfig.moe_intermediate_size) {
    return 'moe';
  }
  
  const modelType = config.model_type?.toLowerCase() || '';
  
  if (modelType.includes('tts') || modelType.includes('speech') || modelType.includes('audio')) {
    if (!config.vision_config) {
      return 'tts';
    }
  }
  
  return 'dense';
}

export function detectFFNActivation(config: HFConfig): FFNActivation {
  const textConfig = getTextConfig(config);
  const modelType = (textConfig.model_type || config.model_type || '').toLowerCase();
  const hiddenAct = textConfig.hidden_act?.toLowerCase() || config.hidden_act?.toLowerCase() || '';

  if (hiddenAct.includes('gelu')) {
    return 'gelu';
  }
  if (hiddenAct.includes('relu')) {
    return 'relu';
  }
  if (hiddenAct.includes('glu')) {
    return 'swiglu';
  }
  if (hiddenAct.includes('silu')) {
    if (modelType.includes('llama') || modelType.includes('qwen') || modelType.includes('mistral') || modelType.includes('mixtral') || modelType.includes('deepseek')) {
      return 'swiglu';
    }
  }
  
  if (modelType.includes('llama') || modelType.includes('qwen') || modelType.includes('mistral') || modelType.includes('deepseek')) {
    return 'swiglu';
  }
  
  if (modelType.includes('bert') || modelType.includes('roberta')) {
    return 'gelu';
  }
  
  return 'swiglu';
}

export function detectNormType(config: HFConfig): NormType {
  const textConfig = getTextConfig(config);
  const explicitNormType = (textConfig.norm_type || config.norm_type || '').toLowerCase();
  const modelType = (textConfig.model_type || config.model_type || '').toLowerCase();

  if (explicitNormType.includes('rms')) {
    return 'rmsnorm';
  }
  if (explicitNormType.includes('layer')) {
    return 'layernorm';
  }
  if (typeof textConfig.rms_norm_eps === 'number' || typeof config.rms_norm_eps === 'number') {
    return 'rmsnorm';
  }
  if (typeof textConfig.layer_norm_eps === 'number' || typeof config.layer_norm_eps === 'number') {
    return 'layernorm';
  }
  if (modelType.includes('llama') || modelType.includes('qwen') || modelType.includes('mistral') || modelType.includes('mixtral') || modelType.includes('deepseek')) {
    return 'rmsnorm';
  }

  return 'layernorm';
}

export function detectAttentionType(config: HFConfig): AttentionType {
  if (isMQA(config)) {
    return 'mqa';
  }
  if (isGQA(config)) {
    return 'gqa';
  }
  return 'mha';
}

export function detectPositionEncodingType(config: HFConfig): PositionEncodingType {
  return inferUsesLearnedPositionEmbeddings(config) ? 'learned' : 'rope';
}

export function isMoEModel(config: HFConfig): boolean {
  const textConfig = getTextConfig(config);
  return !!(textConfig.num_local_experts || textConfig.n_routed_experts || textConfig.num_experts || textConfig.num_experts_per_tok || textConfig.moe_intermediate_size);
}

export function isMultimodalModel(config: HFConfig): boolean {
  return !!(config.vision_config || config.audio_config);
}

export function isGQA(config: HFConfig): boolean {
  const textConfig = getTextConfig(config);
  const kvHeads = textConfig.num_key_value_heads || textConfig.num_attention_heads || 0;
  const attnHeads = textConfig.num_attention_heads || 0;
  return kvHeads > 0 && kvHeads < attnHeads;
}

export function isMQA(config: HFConfig): boolean {
  const textConfig = getTextConfig(config);
  const kvHeads = textConfig.num_key_value_heads || 0;
  return kvHeads === 1;
}

export function estimateParamCount(config: HFConfig): number {
  const parsed = parseHFConfig(JSON.stringify(config));
  const modelConfig = parsed.modelConfig;

  if (!modelConfig.hiddenSize || !modelConfig.numHiddenLayers || !modelConfig.numAttentionHeads || !modelConfig.vocabSize || !modelConfig.maxPositionEmbeddings || !modelConfig.intermediateSize || !modelConfig.ffnActivation) {
    return 0;
  }

  const fullModelConfig: ModelConfig = {
    modelName: modelConfig.modelName || 'Unknown',
    modelFamily: parsed.modelFamily,
    sourceType: 'hf_config_json',
    paramCountTotal: 0,
    hiddenSize: modelConfig.hiddenSize,
    numHiddenLayers: modelConfig.numHiddenLayers,
    numAttentionHeads: modelConfig.numAttentionHeads,
    numKeyValueHeads: modelConfig.numKeyValueHeads || modelConfig.numAttentionHeads,
    intermediateSize: modelConfig.intermediateSize,
    vocabSize: modelConfig.vocabSize,
    maxPositionEmbeddings: modelConfig.maxPositionEmbeddings,
    ropeScaling: modelConfig.ropeScaling,
    tieWordEmbeddings: modelConfig.tieWordEmbeddings,
    usesLearnedPositionEmbeddings: modelConfig.usesLearnedPositionEmbeddings,
    ffnActivation: modelConfig.ffnActivation,
  };

  if (parsed.modelFamily === 'moe' && parsed.moeConfig?.numLocalExperts) {
    return calculateMoEParamCount(fullModelConfig, {
      numLocalExperts: parsed.moeConfig.numLocalExperts,
      numExpertsPerTok: parsed.moeConfig.numExpertsPerTok || 2,
      numSharedExperts: parsed.moeConfig.numSharedExperts,
      sharedExpertIntermediateSize: parsed.moeConfig.sharedExpertIntermediateSize,
      firstKDenseReplace: parsed.moeConfig.firstKDenseReplace,
      moeLayerFrequency: parsed.moeConfig.moeLayerFrequency,
      expertIntermediateSize: parsed.moeConfig.expertIntermediateSize,
    });
  }

  return calculateParamCount(fullModelConfig);
}

export function mapToModelConfig(config: HFConfig): Partial<ModelConfig> {
  const parsed = parseHFConfig(JSON.stringify(config));
  return parsed.modelConfig;
}

function inferUsesLearnedPositionEmbeddings(config: HFConfig): boolean {
  const textConfig = getTextConfig(config);

  if (config.rope_scaling || config.rope_theta || config.rope_parameters || textConfig.rope_scaling || textConfig.rope_theta || textConfig.rope_parameters) {
    return false;
  }

  const modelType = (textConfig.model_type || config.model_type || '').toLowerCase();
  if (modelType.includes('llama') || modelType.includes('qwen') || modelType.includes('mistral') || modelType.includes('mixtral') || modelType.includes('deepseek')) {
    return false;
  }

  return true;
}

function getTextConfig(config: HFConfig): HFConfig {
  return config.text_config || config;
}
