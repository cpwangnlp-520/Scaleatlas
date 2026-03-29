export type ModelFamily = 'dense' | 'moe' | 'multimodal' | 'tts';

export type Locale = 'zh' | 'en';

export type WorkloadType = 'train' | 'infer' | 'compare';

export type TrainingType = 'pretrain' | 'sft' | 'lora' | 'qlora' | 'continued_pretrain';

export type InferenceMode = 'offline' | 'online' | 'streaming';

export type ComputeDtype = 'fp32' | 'bf16' | 'fp16' | 'fp8' | 'int8' | 'int4';

export type ZeROStage = 'none' | '1' | '2' | '3';

export type RiskLevel = 'green' | 'yellow' | 'orange' | 'red';

export type GpuType = 'A100-40G' | 'A100-80G' | 'H100-SXM' | 'H100-PCIe' | 'L40S' | 'L20' | '4090' | 'H20';

export type InterconnectType = 'PCIe' | 'NVLink' | 'IB' | 'RoCE';

export type OptimizerType = 'AdamW' | 'AdamW-8bit' | 'Adafactor' | 'Lion';

export type FFNActivation = 'relu' | 'gelu' | 'swiglu';

export type NormType = 'rmsnorm' | 'layernorm';

export type AttentionType = 'mha' | 'gqa' | 'mqa';

export type PositionEncodingType = 'rope' | 'learned';

export type RecomputationMode = 'none' | 'selective' | 'full';

export type InferenceEngine = 'vLLM' | 'SGLang' | 'TensorRT-LLM' | 'TGI' | 'custom';

export interface ModelConfig {
  modelName: string;
  modelFamily: ModelFamily;
  sourceType: 'template' | 'manual' | 'hf_config_json';
  paramCountTotal: number;
  hiddenSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  intermediateSize: number;
  vocabSize: number;
  maxPositionEmbeddings: number;
  ropeScaling?: {
    type: string;
    factor: number;
  };
  tieWordEmbeddings?: boolean;
  usesLearnedPositionEmbeddings?: boolean;
  normType?: NormType;
  attentionType?: AttentionType;
  positionEncodingType?: PositionEncodingType;
  ffnActivation: FFNActivation;
}

export interface MoEConfig {
  numLocalExperts: number;
  numExpertsPerTok: number;
  numSharedExperts?: number;
  sharedExpertIntermediateSize?: number;
  firstKDenseReplace?: number;
  moeLayerFrequency?: number;
  expertHiddenSize?: number;
  expertIntermediateSize?: number;
}

export interface MultimodalConfig {
  visionEncoderType?: string;
  visionHiddenSize?: number;
  visionNumLayers?: number;
  audioEncoderType?: string;
  audioHiddenSize?: number;
  projectorHiddenSize?: number;
  imagePatchSize?: number;
  maxImagePatches?: number;
  videoFrameLimit?: number;
}

export interface TTSConfig {
  textEncoderHiddenSize?: number;
  acousticDecoderHiddenSize?: number;
  vocoderHiddenSize?: number;
  sampleRate?: number;
  streamingChunkSize?: number;
}

export interface TrainingConfig {
  trainingType: TrainingType;
  seqLen: number;
  globalBatchSize: number;
  microBatchSize: number;
  gradAccumSteps: number;
  targetTokensPerStep?: number;
  computeDtype: ComputeDtype;
  optimizerType: OptimizerType;
  optimizerOffload: boolean;
  activationCheckpointing: boolean;
  flashAttention: boolean;
  recomputation: RecomputationMode;
}

export interface InferenceConfig {
  inferenceMode: InferenceMode;
  engine: InferenceEngine;
  inputTokensAvg: number;
  inputTokensP95: number;
  outputTokensAvg: number;
  outputTokensP95: number;
  targetConcurrency: number;
  computeDtype: ComputeDtype;
  weightDtype: ComputeDtype;
  kvCacheDtype: ComputeDtype;
  continuousBatching: boolean;
  pagedKvCache: boolean;
  speculativeDecoding: boolean;
}

export interface ParallelConfig {
  tpSize: number;
  ppSize: number;
  dpSize: number;
  zeroStage: ZeROStage;
  cpSize: number;
  epSize: number;
}

export interface HardwareConfig {
  gpuType: GpuType;
  gpuMemoryGb: number;
  gpusPerNode: number;
  nodeCount: number;
  interconnectType: InterconnectType;
}

export interface TrainingResult {
  canRun: boolean;
  runnabilityLevel: RiskLevel;
  recommendedGpuCount: number;
  recommendedParallel: ParallelConfig;
  peakMemoryGb: number;
  memParamsGb: number;
  memGradsGb: number;
  memOptimizerGb: number;
  memActivationGb: number;
  memBufferGb: number;
  tokensPerSec: number;
  tokensPerSecMin: number;
  tokensPerSecMax: number;
  estimatedTime: string;
  oomRiskLevel: RiskLevel;
  communicationRiskLevel: RiskLevel;
  explanationTags: string[];
  recommendations: string[];
}

export interface InferenceResult {
  canDeploy: boolean;
  deployabilityLevel: RiskLevel;
  recommendedGpuCount: number;
  recommendedParallel: ParallelConfig;
  memWeightsGb: number;
  memKvCacheGb: number;
  memKvCacheMinGb: number;
  memKvCacheMaxGb: number;
  memRuntimeBufferGb: number;
  peakMemoryGb: number;
  safeConcurrency: number;
  throughputTokS: number;
  latencyRiskLevel: RiskLevel;
  oomRiskLevel: RiskLevel;
  explanationTags: string[];
  recommendations: string[];
}

export interface CompareResult {
  scenarioName: string;
  hardware: HardwareConfig;
  result: TrainingResult | InferenceResult;
}

export interface RiskAssessment {
  category: 'memory' | 'communication' | 'configuration';
  level: RiskLevel;
  title: string;
  description: string;
  suggestion: string;
  affectedComponent: string;
}

export interface HFConfig {
  model_type?: string;
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  torch_dtype?: string;
  hidden_act?: string;
  rms_norm_eps?: number;
  layer_norm_eps?: number;
  norm_type?: string;
  architectures?: string[];
  tie_word_embeddings?: boolean;
  rope_scaling?: {
    type: string;
    factor: number;
  };
  rope_theta?: number;
  rope_parameters?: {
    rope_theta?: number;
    rope_type?: string;
  };
  num_local_experts?: number;
  n_routed_experts?: number;
  num_experts?: number;
  n_shared_experts?: number;
  num_experts_per_tok?: number;
  expert_intermediate_size?: number;
  moe_intermediate_size?: number;
  shared_expert_intermediate_size?: number;
  first_k_dense_replace?: number;
  moe_layer_freq?: number;
  vision_config?: {
    hidden_size?: number;
    num_hidden_layers?: number;
    depth?: number;
    num_heads?: number;
    image_size?: number;
    patch_size?: number;
  };
  audio_config?: {
    hidden_size?: number;
    num_hidden_layers?: number;
  };
  text_config?: HFConfig;
}

export interface ParsedHFConfig {
  modelFamily: ModelFamily;
  modelConfig: Partial<ModelConfig>;
  moeConfig?: Partial<MoEConfig>;
  multimodalConfig?: Partial<MultimodalConfig>;
  unrecognizedFields: string[];
  warnings: string[];
}

export type PlannerInput = {
  model: ModelConfig;
  moe?: MoEConfig;
  multimodal?: MultimodalConfig;
  tts?: TTSConfig;
  hardware: HardwareConfig;
  parallel: ParallelConfig;
};

export type TrainPlannerInput = PlannerInput & {
  training: TrainingConfig;
};

export type InferPlannerInput = PlannerInput & {
  inference: InferenceConfig;
};

export const GPU_MEMORY_MAP: Record<GpuType, number> = {
  'A100-40G': 40,
  'A100-80G': 80,
  'H100-SXM': 80,
  'H100-PCIe': 80,
  'L40S': 48,
  'L20': 48,
  '4090': 24,
  'H20': 96,
};

export const GPU_FLOPS_MAP: Record<GpuType, { bf16: number; fp16: number; fp8: number }> = {
  'A100-40G': { bf16: 156, fp16: 312, fp8: 624 },
  'A100-80G': { bf16: 156, fp16: 312, fp8: 624 },
  'H100-SXM': { bf16: 989, fp16: 1979, fp8: 3958 },
  'H100-PCIe': { bf16: 756, fp16: 1513, fp8: 3026 },
  'L40S': { bf16: 362, fp16: 733, fp8: 1466 },
  'L20': { bf16: 238, fp16: 476, fp8: 952 },
  '4090': { bf16: 83, fp16: 165, fp8: 330 },
  'H20': { bf16: 148, fp16: 296, fp8: 592 },
};

export const GPU_BANDWIDTH_MAP: Record<GpuType, number> = {
  'A100-40G': 1555,
  'A100-80G': 2039,
  'H100-SXM': 3352,
  'H100-PCIe': 2000,
  'L40S': 864,
  'L20': 576,
  '4090': 1008,
  'H20': 4000,
};

export const MODEL_PRESETS: Record<string, Partial<ModelConfig> & { moe?: Partial<MoEConfig> }> = {
  'Llama-3-8B': {
    modelName: 'Llama 3 8B',
    modelFamily: 'dense',
    hiddenSize: 4096,
    numHiddenLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    intermediateSize: 14336,
    vocabSize: 128256,
    maxPositionEmbeddings: 8192,
    tieWordEmbeddings: false,
    usesLearnedPositionEmbeddings: false,
    normType: 'rmsnorm',
    attentionType: 'gqa',
    positionEncodingType: 'rope',
    ffnActivation: 'swiglu',
  },
  'Llama-3-70B': {
    modelName: 'Llama 3 70B',
    modelFamily: 'dense',
    hiddenSize: 8192,
    numHiddenLayers: 80,
    numAttentionHeads: 64,
    numKeyValueHeads: 8,
    intermediateSize: 28672,
    vocabSize: 128256,
    maxPositionEmbeddings: 8192,
    tieWordEmbeddings: false,
    usesLearnedPositionEmbeddings: false,
    normType: 'rmsnorm',
    attentionType: 'gqa',
    positionEncodingType: 'rope',
    ffnActivation: 'swiglu',
  },
  'Qwen2.5-72B': {
    modelName: 'Qwen2.5 72B',
    modelFamily: 'dense',
    hiddenSize: 8192,
    numHiddenLayers: 80,
    numAttentionHeads: 64,
    numKeyValueHeads: 8,
    intermediateSize: 29568,
    vocabSize: 152064,
    maxPositionEmbeddings: 131072,
    tieWordEmbeddings: false,
    usesLearnedPositionEmbeddings: false,
    normType: 'rmsnorm',
    attentionType: 'gqa',
    positionEncodingType: 'rope',
    ffnActivation: 'swiglu',
  },
  'Mixtral-8x7B': {
    modelName: 'Mixtral 8x7B',
    modelFamily: 'moe',
    hiddenSize: 4096,
    numHiddenLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    intermediateSize: 14336,
    vocabSize: 32000,
    maxPositionEmbeddings: 32768,
    tieWordEmbeddings: false,
    usesLearnedPositionEmbeddings: false,
    normType: 'rmsnorm',
    attentionType: 'gqa',
    positionEncodingType: 'rope',
    ffnActivation: 'swiglu',
    moe: {
      numLocalExperts: 8,
      numExpertsPerTok: 2,
    },
  },
  'DeepSeek-V3': {
    modelName: 'DeepSeek V3',
    modelFamily: 'moe',
    hiddenSize: 7168,
    numHiddenLayers: 61,
    numAttentionHeads: 128,
    numKeyValueHeads: 128,
    intermediateSize: 18432,
    vocabSize: 129280,
    maxPositionEmbeddings: 163840,
    tieWordEmbeddings: false,
    usesLearnedPositionEmbeddings: false,
    normType: 'rmsnorm',
    attentionType: 'mha',
    positionEncodingType: 'rope',
    ffnActivation: 'swiglu',
    moe: {
      numLocalExperts: 256,
      numExpertsPerTok: 8,
      numSharedExperts: 1,
      firstKDenseReplace: 3,
      moeLayerFrequency: 1,
      expertIntermediateSize: 2048,
    },
  },
};
