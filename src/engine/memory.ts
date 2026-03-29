import type {
  ModelConfig,
  TrainingConfig,
  ParallelConfig,
  HardwareConfig,
  MoEConfig,
  MultimodalConfig,
  TTSConfig,
  ComputeDtype,
  ZeROStage,
  RecomputationMode,
  FFNActivation,
} from '../types';

const BYTES_PER_DTYPE: Record<ComputeDtype, number> = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
};

export interface MemoryBreakdown {
  params: number;
  grads: number;
  optimizer: number;
  activation: number;
  buffer: number;
  multimodal?: MultimodalMemoryBreakdown;
  tts?: TTSMemoryBreakdown;
  total: number;
}

export interface ActivationMemoryBreakdown {
  attention: number;
  feedforward: number;
  layerNorm: number;
  dropout: number;
  projection: number;
  crossEntropy: number;
  total: number;
}

export function calculateParamCountSimple(h: number, L: number, s: number, v: number): number {
  const emb = h * (v + s);
  const oneLayer = 12 * h ** 2 + 13 * h;
  const other = 2 * h;
  const n = emb + L * oneLayer + other;
  return n;
}

export function calculateParamCount(config: ModelConfig): number {
  if (config.paramCountTotal > 0) {
    return config.paramCountTotal;
  }

  const {
    hiddenSize: h,
    numHiddenLayers: L,
    vocabSize: v,
    intermediateSize: i,
    numAttentionHeads,
    numKeyValueHeads,
    ffnActivation,
  } = config;

  const embeddings = calculateEmbeddingParams(config);
  const attentionParams = calculateAttentionParams(h, numAttentionHeads, numKeyValueHeads || numAttentionHeads);
  const feedforwardParams = calculateDenseFeedforwardParams(h, i, ffnActivation);
  const perLayerNormParams = 2 * h;
  const finalNormParams = h;

  return embeddings + L * (attentionParams + feedforwardParams + perLayerNormParams) + finalNormParams;
}

export function calculateMoEParamCount(config: ModelConfig, moeConfig: MoEConfig): number {
  if (config.paramCountTotal > 0) {
    return config.paramCountTotal;
  }

  return calculateMoEParamCountInternal(config, moeConfig, false);
}

export function calculateMoEActiveParamCount(config: ModelConfig, moeConfig: MoEConfig): number {
  return calculateMoEParamCountInternal(config, moeConfig, true);
}

function calculateMoEParamCountInternal(config: ModelConfig, moeConfig: MoEConfig, activeOnly: boolean): number {
  const {
    hiddenSize: h,
    numHiddenLayers: L,
    intermediateSize: denseInterSize,
    numAttentionHeads,
    numKeyValueHeads,
    ffnActivation,
  } = config;
  const {
    numLocalExperts,
    numExpertsPerTok,
    numSharedExperts = 0,
    sharedExpertIntermediateSize,
    firstKDenseReplace = 0,
    moeLayerFrequency = 1,
    expertIntermediateSize,
  } = moeConfig;

  const embeddings = calculateEmbeddingParams(config);
  const attentionParams = calculateAttentionParams(h, numAttentionHeads, numKeyValueHeads || numAttentionHeads);
  const denseFeedforwardParams = calculateDenseFeedforwardParams(h, denseInterSize, ffnActivation);
  const expertFeedforwardParams = calculateDenseFeedforwardParams(h, expertIntermediateSize || denseInterSize || h * 4, ffnActivation);
  const sharedExpertFeedforwardParams = numSharedExperts > 0
    ? calculateDenseFeedforwardParams(h, sharedExpertIntermediateSize || expertIntermediateSize || denseInterSize || h * 4, ffnActivation)
    : 0;
  const perLayerNormParams = 2 * h;
  const routerParams = h * numLocalExperts;
  const finalNormParams = h;

  let totalLayerParams = 0;

  for (let layer = 0; layer < L; layer += 1) {
    if (isDenseMoeLayer(layer, firstKDenseReplace, moeLayerFrequency)) {
      totalLayerParams += attentionParams + denseFeedforwardParams + perLayerNormParams;
      continue;
    }

    const activeExperts = activeOnly ? numExpertsPerTok : numLocalExperts;
    totalLayerParams += attentionParams + perLayerNormParams + routerParams + expertFeedforwardParams * activeExperts + sharedExpertFeedforwardParams * numSharedExperts;
  }

  return embeddings + totalLayerParams + finalNormParams;
}

function calculateEmbeddingParams(config: ModelConfig): number {
  const tokenEmbeddings = config.hiddenSize * config.vocabSize;
  const positionalEmbeddings = calculatePositionEmbeddingParams(config);
  const lmHead = config.tieWordEmbeddings === false ? config.hiddenSize * config.vocabSize : 0;
  return tokenEmbeddings + positionalEmbeddings + lmHead;
}

function calculatePositionEmbeddingParams(config: ModelConfig): number {
  if (config.usesLearnedPositionEmbeddings === false) {
    return 0;
  }

  return config.hiddenSize * (config.maxPositionEmbeddings || 0);
}

function calculateAttentionParams(hiddenSize: number, numAttentionHeads: number, numKeyValueHeads: number): number {
  const headDim = hiddenSize / numAttentionHeads;
  return 2 * hiddenSize * hiddenSize + 2 * hiddenSize * headDim * numKeyValueHeads;
}

function calculateDenseFeedforwardParams(hiddenSize: number, intermediateSize: number, activation: FFNActivation): number {
  const multiplier = activation === 'swiglu' ? 3 : 2;
  return multiplier * hiddenSize * intermediateSize;
}

function isDenseMoeLayer(layerIndex: number, firstKDenseReplace: number, moeLayerFrequency: number): boolean {
  if (layerIndex < firstKDenseReplace) {
    return true;
  }

  return ((layerIndex - firstKDenseReplace) % moeLayerFrequency) !== 0;
}

export function calculateParamsMemory(
  paramCount: number,
  dtype: ComputeDtype,
  tpSize: number,
  ppSize: number,
  dpSize: number,
  zeroStage: ZeROStage,
): number {
  const bytesPerParam = BYTES_PER_DTYPE[dtype];
  
  const usesTPPP = tpSize > 1 || ppSize > 1;
  const usesZeRO = zeroStage !== 'none';
  
  if (usesTPPP && usesZeRO) {
    // TP/PP 和 ZeRO 互斥，这里给出警告但按 TP/PP 优先计算
    console.warn('Warning: TP/PP and ZeRO are mutually exclusive. Using TP/PP calculation.');
  }
  
  if (usesTPPP) {
    // TP/PP 模式：模型切分到 TP*PP 张卡
    return (paramCount / tpSize / ppSize) * bytesPerParam;
  } else {
    // 纯 DP 模式：可使用 ZeRO
    if (zeroStage === '3') {
      return (paramCount / dpSize) * bytesPerParam;
    }
    return paramCount * bytesPerParam;
  }
}

export function calculateGradientsMemory(
  paramCount: number,
  dtype: ComputeDtype,
  tpSize: number,
  ppSize: number,
  dpSize: number,
  zeroStage: ZeROStage,
): number {
  const bytesPerParam = BYTES_PER_DTYPE[dtype];
  
  const usesTPPP = tpSize > 1 || ppSize > 1;
  
  if (usesTPPP) {
    // TP/PP 模式
    return (paramCount / tpSize / ppSize) * bytesPerParam;
  } else {
    // 纯 DP 模式 + ZeRO
    if (zeroStage === '2' || zeroStage === '3') {
      return (paramCount / dpSize) * bytesPerParam;
    }
    return paramCount * bytesPerParam;
  }
}

export function calculateOptimizerMemory(
  paramCount: number,
  dtype: ComputeDtype,
  optimizerType: string,
  tpSize: number,
  ppSize: number,
  dpSize: number,
  zeroStage: ZeROStage,
): number {
  let optimizerBytes = optimizerType === 'AdamW-8bit' ? 2 : 8;
  const isMixed = dtype === 'bf16' || dtype === 'fp16';
  if (isMixed) {
    optimizerBytes += 4;
  }
  
  const usesTPPP = tpSize > 1 || ppSize > 1;
  
  if (usesTPPP) {
    // TP/PP 模式：优化器状态也切分
    return (paramCount / tpSize / ppSize) * optimizerBytes;
  } else {
    // 纯 DP 模式 + ZeRO
    if (zeroStage !== 'none') {
      return (paramCount / dpSize) * optimizerBytes;
    }
    return paramCount * optimizerBytes;
  }
}

export function calculateActivationMemory(
  config: ModelConfig,
  training: TrainingConfig,
  parallel: ParallelConfig,
): ActivationMemoryBreakdown {
  const { hiddenSize: h, numHiddenLayers: L, numAttentionHeads: a, vocabSize: v, intermediateSize: h_ff, ffnActivation } = config;
  const { seqLen: s, microBatchSize: b, computeDtype, recomputation, flashAttention, activationCheckpointing } = training;
  const { tpSize: tp, ppSize: pp } = parallel;
  
  const effectiveRecomputation = activationCheckpointing && recomputation === 'none' ? 'selective' : recomputation;
  
  const mixed = computeDtype === 'bf16' || computeDtype === 'fp16';
  const bytesPerValue = mixed ? 2 : 4;
  const seq_parallel = false;
  
  const layersPerStage = Math.ceil(L / pp);
  
  // Flash Attention eliminates the O(s²) attention scores memory
  // Without Flash Attention: attention scores = (2*bytes + 1) * a * s² * b / tp
  // With Flash Attention: attention scores = 0 (computed in chunks, not stored)
  const attentionScoresMemory = flashAttention ? 0 : ((2 * bytesPerValue + 1) * a * s * s * b / tp);
  
  let oneLayerAttention: number;
  if (effectiveRecomputation === 'none' || effectiveRecomputation === 'full') {
    if (seq_parallel) {
      oneLayerAttention = s * b * h / tp * (bytesPerValue * 5 + 1) + attentionScoresMemory;
    } else {
      oneLayerAttention = s * b * h * (bytesPerValue * 4 / tp + bytesPerValue + 1) + attentionScoresMemory;
    }
  } else if (effectiveRecomputation === 'selective') {
    if (seq_parallel) {
      oneLayerAttention = s * b * h / tp * (bytesPerValue * 5 + 1);
    } else {
      oneLayerAttention = s * b * h * (bytesPerValue * 4 / tp + bytesPerValue + 1);
    }
  } else {
    oneLayerAttention = 0;
  }
  
  const ff_act = ffnActivation || 'swiglu';
  let oneLayerFeedforward: number;
  
  if (ff_act === 'relu') {
    if (seq_parallel) {
      oneLayerFeedforward = s * b * h * bytesPerValue / tp + s * b * h_ff * bytesPerValue / tp + s * b * h / tp;
    } else {
      oneLayerFeedforward = s * b * h * bytesPerValue + s * b * h_ff * bytesPerValue / tp + s * b * h;
    }
  } else if (ff_act === 'gelu') {
    if (seq_parallel) {
      oneLayerFeedforward = s * b * h * bytesPerValue / tp + s * b * h_ff * bytesPerValue / tp + s * b * h_ff * bytesPerValue / tp + s * b * h / tp;
    } else {
      oneLayerFeedforward = s * b * h * bytesPerValue + s * b * h_ff * bytesPerValue / tp + s * b * h_ff * bytesPerValue / tp + s * b * h;
    }
  } else {
    // swiglu
    if (seq_parallel) {
      oneLayerFeedforward = s * b * h * bytesPerValue / tp + s * b * h_ff * bytesPerValue / tp + s * b * h_ff * bytesPerValue * 3 / tp + s * b * h / tp;
    } else {
      oneLayerFeedforward = s * b * h * bytesPerValue + s * b * h_ff * bytesPerValue / tp + s * b * h_ff * bytesPerValue * 3 / tp + s * b * h;
    }
  }
  
  const layerNorm = seq_parallel ? s * b * h * bytesPerValue / tp : s * b * h * bytesPerValue;
  const inputDropout = seq_parallel ? s * b * h / tp : s * b * h;
  const outputLayerNorm = seq_parallel ? s * b * h * bytesPerValue / tp : s * b * h * bytesPerValue;
  const outputLayerProjection = seq_parallel ? s * b * h * bytesPerValue / tp : s * b * h * bytesPerValue;
  const outputCrossEntropy = seq_parallel ? s * b * v * 4 / tp : s * b * v * 4;
  
  let totalActivation: number;
  
  if (effectiveRecomputation === 'full') {
    const layerInput = s * b * h * bytesPerValue * layersPerStage;
    totalActivation = layerInput + inputDropout + outputLayerNorm + outputLayerProjection + outputCrossEntropy;
    
    return {
      attention: 0,
      feedforward: 0,
      layerNorm: 0,
      dropout: inputDropout,
      projection: outputLayerNorm + outputLayerProjection,
      crossEntropy: outputCrossEntropy,
      total: totalActivation,
    };
  }
  
  const oneLayerTotal = oneLayerAttention + oneLayerFeedforward + 2 * layerNorm;
  
  totalActivation = oneLayerTotal * layersPerStage + inputDropout + outputLayerNorm + outputLayerProjection + outputCrossEntropy;
  
  return {
    attention: oneLayerAttention * layersPerStage,
    feedforward: oneLayerFeedforward * layersPerStage,
    layerNorm: layerNorm * 2 * layersPerStage,
    dropout: inputDropout,
    projection: outputLayerNorm + outputLayerProjection,
    crossEntropy: outputCrossEntropy,
    total: totalActivation,
  };
}

export function calculateLoraParamCount(hiddenSize: number, numLayers: number, rank: number = 16): number {
  return 4 * 2 * hiddenSize * rank * numLayers;
}

export interface MultimodalMemoryBreakdown {
  visionParams: number;
  visionActivation: number;
  audioParams: number;
  audioActivation: number;
  projectorParams: number;
  projectorActivation: number;
  total: number;
}

export function calculateVisionEncoderParams(
  hiddenSize: number,
  numLayers: number,
  numAttentionHeads: number,
  patchSize: number = 14,
  imageSize: number = 336,
): number {
  const numPatches = Math.floor(imageSize / patchSize) ** 2;
  const patchEmbedding = hiddenSize * (patchSize ** 2 * 3);
  const posEmbedding = hiddenSize * (numPatches + 1);
  const clsToken = hiddenSize;
  
  const headDim = hiddenSize / numAttentionHeads;
  const oneLayerAttn = 4 * hiddenSize * hiddenSize;
  const oneLayerFFN = hiddenSize * (hiddenSize * 4) * 2;
  const oneLayerNorm = 2 * hiddenSize;
  const oneLayer = oneLayerAttn + oneLayerFFN + oneLayerNorm;
  
  return patchEmbedding + posEmbedding + clsToken + oneLayer * numLayers;
}

export function calculateVisionEncoderActivation(
  hiddenSize: number,
  numLayers: number,
  numAttentionHeads: number,
  numPatches: number,
  batchSize: number,
  bytesPerValue: number,
  flashAttention: boolean = true,
): number {
  const s = numPatches + 1;
  const b = batchSize;
  const h = hiddenSize;
  const a = numAttentionHeads;
  
  let oneLayerAttention: number;
  if (flashAttention) {
    oneLayerAttention = s * b * h * (bytesPerValue * 4 + bytesPerValue + 1);
  } else {
    oneLayerAttention = s * b * h * (bytesPerValue * 4 + bytesPerValue + 1) + (2 * bytesPerValue + 1) * a * s * s * b;
  }
  
  const h_ff = h * 4;
  const oneLayerFeedforward = s * b * h * bytesPerValue + s * b * h_ff * bytesPerValue * 2 + s * b * h;
  const layerNorm = s * b * h * bytesPerValue;
  
  return (oneLayerAttention + oneLayerFeedforward + 2 * layerNorm) * numLayers;
}

export function calculateAudioEncoderParams(
  hiddenSize: number,
  numLayers: number,
  numAttentionHeads: number,
  inputDim: number = 80,
  maxSeqLen: number = 3000,
): number {
  const convLayers = 2;
  const convParams = inputDim * hiddenSize * 3 * convLayers;
  const posEmbedding = hiddenSize * maxSeqLen;
  
  const headDim = hiddenSize / numAttentionHeads;
  const oneLayerAttn = 4 * hiddenSize * hiddenSize;
  const oneLayerFFN = hiddenSize * (hiddenSize * 4) * 2;
  const oneLayerNorm = 2 * hiddenSize;
  const oneLayer = oneLayerAttn + oneLayerFFN + oneLayerNorm;
  
  return convParams + posEmbedding + oneLayer * numLayers;
}

export function calculateProjectorParams(
  visionHiddenSize: number,
  llmHiddenSize: number,
  projectorHiddenSize?: number,
  projectorType: 'linear' | 'mlp' | 'qformer' = 'mlp',
): number {
  if (projectorType === 'linear') {
    return visionHiddenSize * llmHiddenSize;
  }
  
  if (projectorType === 'qformer') {
    const numQFormerLayers = 2;
    const qFormerHidden = projectorHiddenSize || 768;
    const oneLayer = 4 * qFormerHidden * qFormerHidden + qFormerHidden * qFormerHidden * 4 * 2 + qFormerHidden * 4;
    return visionHiddenSize * qFormerHidden + llmHiddenSize * qFormerHidden + oneLayer * numQFormerLayers;
  }
  
  const mid = projectorHiddenSize || Math.max(visionHiddenSize, llmHiddenSize);
  return visionHiddenSize * mid + mid * llmHiddenSize + visionHiddenSize * llmHiddenSize;
}

export function calculateMultimodalMemory(
  multimodalConfig: MultimodalConfig,
  training: TrainingConfig,
  parallel: ParallelConfig,
): MultimodalMemoryBreakdown {
  const { visionEncoderType, visionHiddenSize, visionNumLayers, audioEncoderType, audioHiddenSize, projectorHiddenSize, imagePatchSize, maxImagePatches } = multimodalConfig;
  
  const mixed = training.computeDtype === 'bf16' || training.computeDtype === 'fp16';
  const bytesPerValue = mixed ? 2 : 4;
  const { tpSize } = parallel;
  
  let visionParams = 0;
  let visionActivation = 0;
  
  if (visionHiddenSize && visionNumLayers) {
    const visionHeads = Math.max(1, Math.floor(visionHiddenSize / 64));
    visionParams = calculateVisionEncoderParams(
      visionHiddenSize,
      visionNumLayers,
      visionHeads,
      imagePatchSize || 14,
      336
    );
    
    const numPatches = maxImagePatches || 256;
    visionActivation = calculateVisionEncoderActivation(
      visionHiddenSize,
      visionNumLayers,
      visionHeads,
      numPatches,
      training.microBatchSize,
      bytesPerValue,
      training.flashAttention
    );
  }
  
  let audioParams = 0;
  let audioActivation = 0;
  
  if (audioHiddenSize) {
    const audioLayers = 12;
    const audioHeads = Math.max(1, Math.floor(audioHiddenSize / 64));
    audioParams = calculateAudioEncoderParams(audioHiddenSize, audioLayers, audioHeads);
    
    const audioSeqLen = 1500;
    audioActivation = (audioSeqLen * training.microBatchSize * audioHiddenSize * bytesPerValue * 10) * audioLayers;
  }
  
  let projectorParams = 0;
  let projectorActivation = 0;
  
  if (visionHiddenSize) {
    projectorParams = calculateProjectorParams(
      visionHiddenSize,
      4096,
      projectorHiddenSize,
      'mlp'
    );
    projectorActivation = (training.microBatchSize * 256 * 4096 * bytesPerValue * 2);
  }
  
  const total = visionParams * bytesPerValue / tpSize + visionActivation / tpSize +
                audioParams * bytesPerValue / tpSize + audioActivation / tpSize +
                projectorParams * bytesPerValue / tpSize + projectorActivation / tpSize;
  
  return {
    visionParams: visionParams * bytesPerValue / tpSize / (1024 ** 3),
    visionActivation: visionActivation / tpSize / (1024 ** 3),
    audioParams: audioParams * bytesPerValue / tpSize / (1024 ** 3),
    audioActivation: audioActivation / tpSize / (1024 ** 3),
    projectorParams: projectorParams * bytesPerValue / tpSize / (1024 ** 3),
    projectorActivation: projectorActivation / tpSize / (1024 ** 3),
    total: total / (1024 ** 3),
  };
}

export interface TTSMemoryBreakdown {
  textEncoderParams: number;
  textEncoderActivation: number;
  acousticDecoderParams: number;
  acousticDecoderActivation: number;
  vocoderParams: number;
  vocoderActivation: number;
  total: number;
}

export function calculateTTSMemory(
  ttsConfig: TTSConfig,
  training: TrainingConfig,
  parallel: ParallelConfig,
): TTSMemoryBreakdown {
  const { textEncoderHiddenSize, acousticDecoderHiddenSize, vocoderHiddenSize } = ttsConfig;
  
  const mixed = training.computeDtype === 'bf16' || training.computeDtype === 'fp16';
  const bytesPerValue = mixed ? 2 : 4;
  const { tpSize } = parallel;
  
  let textEncoderParams = 0;
  let textEncoderActivation = 0;
  if (textEncoderHiddenSize) {
    textEncoderParams = textEncoderHiddenSize * textEncoderHiddenSize * 4 * 6;
    textEncoderActivation = training.seqLen * training.microBatchSize * textEncoderHiddenSize * bytesPerValue * 12;
  }
  
  let acousticDecoderParams = 0;
  let acousticDecoderActivation = 0;
  if (acousticDecoderHiddenSize) {
    acousticDecoderParams = acousticDecoderHiddenSize * acousticDecoderHiddenSize * 4 * 6;
    const acousticFrames = training.seqLen * 4;
    acousticDecoderActivation = acousticFrames * training.microBatchSize * acousticDecoderHiddenSize * bytesPerValue * 12;
  }
  
  let vocoderParams = 0;
  let vocoderActivation = 0;
  if (vocoderHiddenSize) {
    vocoderParams = vocoderHiddenSize * vocoderHiddenSize * 3 * 8;
    vocoderActivation = training.seqLen * 256 * training.microBatchSize * vocoderHiddenSize * bytesPerValue;
  }
  
  const total = (textEncoderParams + acousticDecoderParams + vocoderParams) * bytesPerValue / tpSize +
                (textEncoderActivation + acousticDecoderActivation + vocoderActivation) / tpSize;
  
  return {
    textEncoderParams: textEncoderParams * bytesPerValue / tpSize / (1024 ** 3),
    textEncoderActivation: textEncoderActivation / tpSize / (1024 ** 3),
    acousticDecoderParams: acousticDecoderParams * bytesPerValue / tpSize / (1024 ** 3),
    acousticDecoderActivation: acousticDecoderActivation / tpSize / (1024 ** 3),
    vocoderParams: vocoderParams * bytesPerValue / tpSize / (1024 ** 3),
    vocoderActivation: vocoderActivation / tpSize / (1024 ** 3),
    total: total / (1024 ** 3),
  };
}

export function calculateBufferMemory(
  config: ModelConfig,
  parallel: ParallelConfig,
  moeConfig?: MoEConfig,
): number {
  const { hiddenSize: h, numHiddenLayers: L } = config;
  const { tpSize, ppSize, epSize } = parallel;
  
  const communicationBuffer = h * L * 2 * 4;
  const tempBuffer = 100 * 1024 * 1024;
  
  let epBuffer = 0;
  if (moeConfig && epSize > 1) {
    const expertInterSize = moeConfig.expertIntermediateSize || config.intermediateSize || h * 4;
    epBuffer = h * expertInterSize * 2 * moeConfig.numLocalExperts / epSize * 4;
  }
  
  return communicationBuffer + tempBuffer + epBuffer;
}

export function calculateEPMemory(
  moeConfig: MoEConfig,
  hiddenSize: number,
  epSize: number,
): number {
  if (epSize <= 1) return 0;
  
  const expertInterSize = moeConfig.expertIntermediateSize || hiddenSize * 4;
  const expertParams = 3 * hiddenSize * expertInterSize * moeConfig.numLocalExperts;
  const allToAllBuffer = hiddenSize * moeConfig.numExpertsPerTok * 4 * 2;
  
  return expertParams / epSize * 2 + allToAllBuffer;
}

export function calculateTotalMemory(
  config: ModelConfig,
  training: TrainingConfig,
  parallel: ParallelConfig,
  hardware: HardwareConfig,
  moeConfig?: MoEConfig,
  multimodalConfig?: MultimodalConfig,
  ttsConfig?: TTSConfig,
  loraRank?: number,
): MemoryBreakdown {
  const paramCount = moeConfig
    ? calculateMoEParamCount(config, moeConfig)
    : calculateParamCount(config);
  
  const isLora = training.trainingType === 'lora' || training.trainingType === 'qlora';
  const isQlora = training.trainingType === 'qlora';
  const loraR = loraRank || 16;
  
  let params: number;
  let grads: number;
  let optimizer: number;
  
  if (isLora) {
    const loraParamCount = calculateLoraParamCount(config.hiddenSize, config.numHiddenLayers, loraR);
    
    let mainModelDtype = isQlora ? 'int4' : training.computeDtype;
    const mainModelBytes = BYTES_PER_DTYPE[mainModelDtype] || 2;
    
    const mainModelParams = paramCount * mainModelBytes;
    const shardedMainModelParams = mainModelParams / parallel.tpSize / parallel.ppSize;
    
    const loraBytes = BYTES_PER_DTYPE[training.computeDtype];
    const loraOptimizerBytes = 12;
    
    params = shardedMainModelParams + loraParamCount * loraBytes / parallel.tpSize / parallel.ppSize;
    grads = loraParamCount * loraBytes / parallel.tpSize / parallel.ppSize;
    optimizer = loraParamCount * loraOptimizerBytes / parallel.tpSize / parallel.ppSize;
    
  } else {
    params = calculateParamsMemory(
      paramCount,
      training.computeDtype,
      parallel.tpSize,
      parallel.ppSize,
      parallel.dpSize,
      parallel.zeroStage,
    );
    
    grads = calculateGradientsMemory(
      paramCount,
      training.computeDtype,
      parallel.tpSize,
      parallel.ppSize,
      parallel.dpSize,
      parallel.zeroStage,
    );
    
    optimizer = calculateOptimizerMemory(
      paramCount,
      training.computeDtype,
      training.optimizerType,
      parallel.tpSize,
      parallel.ppSize,
      parallel.dpSize,
      parallel.zeroStage,
    );
  }
  
  const activationBreakdown = calculateActivationMemory(config, training, parallel);
  const activation = activationBreakdown.total;
  const buffer = calculateBufferMemory(config, parallel, moeConfig);
  
  let multimodalMemory: MultimodalMemoryBreakdown | undefined;
  let ttsMemory: TTSMemoryBreakdown | undefined;
  let extraMemory = 0;
  
  if (multimodalConfig && (multimodalConfig.visionHiddenSize || multimodalConfig.audioHiddenSize)) {
    multimodalMemory = calculateMultimodalMemory(multimodalConfig, training, parallel);
    extraMemory += multimodalMemory.total * (1024 ** 3);
  }
  
  if (ttsConfig && (ttsConfig.textEncoderHiddenSize || ttsConfig.acousticDecoderHiddenSize)) {
    ttsMemory = calculateTTSMemory(ttsConfig, training, parallel);
    extraMemory += ttsMemory.total * (1024 ** 3);
  }
  
  if (moeConfig && parallel.epSize > 1) {
    const epMem = calculateEPMemory(moeConfig, config.hiddenSize, parallel.epSize);
    extraMemory += epMem;
  }
  
  const total = params + grads + optimizer + activation + buffer + extraMemory;
  
  return {
    params: params / (1024 ** 3),
    grads: grads / (1024 ** 3),
    optimizer: optimizer / (1024 ** 3),
    activation: activation / (1024 ** 3),
    buffer: buffer / (1024 ** 3),
    multimodal: multimodalMemory,
    tts: ttsMemory,
    total: total / (1024 ** 3),
  };
}

export function bytesToGB(bytes: number): number {
  return bytes / (1024 ** 3);
}

export function GBToBytes(gb: number): number {
  return gb * (1024 ** 3);
}
