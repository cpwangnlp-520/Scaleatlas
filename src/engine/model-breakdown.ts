import type { ComputeDtype, FFNActivation, Locale, ModelConfig, MoEConfig, MultimodalConfig } from '../types';
import {
  calculateAudioEncoderParams,
  calculateProjectorParams,
  calculateVisionEncoderParams,
} from './memory.ts';

const DTYPES: ComputeDtype[] = ['fp32', 'bf16', 'fp16', 'fp8', 'int8', 'int4'];

const BYTES_PER_DTYPE: Record<ComputeDtype, number> = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
};

export interface MemoryByDtype {
  fp32: number;
  bf16: number;
  fp16: number;
  fp8: number;
  int8: number;
  int4: number;
}

export interface ModelBreakdownDetail {
  id: string;
  label: string;
  paramCount: number;
  memoryGb: MemoryByDtype;
  note?: string;
}

export interface ModelBreakdownSection extends ModelBreakdownDetail {
  percentage: number;
  children: ModelBreakdownDetail[];
}

export interface ModelBreakdown {
  totalParamCount: number;
  auxiliaryParamCount: number;
  totalMemoryGb: MemoryByDtype;
  auxiliaryMemoryGb: MemoryByDtype;
  sections: ModelBreakdownSection[];
  auxiliarySections: ModelBreakdownSection[];
}

const MODEL_BREAKDOWN_COPY: Record<Locale, any> = {
  zh: {
    denseAttention: '包含 q / k / v / o 投影',
    denseMlp: '按激活函数选择 ReLU / GELU / SwiGLU 公式',
    denseSection: '每层展示 Attention、MLP 和归一化的聚合占用',
    moeAttention: 'MoE 层和 dense 层都包含 attention',
    moeDenseRetained: '仅统计保留 dense FFN 的层',
    moeDenseMissing: '当前模型没有保留 dense FFN 层',
    moeLayerCount: (count: number) => `其余 ${count} 层的 FFN 参数在下方 Router / Experts 中单独展示`,
    moeSection: 'MoE 模型将专家与 router 参数拆出单独展示',
    router: (hiddenSize: number, expertCount: number) => `每层 router 大小约为 hidden_size x experts = ${hiddenSize} x ${expertCount}`,
    routedExperts: (topK: number) => `单层激活 top-${topK} experts，但总参数按全部 experts 计入`,
    sharedExpertsEnabled: '共享 experts 在每个 MoE 层都常驻',
    sharedExpertsMissing: '当前模型没有 shared experts',
    vision: '额外视觉编码器参数，不计入当前主干语言模型总参数',
    audio: '按默认 12 层音频编码器估算',
    projector: '负责将视觉特征映射到语言模型隐层空间',
  },
  en: {
    denseAttention: 'Includes q / k / v / o projections',
    denseMlp: 'Uses the FFN formula for ReLU / GELU / SwiGLU',
    denseSection: 'Aggregates attention, MLP, and normalization for each layer',
    moeAttention: 'Both MoE and dense layers include attention',
    moeDenseRetained: 'Only counts layers that keep dense FFN blocks',
    moeDenseMissing: 'This model does not keep dense FFN layers',
    moeLayerCount: (count: number) => `The remaining ${count} layers expose FFN parameters below as Router / Experts`,
    moeSection: 'Splits router and expert parameters out of the decoder view',
    router: (hiddenSize: number, expertCount: number) => `Per-layer router size is about hidden_size x experts = ${hiddenSize} x ${expertCount}`,
    routedExperts: (topK: number) => `Each layer activates top-${topK} experts, while total parameters still count all experts`,
    sharedExpertsEnabled: 'Shared experts stay resident in every MoE layer',
    sharedExpertsMissing: 'This model does not use shared experts',
    vision: 'Extra vision encoder parameters, excluded from the language backbone total',
    audio: 'Estimated with a default 12-layer audio encoder',
    projector: 'Maps vision features into the language model hidden space',
  },
};

export function getModelBreakdown(
  config: ModelConfig,
  moeConfig?: MoEConfig,
  multimodalConfig?: MultimodalConfig,
  locale: Locale = 'zh',
): ModelBreakdown {
  const copy = MODEL_BREAKDOWN_COPY[locale];
  const mainSections = moeConfig
    ? buildMoeSections(config, moeConfig, copy)
    : buildDenseSections(config, copy);
  const totalParamCount = sumSections(mainSections);
  const sections = withPercentages(mainSections, totalParamCount);

  const auxiliarySections = multimodalConfig
    ? buildMultimodalSections(config, multimodalConfig, copy)
    : [];
  const auxiliaryParamCount = sumSections(auxiliarySections);

  return {
    totalParamCount,
    auxiliaryParamCount,
    totalMemoryGb: toMemoryMap(totalParamCount),
    auxiliaryMemoryGb: toMemoryMap(auxiliaryParamCount),
    sections,
    auxiliarySections: withPercentages(auxiliarySections, auxiliaryParamCount || 1),
  };
}

function buildDenseSections(config: ModelConfig, copy: any): ModelBreakdownSection[] {
  const embeddings = buildEmbeddingSection(config);
  const decoder = buildDenseDecoderSection(config, copy);

  return [embeddings, decoder];
}

function buildMoeSections(config: ModelConfig, moeConfig: MoEConfig, copy: any): ModelBreakdownSection[] {
  const embeddings = buildEmbeddingSection(config);
  const decoder = buildMoeDecoderSection(config, moeConfig, copy);
  const router = buildMoeRouterSection(config, moeConfig, copy);
  const experts = buildMoeExpertsSection(config, moeConfig, copy);

  return [embeddings, decoder, router, experts];
}

function buildEmbeddingSection(config: ModelConfig): ModelBreakdownSection {
  const tokenEmbeddings = config.hiddenSize * config.vocabSize;
  const positionalEmbeddings = config.usesLearnedPositionEmbeddings === false
    ? 0
    : config.hiddenSize * (config.maxPositionEmbeddings || 0);
  const lmHead = config.tieWordEmbeddings === false ? config.hiddenSize * config.vocabSize : 0;

  const children = [
    createDetail('token-embedding', 'Token Embedding', tokenEmbeddings),
    createDetail('position-embedding', 'Position Embedding', positionalEmbeddings),
    createDetail('lm-head', 'LM Head', lmHead),
  ].filter((detail) => detail.paramCount > 0);

  return createSection(
    'embedding',
    'Embedding',
    children.reduce((acc, detail) => acc + detail.paramCount, 0),
    children,
  );
}

function buildDenseDecoderSection(config: ModelConfig, copy: any): ModelBreakdownSection {
  const attentionPerLayer = calculateAttentionParams(
    config.hiddenSize,
    config.numAttentionHeads,
    config.numKeyValueHeads || config.numAttentionHeads,
  );
  const feedforwardPerLayer = calculateDenseFeedforwardParams(
    config.hiddenSize,
    config.intermediateSize,
    config.ffnActivation,
  );
  const layerNormPerLayer = 2 * config.hiddenSize;
  const finalNorm = config.hiddenSize;

  const children = [
    createDetail(
      'decoder-attention',
      `Attention x ${config.numHiddenLayers}`,
      attentionPerLayer * config.numHiddenLayers,
      copy.denseAttention,
    ),
    createDetail(
      'decoder-mlp',
      `MLP x ${config.numHiddenLayers}`,
      feedforwardPerLayer * config.numHiddenLayers,
      copy.denseMlp,
    ),
    createDetail(
      'decoder-norm',
      `LayerNorm x ${config.numHiddenLayers}`,
      layerNormPerLayer * config.numHiddenLayers,
    ),
    createDetail('decoder-final-norm', 'Final Norm', finalNorm),
  ];

  return createSection(
    'decoder',
    `Decoder Layers x ${config.numHiddenLayers}`,
    children.reduce((acc, detail) => acc + detail.paramCount, 0),
    children,
    copy.denseSection,
  );
}

function buildMoeDecoderSection(config: ModelConfig, moeConfig: MoEConfig, copy: any): ModelBreakdownSection {
  const attentionPerLayer = calculateAttentionParams(
    config.hiddenSize,
    config.numAttentionHeads,
    config.numKeyValueHeads || config.numAttentionHeads,
  );
  const denseFeedforwardPerLayer = calculateDenseFeedforwardParams(
    config.hiddenSize,
    config.intermediateSize,
    config.ffnActivation,
  );
  const denseLayerCount = countDenseLayers(config.numHiddenLayers, moeConfig);
  const moeLayerCount = config.numHiddenLayers - denseLayerCount;
  const layerNormPerLayer = 2 * config.hiddenSize;
  const finalNorm = config.hiddenSize;

  const children = [
    createDetail(
      'moe-decoder-attention',
      `Attention x ${config.numHiddenLayers}`,
      attentionPerLayer * config.numHiddenLayers,
      copy.moeAttention,
    ),
    createDetail(
      'moe-decoder-dense',
      denseLayerCount > 0 ? `Dense MLP x ${denseLayerCount}` : 'Dense MLP',
      denseFeedforwardPerLayer * denseLayerCount,
      denseLayerCount > 0 ? copy.moeDenseRetained : copy.moeDenseMissing,
    ),
    createDetail(
      'moe-decoder-norm',
      `LayerNorm x ${config.numHiddenLayers}`,
      layerNormPerLayer * config.numHiddenLayers,
    ),
    createDetail('moe-decoder-final-norm', 'Final Norm', finalNorm),
    createDetail(
      'moe-layer-count',
      `MoE Layers x ${moeLayerCount}`,
      0,
      copy.moeLayerCount(moeLayerCount),
    ),
  ];

  return createSection(
    'decoder',
    `Decoder Layers x ${config.numHiddenLayers}`,
    children.reduce((acc, detail) => acc + detail.paramCount, 0),
    children,
    copy.moeSection,
  );
}

function buildMoeRouterSection(config: ModelConfig, moeConfig: MoEConfig, copy: any): ModelBreakdownSection {
  const moeLayerCount = config.numHiddenLayers - countDenseLayers(config.numHiddenLayers, moeConfig);
  const routerPerLayer = config.hiddenSize * moeConfig.numLocalExperts;

  const children = [
    createDetail(
      'moe-router-total',
      `Router x ${moeLayerCount}`,
      routerPerLayer * moeLayerCount,
      copy.router(config.hiddenSize, moeConfig.numLocalExperts),
    ),
  ];

  return createSection(
    'moe-router',
    'MoE Router',
    children[0].paramCount,
    children,
  );
}

function buildMoeExpertsSection(config: ModelConfig, moeConfig: MoEConfig, copy: any): ModelBreakdownSection {
  const moeLayerCount = config.numHiddenLayers - countDenseLayers(config.numHiddenLayers, moeConfig);
  const routedExpertParams = calculateDenseFeedforwardParams(
    config.hiddenSize,
    moeConfig.expertIntermediateSize || config.intermediateSize,
    config.ffnActivation,
  );
  const sharedExpertParams = moeConfig.numSharedExperts
    ? calculateDenseFeedforwardParams(
        config.hiddenSize,
        moeConfig.sharedExpertIntermediateSize || moeConfig.expertIntermediateSize || config.intermediateSize,
        config.ffnActivation,
      )
    : 0;
  const routedExpertsTotal = routedExpertParams * moeConfig.numLocalExperts * moeLayerCount;
  const sharedExpertsTotal = sharedExpertParams * (moeConfig.numSharedExperts || 0) * moeLayerCount;

  const children = [
    createDetail(
      'moe-routed-experts',
      `Routed Experts x ${moeConfig.numLocalExperts}`,
      routedExpertsTotal,
      copy.routedExperts(moeConfig.numExpertsPerTok),
    ),
    createDetail(
      'moe-shared-experts',
      moeConfig.numSharedExperts ? `Shared Experts x ${moeConfig.numSharedExperts}` : 'Shared Experts',
      sharedExpertsTotal,
      moeConfig.numSharedExperts ? copy.sharedExpertsEnabled : copy.sharedExpertsMissing,
    ),
  ].filter((detail) => detail.paramCount > 0 || detail.note);

  return createSection(
    'moe-experts',
    'MoE Experts',
    children.reduce((acc, detail) => acc + detail.paramCount, 0),
    children,
  );
}

function buildMultimodalSections(
  config: ModelConfig,
  multimodalConfig: MultimodalConfig,
  copy: any,
): ModelBreakdownSection[] {
  const sections: ModelBreakdownSection[] = [];

  if (multimodalConfig.visionHiddenSize && multimodalConfig.visionNumLayers) {
    const visionHeads = Math.max(1, Math.floor(multimodalConfig.visionHiddenSize / 64));
    const visionParams = calculateVisionEncoderParams(
      multimodalConfig.visionHiddenSize,
      multimodalConfig.visionNumLayers,
      visionHeads,
      multimodalConfig.imagePatchSize || 14,
    );
    sections.push(createSection(
      'vision',
      'Vision Encoder',
      visionParams,
      [
        createDetail(
          'vision-stack',
          `Vision Blocks x ${multimodalConfig.visionNumLayers}`,
          visionParams,
          copy.vision,
        ),
      ],
    ));
  }

  if (multimodalConfig.audioHiddenSize) {
    const audioHeads = Math.max(1, Math.floor(multimodalConfig.audioHiddenSize / 64));
    const audioParams = calculateAudioEncoderParams(
      multimodalConfig.audioHiddenSize,
      12,
      audioHeads,
    );
    sections.push(createSection(
      'audio',
      'Audio Encoder',
      audioParams,
      [
        createDetail(
          'audio-stack',
          'Audio Blocks',
          audioParams,
          copy.audio,
        ),
      ],
    ));
  }

  if (multimodalConfig.visionHiddenSize) {
    const projectorParams = calculateProjectorParams(
      multimodalConfig.visionHiddenSize,
      config.hiddenSize,
      multimodalConfig.projectorHiddenSize,
    );
    sections.push(createSection(
      'projector',
      'Projector',
      projectorParams,
      [
        createDetail(
          'projector-stack',
          'Vision-to-LLM Projector',
          projectorParams,
          copy.projector,
        ),
      ],
    ));
  }

  return sections;
}

function createSection(
  id: string,
  label: string,
  paramCount: number,
  children: ModelBreakdownDetail[],
  note?: string,
): ModelBreakdownSection {
  return {
    id,
    label,
    paramCount,
    memoryGb: toMemoryMap(paramCount),
    percentage: 0,
    children,
    note,
  };
}

function createDetail(
  id: string,
  label: string,
  paramCount: number,
  note?: string,
): ModelBreakdownDetail {
  return {
    id,
    label,
    paramCount,
    memoryGb: toMemoryMap(paramCount),
    note,
  };
}

function withPercentages(
  sections: ModelBreakdownSection[],
  totalParamCount: number,
): ModelBreakdownSection[] {
  return sections.map((section) => ({
    ...section,
    percentage: totalParamCount > 0 ? section.paramCount / totalParamCount : 0,
  }));
}

function toMemoryMap(paramCount: number): MemoryByDtype {
  const memory = {} as MemoryByDtype;

  for (const dtype of DTYPES) {
    memory[dtype] = Math.round((paramCount * BYTES_PER_DTYPE[dtype]) / (1024 ** 3) * 10) / 10;
  }

  return memory;
}

function sumSections(sections: ModelBreakdownSection[]): number {
  return sections.reduce((acc, section) => acc + section.paramCount, 0);
}

function calculateAttentionParams(
  hiddenSize: number,
  numAttentionHeads: number,
  numKeyValueHeads: number,
): number {
  const headDim = hiddenSize / numAttentionHeads;
  return 2 * hiddenSize * hiddenSize + 2 * hiddenSize * headDim * numKeyValueHeads;
}

function calculateDenseFeedforwardParams(
  hiddenSize: number,
  intermediateSize: number,
  activation: FFNActivation,
): number {
  const multiplier = activation === 'swiglu' ? 3 : 2;
  return multiplier * hiddenSize * intermediateSize;
}

function countDenseLayers(totalLayers: number, moeConfig: MoEConfig): number {
  let denseLayers = 0;

  for (let layerIndex = 0; layerIndex < totalLayers; layerIndex += 1) {
    if (isDenseMoeLayer(layerIndex, moeConfig.firstKDenseReplace || 0, moeConfig.moeLayerFrequency || 1)) {
      denseLayers += 1;
    }
  }

  return denseLayers;
}

function isDenseMoeLayer(layerIndex: number, firstKDenseReplace: number, moeLayerFrequency: number): boolean {
  if (layerIndex < firstKDenseReplace) {
    return true;
  }

  return ((layerIndex - firstKDenseReplace) % moeLayerFrequency) !== 0;
}
