import type {
  MemoryByDtype,
  ModelBreakdown,
  ModelBreakdownDetail,
  ModelBreakdownSection,
} from '../../engine/model-breakdown.ts';
import type { Locale, ModelFamily } from '../../types';

export interface SkeletonBlock {
  id: string;
  label: string;
  note: string;
  paramCount: number;
  memoryGb: MemoryByDtype;
  details: ModelBreakdownDetail[];
  internalLabels: string[];
  flow: 'main' | 'side';
  tone: 'token' | 'position' | 'decoder' | 'output' | 'vision' | 'projector' | 'audio';
}

const SKELETON_COPY = {
  zh: {
    token: '输入嵌入',
    positionLearned: '显式位置编码',
    positionDefault: '旋转位置编码',
    positionRope: (ropeType: string) => `${ropeType} RoPE`,
    decoder: '主干重复层',
    vision: '图像特征编码',
    projector: '映射到主干',
    audio: '音频特征编码',
    output: '输出到词表',
  },
  en: {
    token: 'Input embedding',
    positionLearned: 'Explicit position encoding',
    positionDefault: 'Rotary position encoding',
    positionRope: (ropeType: string) => `${ropeType} RoPE`,
    decoder: 'Repeated core layers',
    vision: 'Image feature encoder',
    projector: 'Maps into the LLM',
    audio: 'Audio feature encoder',
    output: 'Vocabulary projection',
  },
} as const;

export function buildSkeletonBlocks(
  breakdown: ModelBreakdown,
  activeModel: {
    modelFamily: ModelFamily;
    numHiddenLayers: number;
    usesLearnedPositionEmbeddings?: boolean;
    ropeScaling?: { type: string; factor: number };
    normType?: 'rmsnorm' | 'layernorm';
    attentionType?: 'mha' | 'gqa' | 'mqa';
    positionEncodingType?: 'rope' | 'learned';
    ffnActivation: string;
  },
  activeMoe?: {
    numLocalExperts: number;
    numSharedExperts?: number;
  },
  activeMultimodal?: {
    visionHiddenSize?: number;
    audioHiddenSize?: number;
  },
  locale: Locale = 'zh',
): SkeletonBlock[] {
  const skeletonCopy = SKELETON_COPY[locale];
  const embeddingSection = breakdown.sections.find((section) => section.id === 'embedding');
  const decoderSection = breakdown.sections.find((section) => section.id === 'decoder');
  const routerSection = breakdown.sections.find((section) => section.id === 'moe-router');
  const expertsSection = breakdown.sections.find((section) => section.id === 'moe-experts');
  const visionSection = breakdown.auxiliarySections.find((section) => section.id === 'vision');
  const projectorSection = breakdown.auxiliarySections.find((section) => section.id === 'projector');
  const audioSection = breakdown.auxiliarySections.find((section) => section.id === 'audio');

  const tokenDetails = filterDetailsById(embeddingSection, ['token-embedding']);
  const positionDetails = filterDetailsById(embeddingSection, ['position-embedding']);
  const outputDetails = [
    ...filterDetailsById(embeddingSection, ['lm-head']),
    ...filterDetailsById(decoderSection, ['decoder-final-norm', 'moe-decoder-final-norm']),
  ];
  const decoderDetails = [
    ...decoderSection?.children.filter((detail) => !detail.id.includes('final-norm') && !detail.id.includes('layer-count')) || [],
    ...(routerSection ? [routerSection] : []),
    ...(expertsSection ? [expertsSection] : []),
  ].flatMap((detail) => ('children' in detail ? [detail] : [detail])) as Array<ModelBreakdownDetail | ModelBreakdownSection>;

  const blocks: SkeletonBlock[] = [
    createSkeletonBlock(
      'token-embedding',
      'Token Embedding',
      skeletonCopy.token,
      tokenDetails.reduce((sum, detail) => sum + detail.paramCount, 0),
      tokenDetails.reduce((sum, detail) => mergeMemory(sum, detail.memoryGb), emptyMemory()),
      tokenDetails.length > 0 ? tokenDetails : [virtualDetail('token-embedding-virtual', 'Token Embedding', 0)],
      ['Input Tokens', 'Token Embedding'],
      'main',
      'token',
    ),
    buildPositionSkeletonBlock(activeModel, positionDetails, locale),
    createSkeletonBlock(
      'decoder-stack',
      `Decoder Stack x ${activeModel.numHiddenLayers}`,
      skeletonCopy.decoder,
      (decoderSection?.paramCount || 0) + (routerSection?.paramCount || 0) + (expertsSection?.paramCount || 0),
      mergeMemory(
        mergeMemory(decoderSection?.memoryGb || emptyMemory(), routerSection?.memoryGb || emptyMemory()),
        expertsSection?.memoryGb || emptyMemory(),
      ),
      normalizeDetails(decoderDetails),
      buildDecoderInternalLabels(
        normalizeDetails(decoderDetails),
        activeModel,
        activeMoe,
      ),
      'main',
      'decoder',
    ),
    buildOutputSkeletonBlock(outputDetails, locale),
  ];

  if (activeMultimodal) {
    if (visionSection) {
      blocks.push(createSkeletonBlock(
        'vision-encoder',
        'Vision Encoder',
        skeletonCopy.vision,
        visionSection.paramCount,
        visionSection.memoryGb,
        visionSection.children,
        buildBranchConnectionLabels(visionSection.label, 'Decoder Stack'),
        'side',
        'vision',
      ));
    }
    if (projectorSection) {
      blocks.push(createSkeletonBlock(
        'projector',
        'Projector',
        skeletonCopy.projector,
        projectorSection.paramCount,
        projectorSection.memoryGb,
        projectorSection.children,
        buildBranchConnectionLabels(projectorSection.label, 'Decoder Stack'),
        'side',
        'projector',
      ));
    }
    if (audioSection) {
      blocks.push(createSkeletonBlock(
        'audio-encoder',
        'Audio Encoder',
        skeletonCopy.audio,
        audioSection.paramCount,
        audioSection.memoryGb,
        audioSection.children,
        buildBranchConnectionLabels(audioSection.label, 'Decoder Stack'),
        'side',
        'audio',
      ));
    }
  }

  return blocks;
}

function createSkeletonBlock(
  id: string,
  label: string,
  note: string,
  paramCount: number,
  memoryGb: MemoryByDtype,
  details: ModelBreakdownDetail[],
  internalLabels: string[],
  flow: 'main' | 'side',
  tone: SkeletonBlock['tone'],
): SkeletonBlock {
  return {
    id,
    label,
    note,
    paramCount,
    memoryGb,
    details,
    internalLabels,
    flow,
    tone,
  };
}

function normalizeDetails(details: Array<ModelBreakdownDetail | ModelBreakdownSection>): ModelBreakdownDetail[] {
  return details.map((detail) => ({
    id: detail.id,
    label: detail.label,
    paramCount: detail.paramCount,
    memoryGb: detail.memoryGb,
    note: detail.note,
  }));
}

function virtualDetail(id: string, label: string, paramCount: number): ModelBreakdownDetail {
  return {
    id,
    label,
    paramCount,
    memoryGb: emptyMemory(),
  };
}

function buildPositionSkeletonBlock(
  activeModel: {
    usesLearnedPositionEmbeddings?: boolean;
    ropeScaling?: { type: string; factor: number };
    positionEncodingType?: 'rope' | 'learned';
  },
  positionDetails: ModelBreakdownDetail[],
  locale: Locale = 'zh',
): SkeletonBlock {
  const skeletonCopy = SKELETON_COPY[locale];
  const hasLearnedPosition = positionDetails.length > 0;
  const positionKind = activeModel.positionEncodingType || (activeModel.usesLearnedPositionEmbeddings ? 'learned' : 'rope');
  const label = hasLearnedPosition || positionKind === 'learned'
    ? 'Position Embedding'
    : activeModel.usesLearnedPositionEmbeddings === false && activeModel.ropeScaling
      ? `RoPE (${activeModel.ropeScaling.type})`
      : activeModel.ropeScaling
        ? `RoPE (${activeModel.ropeScaling.type})`
        : 'RoPE / Position';
  const note = hasLearnedPosition || positionKind === 'learned'
    ? skeletonCopy.positionLearned
    : activeModel.ropeScaling
      ? skeletonCopy.positionRope(activeModel.ropeScaling.type)
      : skeletonCopy.positionDefault;

  return createSkeletonBlock(
    'rope-position',
    label,
    note,
    positionDetails.reduce((sum, detail) => sum + detail.paramCount, 0),
    positionDetails.reduce((sum, detail) => mergeMemory(sum, detail.memoryGb), emptyMemory()),
    positionDetails.length > 0 ? positionDetails : [virtualDetail('position-runtime', label, 0)],
    hasLearnedPosition ? ['Position Embedding', 'Decoder Stack'] : [label, 'Decoder Stack'],
    'main',
    'position',
  );
}

function buildDecoderInternalLabels(
  details: ModelBreakdownDetail[],
  activeModel: {
    ffnActivation: string;
    normType?: 'rmsnorm' | 'layernorm';
    attentionType?: 'mha' | 'gqa' | 'mqa';
  },
  activeMoe?: {
    numSharedExperts?: number;
  },
): string[] {
  return details.map((detail) => sanitizeModuleLabel(detail.label, activeModel, activeMoe));
}

function sanitizeModuleLabel(
  label: string,
  activeModel: {
    ffnActivation: string;
    normType?: 'rmsnorm' | 'layernorm';
    attentionType?: 'mha' | 'gqa' | 'mqa';
  },
  activeMoe?: {
    numSharedExperts?: number;
  },
): string {
  const normalized = label.replace(/\s+x\s+\d+$/i, '').trim();

  if (normalized === 'LayerNorm') {
    return activeModel.normType === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
  }
  if (normalized === 'Attention') {
    return `${(activeModel.attentionType || 'mha').toUpperCase()} Attention`;
  }
  if (normalized === 'Dense MLP' || normalized === 'MLP') {
    return `${activeModel.ffnActivation.toUpperCase()} MLP`;
  }
  if (normalized === 'MoE Router') {
    return 'Router';
  }
  if (normalized === 'MoE Experts') {
    return activeMoe?.numSharedExperts ? 'Shared + Routed Experts' : 'Routed Experts';
  }
  if (normalized === 'Final Norm') {
    return 'Final Norm';
  }

  return normalized;
}

function buildOutputSkeletonBlock(
  outputDetails: ModelBreakdownDetail[],
  locale: Locale = 'zh',
): SkeletonBlock {
  const outputLabels = outputDetails.length > 0
    ? outputDetails.map((detail) => sanitizeOutputLabel(detail.label))
    : ['LM Head', 'Output'];

  return createSkeletonBlock(
    'lm-head-output',
    outputLabels.join(' / '),
    SKELETON_COPY[locale].output,
    outputDetails.reduce((sum, detail) => sum + detail.paramCount, 0),
    outputDetails.reduce((sum, detail) => mergeMemory(sum, detail.memoryGb), emptyMemory()),
    outputDetails.length > 0 ? outputDetails : [virtualDetail('output-virtual', 'LM Head / Output', 0)],
    [...outputLabels, 'Output'],
    'main',
    'output',
  );
}

function sanitizeOutputLabel(label: string): string {
  return label.replace(/\s+x\s+\d+$/i, '').trim();
}

function buildBranchConnectionLabels(moduleLabel: string, sinkLabel: string): string[] {
  const sourceLabel = moduleLabel.includes('Audio')
    ? 'Audio Input'
    : moduleLabel.includes('Projector')
      ? 'Vision Features'
      : 'Image Input';
  return [sourceLabel, moduleLabel, sinkLabel];
}

function filterDetailsById(section: ModelBreakdownSection | undefined, ids: string[]) {
  if (!section) {
    return [];
  }

  return section.children.filter((detail) => ids.includes(detail.id));
}

function emptyMemory(): MemoryByDtype {
  return {
    fp32: 0,
    bf16: 0,
    fp16: 0,
    fp8: 0,
    int8: 0,
    int4: 0,
  };
}

function mergeMemory(base: MemoryByDtype, next: MemoryByDtype): MemoryByDtype {
  return {
    fp32: base.fp32 + next.fp32,
    bf16: base.bf16 + next.bf16,
    fp16: base.fp16 + next.fp16,
    fp8: base.fp8 + next.fp8,
    int8: base.int8 + next.int8,
    int4: base.int4 + next.int4,
  };
}

export function formatParams(paramCount: number): string {
  if (paramCount >= 1e9) {
    return `${(paramCount / 1e9).toFixed(2)}B`;
  }
  if (paramCount >= 1e6) {
    return `${(paramCount / 1e6).toFixed(1)}M`;
  }
  return paramCount.toLocaleString();
}

export function formatDetailShare(paramCount: number, totalParamCount: number): string {
  if (totalParamCount <= 0) {
    return '0%';
  }

  return `${((paramCount / totalParamCount) * 100).toFixed(1)}%`;
}

export function getDetailToneClass(detailId: string): 'router' | 'experts' | 'default' {
  if (detailId.includes('router')) {
    return 'router';
  }
  if (detailId.includes('expert')) {
    return 'experts';
  }
  return 'default';
}
