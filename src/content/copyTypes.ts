import type { Locale } from '../types';

export type LocaleCopyMap<T> = Record<Locale, T>;

export interface AppChromeNavItem {
  to: string;
  label: string;
  tone: 'parameter' | 'train' | 'infer';
}

export interface AppChromeCopy {
  brandTitle: string;
  nav: AppChromeNavItem[];
}

export interface ParameterStepFlowCopy {
  pageTitle: string;
  steps: {
    model: string;
    result: string;
    advanced: string;
  };
}

export interface TrainStepFlowCopy {
  pageTitle: string;
  steps: {
    model: string;
    task: string;
    hardware: string;
    result: string;
  };
}

export interface InferStepFlowCopy {
  pageTitle: string;
  steps: {
    model: string;
    service: string;
    hardware: string;
    result: string;
  };
}

export interface StepFlowCopy {
  parameter: ParameterStepFlowCopy;
  train: TrainStepFlowCopy;
  infer: InferStepFlowCopy;
}

export interface SegmentedOptionCopy {
  value: string;
  label: string;
  note: string;
}

export interface ParameterPageCopy {
  pageTitle: string;
  pageDescription: string;
  modelSectionTitle: string;
  modelSectionSummary: string;
  modelSectionToggle: {
    open: string;
    close: string;
  };
  modeLabel: string;
  modeOptions: SegmentedOptionCopy[];
  presetLabel: string;
  presetPlaceholder: string;
  modelType: string;
  modelName: string;
  hiddenSize: string;
  layers: string;
  attentionHeads: string;
  kvHeads: string;
  intermediateSize: string;
  vocabSize: string;
  moeTitle: string;
  experts: string;
  expertsPerToken: string;
  multimodalTitle: string;
  visionHiddenSize: string;
  visionLayers: string;
  projectorHiddenSize: string;
  coreMetrics: {
    params: string;
    bf16: string;
    structure: string;
  };
  summary: {
    source: string;
    imported: string;
    manual: string;
    waiting: string;
  };
  architectureTitle: string;
  overviewTitle: string;
  overviewHint: string;
  skeletonLegend: string;
  branchLegend: string;
  clickHint: string;
  configSummaryFields: {
    layers: string;
    hidden: string;
    experts: string;
    multimodal: string;
  };
  skeletonDetailHeaders: {
    module: string;
    params: string;
    memory: string;
    share: string;
  };
  precisionTitle: string;
  advancedTitle: string;
  advancedToggle: {
    open: string;
    close: string;
  };
  invalidSharding: string;
  waitingStructure: string;
  waitingConstraint: string;
  structureTypes: {
    dense: string;
    moe: string;
    multimodal: string;
    tts: string;
  };
  candidateLabels: {
    tp: string;
    pp: string;
    ep: string;
  };
}

export interface TrainPageCopy {
  pageTitle: string;
  pageDescription: string;
  modelTitle: string;
  taskTitle: string;
  hardwareTitle: string;
  resultTitle: string;
  preset: string;
  modelType: string;
  modelName: string;
  hiddenSize: string;
  layers: string;
  attentionHeads: string;
  kvHeads: string;
  ffnDim: string;
  vocabSize: string;
  moeTitle: string;
  moeFields: {
    experts: string;
    expertsPerToken: string;
  };
  multimodalTitle: string;
  multimodalFields: {
    visionHiddenSize: string;
    visionLayers: string;
  };
  ttsTitle: string;
  ttsFields: {
    textEncoderHidden: string;
    acousticDecoderHidden: string;
  };
  trainingType: string;
  precision: string;
  seqLen: string;
  globalBatch: string;
  microBatch: string;
  optimizer: string;
  optimizerOptions: {
    flashAttention: string;
    optimizerOffload: string;
  };
  recomputation: string;
  loraTitle: string;
  loraRank: string;
  gpuType: string;
  gpuMemory: string;
  interconnect: string;
  gpusPerNode: string;
  nodeCount: string;
  totalGpu: string;
  advancedToggle: {
    open: string;
    close: string;
  };
  parallelFields: {
    tp: string;
    pp: string;
    dp: string;
    ep: string;
    zeroStage: string;
  };
  actionCopy: {
    fillModel: string;
    run: string;
    empty: string;
  };
  statusCopy: {
    fit: string;
    blocked: string;
    badgeFit: string;
    badgeBlocked: string;
  };
  parallelValidation: string;
  resultSummary: {
    canRun: string;
    peakMemory: string;
    parallel: string;
    throughput: string;
  };
  memoryBreakdown: string;
  memoryLabels: {
    params: string;
    grads: string;
    optimizer: string;
    activation: string;
  };
  mappingTitle: string;
  mappingDescription: string;
  recommendationTitle: string;
}

export interface InferPageCopy {
  pageTitle: string;
  pageDescription: string;
  modelTitle: string;
  serviceTitle: string;
  hardwareTitle: string;
  resultTitle: string;
  preset: string;
  modelType: string;
  modelName: string;
  hiddenSize: string;
  layers: string;
  attentionHeads: string;
  kvHeads: string;
  ffnDim: string;
  vocabSize: string;
  moeTitle: string;
  moeFields: {
    experts: string;
    expertsPerToken: string;
  };
  engine: string;
  mode: string;
  modeOptions: {
    offline: string;
    online: string;
    streaming: string;
  };
  weightDtype: string;
  kvDtype: string;
  inputAvg: string;
  inputP95: string;
  outputAvg: string;
  outputP95: string;
  concurrency: string;
  options: string;
  optionLabels: {
    continuousBatching: string;
    pagedKvCache: string;
    speculativeDecoding: string;
  };
  gpuType: string;
  gpuMemory: string;
  gpusPerNode: string;
  nodeCount: string;
  totalGpu: string;
  actionCopy: {
    fillModel: string;
    run: string;
    empty: string;
  };
  statusCopy: {
    fit: string;
    blocked: string;
    badgeFit: string;
    badgeBlocked: string;
  };
  resultSummary: {
    deploy: string;
    concurrency: string;
    peakMemory: string;
    parallel: string;
  };
  memoryLayers: {
    title: string;
    weight: string;
    kv: string;
    runtime: string;
  };
  mappingTitle: string;
  mappingDescription: string;
  recommendationTitle: string;
}

export interface RuntimeCopy {
  common: {
    yes: string;
    no: string;
    off: string;
    customModel: string;
    presetPlaceholder: string;
    buffer: string;
    cluster: string;
    parallel: string;
    used: string;
    node: string;
    idle: string;
    validation: string;
    totalGpu: string;
  };
  parser: {
    invalidJson: string;
    missingHiddenSize: string;
    missingNumHiddenLayers: string;
    missingRoutedExperts: string;
    multimodalNeedsVision: string;
  };
  parallelConstraints: {
    invalidTp: string;
    invalidPp: string;
    invalidEp: string;
    invalidDp: string;
    tpMustDivide: (tpSize: number, invalidTargets: string[]) => string;
    tpValid: (tpSize: number, hiddenSize: number, numAttentionHeads: number, numKeyValueHeads: number) => string;
    ppMustDivide: (ppSize: number, numHiddenLayers: number) => string;
    ppValid: (ppSize: number, numHiddenLayers: number) => string;
    epDisabled: string;
    epOnlyForMoe: (epSize: number) => string;
    epMustDivide: (epSize: number, expertCount: number) => string;
    epValid: (epSize: number, expertCount: number) => string;
    hardwareExceeded: (requestedGpus: number, totalAvailableGpus: number) => string;
    hardwareUsage: (requestedGpus: number, totalAvailableGpus: number) => string;
    tpCrossNode: (tpSize: number) => string;
    epCrossNode: (epSize: number) => string;
  };
  risk: {
    runnable: string;
    atRisk: string;
    highRisk: string;
    blocked: string;
    descriptions: Record<'green' | 'yellow' | 'orange' | 'red', string>;
    oomTitle: string;
    oomDescription: (peakMemoryGb: number, gpuMemoryGb: number) => string;
    oomSuggestion: string;
    commTitle: string;
    commDescription: (tpSize: number, ppSize: number) => string;
    commSuggestion: string;
    actTitle: string;
    actDescription: (seqLen: number) => string;
    actSuggestion: string;
    moeTitle: string;
    moeDescription: (expertCount?: number) => string;
    moeSuggestion: string;
  };
  training: {
    overMemory: (requiredGb: number, limitGb: number) => string;
    useTensorParallel: (tpSize: number) => string;
    enableActivationCheckpointing: string;
    recommendedSetup: (tpSize: number, ppSize: number) => string;
    useLora: string;
    reduceActivationMemory: string;
    tpCrossNode: string;
    enableEp: string;
    multimodalVision: string;
    ttsVocoder: string;
    keepHeadroom: string;
    qloraPrecision: string;
  };
  inference: {
    minDeployFail: string;
    minDeployTag: string;
    reduceConcurrency: (target: number, safe: number) => string;
    concurrencyTag: string;
    int8Kv: string;
    enableContinuousBatching: string;
    continuousBatchingTag: string;
    enablePagedKv: string;
    pagedKvTag: string;
    memoryRiskTag: string;
    kvRiskTag: string;
  };
  shared: {
    suggestionPrefix: string;
    recommendationTitle: string;
    recommendationDescription: string;
    parallelTitle: string;
    parallelDescription: string;
    zeroLabel: string;
    throughputLabel: string;
    throughputDescription: string;
    throughputRange: (min: number, max: number) => string;
    estimatedTime: (time: string) => string;
    machineTitle: string;
    machineDescription: string;
    legend: {
      weight: string;
      stage: string;
      replica: string;
      expert: string;
    };
    trainNote: string;
    inferNote: string;
    layersPerStage: (layersPerStage: number) => string;
    noPipeline: string;
    expertsPerShard: (expertCount: number, epSize: number) => string;
    noExpertParallel: string;
  };
  memoryBar: {
    title: string;
    description: string;
    params: string;
    grads: string;
    optimizer: string;
    activation: string;
    buffer: string;
    weights: string;
    kvCache: string;
  };
}
