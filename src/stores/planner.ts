import { create } from 'zustand';
import type {
  ModelConfig,
  TrainingConfig,
  InferenceConfig,
  HardwareConfig,
  ParallelConfig,
  MoEConfig,
  MultimodalConfig,
  TTSConfig,
  TrainingResult,
  InferenceResult,
  ParsedHFConfig,
  Locale,
} from '../types/index.ts';
import { MODEL_PRESETS, GPU_MEMORY_MAP } from '../types/index.ts';

type ModelInputMode = 'custom' | 'config';

interface PlannerState {
  model: Partial<ModelConfig>;
  moe: Partial<MoEConfig>;
  multimodal: Partial<MultimodalConfig>;
  tts: Partial<TTSConfig>;
  locale: Locale;
  modelInputMode: ModelInputMode;
  importedConfigText: string;
  lastImportedConfig: ParsedHFConfig | null;
  hardware: HardwareConfig;
  training: TrainingConfig;
  inference: InferenceConfig;
  parallel: ParallelConfig;
  loraRank: number;
  trainingResult: TrainingResult | null;
  inferenceResult: InferenceResult | null;
  
  setModel: (model: Partial<ModelConfig>) => void;
  setMoe: (moe: Partial<MoEConfig>) => void;
  setMultimodal: (multimodal: Partial<MultimodalConfig>) => void;
  setTTS: (tts: Partial<TTSConfig>) => void;
  setLocale: (locale: Locale) => void;
  setModelInputMode: (mode: ModelInputMode) => void;
  setImportedConfigText: (text: string) => void;
  setHardware: (hardware: Partial<HardwareConfig>) => void;
  setTraining: (training: Partial<TrainingConfig>) => void;
  setInference: (inference: Partial<InferenceConfig>) => void;
  setParallel: (parallel: Partial<ParallelConfig>) => void;
  setLoraRank: (rank: number) => void;
  setTrainingResult: (result: TrainingResult | null) => void;
  setInferenceResult: (result: InferenceResult | null) => void;
  applyImportedConfig: (rawText: string, parsed: ParsedHFConfig) => void;
  clearImportedConfig: () => void;
  
  loadPreset: (presetName: string) => void;
  reset: () => void;
}

const defaultHardware: HardwareConfig = {
  gpuType: 'H100-SXM',
  gpuMemoryGb: 80,
  gpusPerNode: 8,
  nodeCount: 1,
  interconnectType: 'NVLink',
};

const defaultTraining: TrainingConfig = {
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

const defaultInference: InferenceConfig = {
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

const defaultParallel: ParallelConfig = {
  tpSize: 1,
  ppSize: 1,
  dpSize: 1,
  zeroStage: '1',
  cpSize: 1,
  epSize: 1,
};

function getImportedFieldState(parsed: ParsedHFConfig) {
  return {
    model: parsed.modelConfig,
    moe: parsed.moeConfig || {},
    multimodal: parsed.multimodalConfig || {},
    tts: {},
  };
}

export const usePlannerStore = create<PlannerState>((set) => ({
  model: {},
  moe: {},
  multimodal: {},
  tts: {},
  locale: 'zh',
  modelInputMode: 'custom',
  importedConfigText: '',
  lastImportedConfig: null,
  hardware: defaultHardware,
  training: defaultTraining,
  inference: defaultInference,
  parallel: defaultParallel,
  loraRank: 16,
  trainingResult: null,
  inferenceResult: null,
  
  setModel: (model) => set((state) => ({
    model: { ...state.model, ...model },
    modelInputMode: state.modelInputMode === 'config' ? 'custom' : state.modelInputMode,
  })),
  setMoe: (moe) => set((state) => ({
    moe: { ...state.moe, ...moe },
    modelInputMode: state.modelInputMode === 'config' ? 'custom' : state.modelInputMode,
  })),
  setMultimodal: (multimodal) => set((state) => ({
    multimodal: { ...state.multimodal, ...multimodal },
    modelInputMode: state.modelInputMode === 'config' ? 'custom' : state.modelInputMode,
  })),
  setTTS: (tts) => set((state) => ({
    tts: { ...state.tts, ...tts },
    modelInputMode: state.modelInputMode === 'config' ? 'custom' : state.modelInputMode,
  })),
  setLocale: (locale) => set({ locale }),
  setModelInputMode: (mode) => set((state) => {
    if (mode === 'config' && state.lastImportedConfig) {
      return {
        modelInputMode: mode,
        ...getImportedFieldState(state.lastImportedConfig),
      };
    }

    return { modelInputMode: mode };
  }),
  setImportedConfigText: (importedConfigText) => set({ importedConfigText }),
  setHardware: (hardware) => set((state) => {
    const newHardware = { ...state.hardware, ...hardware };
    if (hardware.gpuType && !hardware.gpuMemoryGb) {
      newHardware.gpuMemoryGb = GPU_MEMORY_MAP[hardware.gpuType];
    }
    return { hardware: newHardware };
  }),
  setTraining: (training) => set((state) => ({ training: { ...state.training, ...training } })),
  setInference: (inference) => set((state) => ({ inference: { ...state.inference, ...inference } })),
  setParallel: (parallel) => set((state) => ({ parallel: { ...state.parallel, ...parallel } })),
  setLoraRank: (loraRank) => set({ loraRank }),
  setTrainingResult: (trainingResult) => set({ trainingResult }),
  setInferenceResult: (inferenceResult) => set({ inferenceResult }),
  applyImportedConfig: (rawText, parsed) => set({
    importedConfigText: rawText,
    lastImportedConfig: parsed,
    modelInputMode: 'config',
    ...getImportedFieldState(parsed),
  }),
  clearImportedConfig: () => set({
    importedConfigText: '',
    lastImportedConfig: null,
    modelInputMode: 'custom',
  }),
  
  loadPreset: (presetName) => {
    const preset = MODEL_PRESETS[presetName];
    if (preset) {
      set({
        model: preset,
        moe: preset.moe || {},
        multimodal: {},
        tts: {},
        modelInputMode: 'custom',
      });
    }
  },
  
  reset: () => set({
    locale: 'zh',
    model: {},
    moe: {},
    multimodal: {},
    tts: {},
    hardware: defaultHardware,
    training: defaultTraining,
    inference: defaultInference,
    parallel: defaultParallel,
    loraRank: 16,
    modelInputMode: 'custom',
    importedConfigText: '',
    lastImportedConfig: null,
    trainingResult: null,
    inferenceResult: null,
  }),
}));
