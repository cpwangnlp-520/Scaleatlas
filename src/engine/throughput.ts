import type {
  ModelConfig,
  TrainingConfig,
  ParallelConfig,
  HardwareConfig,
  GpuType,
  ComputeDtype,
  MoEConfig,
} from '../types';
import { GPU_FLOPS_MAP } from '../types/index.ts';
import { calculateParamCount, calculateMoEActiveParamCount, calculateMoEParamCount } from './memory.ts';

export function calculateTrainingThroughput(
  config: ModelConfig,
  training: TrainingConfig,
  parallel: ParallelConfig,
  hardware: HardwareConfig,
  moeConfig?: MoEConfig,
): { tokensPerSec: number; tokensPerSecMin: number; tokensPerSecMax: number } {
  const { seqLen, globalBatchSize, computeDtype, flashAttention } = training;
  const { tpSize, ppSize, dpSize } = parallel;
  const { gpuType, gpusPerNode, nodeCount } = hardware;
  
  const totalGpus = tpSize * ppSize * dpSize;
  const flopsPerToken = 6 * (moeConfig
    ? calculateMoEParamCount(config, moeConfig)
    : calculateParamCount(config));
  
  if (moeConfig) {
    const activeParamCount = calculateMoEActiveParamCount(config, moeConfig);
    const flopsPerTokenActive = 6 * activeParamCount;
    const throughput = calculateThroughputForFlops(
      flopsPerTokenActive,
      totalGpus,
      gpuType,
      computeDtype,
      seqLen,
      globalBatchSize,
      parallel,
      hardware,
    );
    return throughput;
  }
  
  return calculateThroughputForFlops(
    flopsPerToken,
    totalGpus,
    gpuType,
    computeDtype,
    seqLen,
    globalBatchSize,
    parallel,
    hardware,
  );
}

function calculateThroughputForFlops(
  flopsPerToken: number,
  totalGpus: number,
  gpuType: GpuType,
  computeDtype: ComputeDtype,
  seqLen: number,
  globalBatchSize: number,
  parallel: ParallelConfig,
  hardware: HardwareConfig,
): { tokensPerSec: number; tokensPerSecMin: number; tokensPerSecMax: number } {
  const gpuFlops = GPU_FLOPS_MAP[gpuType];
  const dtypeKey = (computeDtype === 'bf16' || computeDtype === 'fp16') ? computeDtype : 'fp16';
  const peakFlops = gpuFlops[dtypeKey] * 1e12;
  
  const { tpSize, ppSize } = parallel;
  const { interconnectType, gpusPerNode } = hardware;
  
  let efficiency = 0.4;
  
  if (interconnectType === 'NVLink') efficiency = 0.5;
  else if (interconnectType === 'IB' || interconnectType === 'RoCE') efficiency = 0.45;
  
  if (tpSize > gpusPerNode) efficiency *= 0.7;
  if (ppSize > 1) efficiency *= 0.85;
  
  if (seqLen > 32768) efficiency *= 0.9;
  if (seqLen > 65536) efficiency *= 0.85;
  
  efficiency = Math.max(0.25, Math.min(0.6, efficiency));
  
  const effectiveFlops = peakFlops * efficiency * totalGpus;
  const tokensPerSec = effectiveFlops / flopsPerToken;
  
  const tokensPerSecMin = tokensPerSec * 0.8;
  const tokensPerSecMax = tokensPerSec * 1.2;
  
  return {
    tokensPerSec: Math.round(tokensPerSec),
    tokensPerSecMin: Math.round(tokensPerSecMin),
    tokensPerSecMax: Math.round(tokensPerSecMax),
  };
}

export function estimateTrainingTime(
  tokensPerSec: number,
  totalTokens: number,
): string {
  const totalSeconds = totalTokens / tokensPerSec;
  
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  
  if (days > 0) {
    return `${days}天 ${hours}小时`;
  }
  if (hours > 0) {
    return `${hours}小时 ${minutes}分钟`;
  }
  return `${minutes}分钟`;
}

export function calculateInferenceThroughput(
  config: ModelConfig,
  parallel: ParallelConfig,
  hardware: HardwareConfig,
  batchSize: number,
  inputTokens: number,
  outputTokens: number,
  moeConfig?: MoEConfig,
): number {
  const { tpSize, ppSize } = parallel;
  const { gpuType } = hardware;
  
  const paramCount = moeConfig
    ? calculateMoEParamCount(config, moeConfig)
    : calculateParamCount(config);
  
  const gpuFlops = GPU_FLOPS_MAP[gpuType];
  const peakFlops = gpuFlops['bf16'] * 1e12;
  
  const efficiency = 0.3;
  const effectiveFlops = peakFlops * efficiency * tpSize;
  
  const flopsPerToken = 2 * paramCount;
  
  const prefillTime = (inputTokens * flopsPerToken) / effectiveFlops;
  const decodeTimePerToken = flopsPerToken / effectiveFlops;
  const totalDecodeTime = outputTokens * decodeTimePerToken;
  
  const latency = prefillTime + totalDecodeTime;
  const throughput = batchSize / latency;
  
  return Math.round(throughput);
}

export function calculateKvCacheMemory(
  config: ModelConfig,
  batchSize: number,
  seqLen: number,
  dtype: ComputeDtype,
): number {
  const { hiddenSize, numHiddenLayers, numKeyValueHeads, numAttentionHeads } = config;
  
  const bytesPerValue = dtype === 'fp32' ? 4 : dtype === 'bf16' || dtype === 'fp16' ? 2 : 1;
  const headDim = hiddenSize / numAttentionHeads;
  
  const kvCacheSize = 2 * numKeyValueHeads * headDim * numHiddenLayers * seqLen * batchSize * bytesPerValue;
  
  return kvCacheSize / (1024 ** 3);
}
