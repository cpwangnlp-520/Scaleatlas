import type {
  ModelConfig,
  InferenceConfig,
  ParallelConfig,
  HardwareConfig,
  InferenceResult,
  MoEConfig,
  RiskLevel,
  Locale,
} from '../types';
import { calculateParamCount, calculateMoEParamCount } from './memory.ts';
import { calculateInferenceThroughput, calculateKvCacheMemory } from './throughput.ts';
import { assessOOMRisk, assessKvCacheRisk, getOverallRiskLevel } from './risk.ts';
import { isPipelineParallelCompatible, isTensorParallelCompatible } from './parallel-constraints.ts';
import { RUNTIME_COPY } from '../content/runtimeCopy.ts';

export function planInference(
  config: ModelConfig,
  inference: InferenceConfig,
  hardware: HardwareConfig,
  parallel: ParallelConfig,
  moeConfig?: MoEConfig,
  locale: Locale = 'zh',
): InferenceResult {
  const copy = RUNTIME_COPY[locale].inference;
  const paramCount = moeConfig
    ? calculateMoEParamCount(config, moeConfig)
    : calculateParamCount(config);

  const { weightsMemoryGb, avgKvCacheGb, maxKvCacheGb, runtimeBufferGb, peakMemoryGb, maxPeakMemoryGb } =
    estimateInferenceMemoryPerGpu(config, inference, parallel, paramCount);
  
  const safeConcurrency = calculateSafeConcurrency(
    config,
    inference,
    parallel,
    hardware.gpuMemoryGb,
    weightsMemoryGb,
    runtimeBufferGb,
  );
  
  const throughput = calculateInferenceThroughput(
    config,
    parallel,
    hardware,
    safeConcurrency,
    inference.inputTokensAvg,
    inference.outputTokensAvg,
    moeConfig,
  );
  
  const oomRisk = assessOOMRisk(maxPeakMemoryGb, hardware.gpuMemoryGb);
  const kvRisk = assessKvCacheRisk(
    config,
    inference.inputTokensP95 + inference.outputTokensP95,
    inference.targetConcurrency,
  );
  
  const latencyRisk = assessLatencyRisk(
    inference.inputTokensP95,
    inference.outputTokensP95,
    parallel,
    hardware,
  );
  
  const recommendations: string[] = [];
  const explanationTags: string[] = [];
  
  if (safeConcurrency <= 0) {
    recommendations.push(copy.minDeployFail);
    explanationTags.push(copy.minDeployTag);
  } else if (safeConcurrency < inference.targetConcurrency) {
    recommendations.push(copy.reduceConcurrency(inference.targetConcurrency, safeConcurrency));
    explanationTags.push(copy.concurrencyTag);
  }
  
  if (inference.weightDtype !== inference.kvCacheDtype && inference.kvCacheDtype === 'int8') {
    recommendations.push(copy.int8Kv);
  }
  
  if (!inference.continuousBatching && inference.targetConcurrency > 8) {
    recommendations.push(copy.enableContinuousBatching);
    explanationTags.push(copy.continuousBatchingTag);
  }
  
  if (!inference.pagedKvCache && avgKvCacheGb > 10) {
    recommendations.push(copy.enablePagedKv);
    explanationTags.push(copy.pagedKvTag);
  }
  
  if (oomRisk !== 'green') {
    explanationTags.push(copy.memoryRiskTag);
  }
  if (kvRisk !== 'green') {
    explanationTags.push(copy.kvRiskTag);
  }
  
  const canDeploy = oomRisk !== 'red' && safeConcurrency > 0;
  const overallRisk = getRiskFromLevels([oomRisk, kvRisk, latencyRisk]);
  
  return {
    canDeploy,
    deployabilityLevel: overallRisk,
    recommendedGpuCount: parallel.tpSize * parallel.ppSize,
    recommendedParallel: parallel,
    memWeightsGb: Math.round(weightsMemoryGb * 10) / 10,
    memKvCacheGb: Math.round(avgKvCacheGb * 10) / 10,
    memKvCacheMinGb: Math.round(avgKvCacheGb * 0.8 * 10) / 10,
    memKvCacheMaxGb: Math.round(maxKvCacheGb * 10) / 10,
    memRuntimeBufferGb: runtimeBufferGb,
    peakMemoryGb: Math.round(peakMemoryGb * 10) / 10,
    safeConcurrency,
    throughputTokS: throughput,
    latencyRiskLevel: latencyRisk,
    oomRiskLevel: oomRisk,
    explanationTags,
    recommendations,
  };
}

function getBytesPerDtype(dtype: string): number {
  const map: Record<string, number> = {
    fp32: 4,
    bf16: 2,
    fp16: 2,
    fp8: 1,
    int8: 1,
    int4: 0.5,
  };
  return map[dtype] || 2;
}

function calculateSafeConcurrency(
  config: ModelConfig,
  inference: InferenceConfig,
  parallel: ParallelConfig,
  gpuMemoryGb: number,
  weightsMemoryGb: number,
  runtimeBufferGb: number,
): number {
  const availableForKv = gpuMemoryGb - weightsMemoryGb - runtimeBufferGb - 2;
  if (availableForKv <= 0) {
    return 0;
  }
  
  const modelParallel = Math.max(1, parallel.tpSize * parallel.ppSize);

  const kvPerToken = calculateKvCacheMemory(
    config,
    1,
    1,
    inference.kvCacheDtype,
  ) / modelParallel;
  
  const tokensPerRequest = inference.inputTokensAvg + inference.outputTokensAvg;
  const kvPerRequest = kvPerToken * tokensPerRequest;
  
  if (kvPerRequest <= 0) return 0;
  
  const safeConcurrency = Math.floor(availableForKv / kvPerRequest);

  return Math.max(0, safeConcurrency);
}

function assessLatencyRisk(
  inputTokens: number,
  outputTokens: number,
  parallel: ParallelConfig,
  hardware: HardwareConfig,
): RiskLevel {
  if (outputTokens > 4096 && parallel.tpSize < 4) {
    return 'orange';
  }
  
  if (inputTokens > 16384 && parallel.tpSize < 8) {
    return 'yellow';
  }
  
  return 'green';
}

function getRiskFromLevels(levels: RiskLevel[]): RiskLevel {
  if (levels.includes('red')) return 'red';
  if (levels.includes('orange')) return 'orange';
  if (levels.includes('yellow')) return 'yellow';
  return 'green';
}

export function recommendInferenceParallel(
  config: ModelConfig,
  hardware: HardwareConfig,
  inference: InferenceConfig,
  moeConfig?: MoEConfig,
): ParallelConfig {
  const { gpusPerNode, nodeCount, gpuMemoryGb } = hardware;
  const totalGpus = gpusPerNode * nodeCount;
  const paramCount = moeConfig
    ? calculateMoEParamCount(config, moeConfig)
    : calculateParamCount(config);

  const tpCandidates = getDivisors(Math.min(gpusPerNode, totalGpus)).filter((tpSize) =>
    isTensorParallelCompatible(config, tpSize)
  );
  const candidates: InferenceParallelCandidate[] = [];

  for (const tpSize of tpCandidates) {
    const maxPipelineStages = Math.max(1, Math.floor(totalGpus / tpSize));
    const ppCandidates = getDivisors(maxPipelineStages).filter((ppSize) =>
      isPipelineParallelCompatible(config, ppSize)
    );

    for (const ppSize of ppCandidates) {
      const parallel = {
        tpSize,
        ppSize,
        dpSize: 1,
        zeroStage: 'none' as const,
        cpSize: 1,
        epSize: 1,
      };
      const memory = estimateInferenceMemoryPerGpu(config, inference, parallel, paramCount);

      candidates.push({
        ...parallel,
        gpuCount: tpSize * ppSize,
        peakMemoryGb: memory.peakMemoryGb,
        maxPeakMemoryGb: memory.maxPeakMemoryGb,
        fits: memory.maxPeakMemoryGb <= gpuMemoryGb,
      });
    }
  }

  const bestCandidate = selectBestInferenceParallelCandidate(candidates);

  return {
    tpSize: bestCandidate.tpSize,
    ppSize: bestCandidate.ppSize,
    dpSize: bestCandidate.dpSize,
    zeroStage: bestCandidate.zeroStage,
    cpSize: 1,
    epSize: 1,
  };
}

interface InferenceParallelCandidate extends ParallelConfig {
  gpuCount: number;
  peakMemoryGb: number;
  maxPeakMemoryGb: number;
  fits: boolean;
}

function estimateInferenceMemoryPerGpu(
  config: ModelConfig,
  inference: InferenceConfig,
  parallel: ParallelConfig,
  paramCount: number,
) {
  const weightBytes = getBytesPerDtype(inference.weightDtype);
  const modelParallel = Math.max(1, parallel.tpSize * parallel.ppSize);
  const weightsMemoryGb = (paramCount * weightBytes) / modelParallel / (1024 ** 3);
  const avgKvCacheGb = calculateKvCacheMemory(
    config,
    inference.targetConcurrency,
    inference.inputTokensAvg + inference.outputTokensAvg,
    inference.kvCacheDtype,
  ) / modelParallel;
  const maxKvCacheGb = calculateKvCacheMemory(
    config,
    inference.targetConcurrency,
    inference.inputTokensP95 + inference.outputTokensP95,
    inference.kvCacheDtype,
  ) / modelParallel;
  const runtimeBufferGb = 2;

  return {
    weightsMemoryGb,
    avgKvCacheGb,
    maxKvCacheGb,
    runtimeBufferGb,
    peakMemoryGb: weightsMemoryGb + avgKvCacheGb + runtimeBufferGb,
    maxPeakMemoryGb: weightsMemoryGb + maxKvCacheGb + runtimeBufferGb,
  };
}

function getDivisors(value: number): number[] {
  const divisors: number[] = [];

  for (let current = 1; current <= value; current += 1) {
    if (value % current === 0) {
      divisors.push(current);
    }
  }

  return divisors;
}

function selectBestInferenceParallelCandidate(
  candidates: InferenceParallelCandidate[],
): InferenceParallelCandidate {
  const sorted = [...candidates].sort((left, right) => {
    const leftFits = left.fits ? 1 : 0;
    const rightFits = right.fits ? 1 : 0;
    if (leftFits !== rightFits) {
      return rightFits - leftFits;
    }

    if (left.fits && right.fits && left.gpuCount !== right.gpuCount) {
      return left.gpuCount - right.gpuCount;
    }

    if (left.ppSize !== right.ppSize) {
      return left.ppSize - right.ppSize;
    }

    if (left.tpSize !== right.tpSize) {
      return left.tpSize - right.tpSize;
    }

    if (left.maxPeakMemoryGb !== right.maxPeakMemoryGb) {
      return left.maxPeakMemoryGb - right.maxPeakMemoryGb;
    }

    return left.peakMemoryGb - right.peakMemoryGb;
  });

  return sorted[0];
}
