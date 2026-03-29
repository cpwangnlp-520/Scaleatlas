import type {
  ModelConfig,
  TrainingConfig,
  ParallelConfig,
  HardwareConfig,
  TrainingResult,
  MoEConfig,
  MultimodalConfig,
  TTSConfig,
  Locale,
} from '../types';
import { calculateTotalMemory, calculateParamCount, calculateMoEParamCount } from './memory.ts';
import { hasExplicitParallelOverride } from './parallel-constraints.ts';
import { getAllRisks, getOverallRiskLevel, assessOOMRisk } from './risk.ts';
import { calculateTrainingThroughput, estimateTrainingTime } from './throughput.ts';
import { RUNTIME_COPY } from '../content/runtimeCopy.ts';

export function recommendParallelStrategy(
  config: ModelConfig,
  hardware: HardwareConfig,
  training: TrainingConfig,
  moeConfig?: MoEConfig,
  multimodalConfig?: MultimodalConfig,
  ttsConfig?: TTSConfig,
  loraRank?: number,
): ParallelConfig {
  const { gpuMemoryGb, gpusPerNode, nodeCount } = hardware;
  const totalGpus = gpusPerNode * nodeCount;
  
  const paramCount = moeConfig
    ? calculateMoEParamCount(config, moeConfig)
    : calculateParamCount(config);
  
  const paramSizeGB = (paramCount * 2) / (1024 ** 3);
  
  const candidates: ParallelCandidate[] = [];
  const tpCandidates = getDivisors(gpusPerNode);

  for (const tpSize of tpCandidates) {
    const remainingGpus = totalGpus / tpSize;
    const ppCandidates = getDivisors(remainingGpus);

    for (const ppSize of ppCandidates) {
      const dpSize = totalGpus / (tpSize * ppSize);
      const epSize = getRecommendedEPSize(moeConfig, gpusPerNode, dpSize);
      const zeroStage = getRecommendedZeroStage(tpSize, ppSize, config, training, hardware, moeConfig, multimodalConfig, ttsConfig, loraRank, dpSize, epSize);
      const estimatedMemory = estimateMemoryForConfig(
        config,
        training,
        { tpSize, ppSize, dpSize, zeroStage, cpSize: 1, epSize },
        hardware,
        moeConfig,
        multimodalConfig,
        ttsConfig,
        loraRank,
      );
      const throughput = calculateTrainingThroughput(
        config,
        training,
        { tpSize, ppSize, dpSize, zeroStage, cpSize: 1, epSize },
        hardware,
        moeConfig,
      ).tokensPerSec;

      candidates.push({
        tpSize,
        ppSize,
        dpSize,
        epSize,
        zeroStage,
        peakMemoryGb: estimatedMemory.total,
        throughput,
      });
    }
  }

  const bestCandidate = selectBestParallelCandidate(candidates, gpuMemoryGb);

  return {
    tpSize: bestCandidate.tpSize,
    ppSize: bestCandidate.ppSize,
    dpSize: bestCandidate.dpSize,
    zeroStage: bestCandidate.zeroStage,
    cpSize: 1,
    epSize: bestCandidate.epSize,
  };
}

interface ParallelCandidate {
  tpSize: number;
  ppSize: number;
  dpSize: number;
  epSize: number;
  zeroStage: 'none' | '1' | '2' | '3';
  peakMemoryGb: number;
  throughput: number;
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

function getRecommendedEPSize(
  moeConfig: MoEConfig | undefined,
  gpusPerNode: number,
  dpSize: number,
): number {
  if (!moeConfig || moeConfig.numLocalExperts < 8) {
    return 1;
  }

  return Math.min(moeConfig.numLocalExperts, gpusPerNode, dpSize);
}

function getRecommendedZeroStage(
  tpSize: number,
  ppSize: number,
  config: ModelConfig,
  training: TrainingConfig,
  hardware: HardwareConfig,
  moeConfig: MoEConfig | undefined,
  multimodalConfig: MultimodalConfig | undefined,
  ttsConfig: TTSConfig | undefined,
  loraRank: number | undefined,
  dpSize: number,
  epSize: number,
): 'none' | '1' | '2' | '3' {
  if (tpSize > 1 || ppSize > 1) {
    return 'none';
  }

  const stages: Array<'1' | '2' | '3'> = ['1', '2', '3'];

  for (const zeroStage of stages) {
    const estimatedMemory = estimateMemoryForConfig(
      config,
      training,
      { tpSize, ppSize, dpSize, zeroStage, cpSize: 1, epSize },
      hardware,
      moeConfig,
      multimodalConfig,
      ttsConfig,
      loraRank,
    );

    if (estimatedMemory.total <= hardware.gpuMemoryGb) {
      return zeroStage;
    }
  }

  return '3';
}

function selectBestParallelCandidate(
  candidates: ParallelCandidate[],
  gpuMemoryGb: number,
): ParallelCandidate {
  const sorted = [...candidates].sort((left, right) => {
    const leftFits = left.peakMemoryGb <= gpuMemoryGb ? 1 : 0;
    const rightFits = right.peakMemoryGb <= gpuMemoryGb ? 1 : 0;
    if (leftFits !== rightFits) {
      return rightFits - leftFits;
    }

    if (leftFits && rightFits && left.throughput !== right.throughput) {
      return right.throughput - left.throughput;
    }

    if (left.ppSize !== right.ppSize) {
      return left.ppSize - right.ppSize;
    }

    if (left.tpSize !== right.tpSize) {
      return left.tpSize - right.tpSize;
    }

    return left.peakMemoryGb - right.peakMemoryGb;
  });

  return sorted[0];
}

function estimateMemoryForConfig(
  config: ModelConfig,
  training: TrainingConfig,
  parallel: ParallelConfig,
  hardware: HardwareConfig,
  moeConfig?: MoEConfig,
  multimodalConfig?: MultimodalConfig,
  ttsConfig?: TTSConfig,
  loraRank?: number,
): { total: number } {
  const memory = calculateTotalMemory(config, training, parallel, hardware, moeConfig, multimodalConfig, ttsConfig, loraRank);
  return memory;
}

export function calculateMinGpuCount(
  config: ModelConfig,
  hardware: HardwareConfig,
  training: TrainingConfig,
  moeConfig?: MoEConfig,
): number {
  const paramCount = moeConfig
    ? calculateMoEParamCount(config, moeConfig)
    : calculateParamCount(config);
  
  const paramSizeGB = (paramCount * 2) / (1024 ** 3);
  
  const minTensoryParallel = Math.ceil(paramSizeGB / (hardware.gpuMemoryGb * 0.4));
  
  const minGpus = Math.max(1, minTensoryParallel);
  
  return minGpus;
}

export function planTraining(
  config: ModelConfig,
  training: TrainingConfig,
  hardware: HardwareConfig,
  parallel: ParallelConfig,
  moeConfig?: MoEConfig,
  multimodalConfig?: MultimodalConfig,
  ttsConfig?: TTSConfig,
  loraRank?: number,
  locale: Locale = 'zh',
): TrainingResult {
  const copy = RUNTIME_COPY[locale].training;
  const recommended = recommendParallelStrategy(config, hardware, training, moeConfig, multimodalConfig, ttsConfig, loraRank);
  const effectiveParallel = hasExplicitParallelOverride(parallel) ? parallel : recommended;
  
  const memory = calculateTotalMemory(config, training, effectiveParallel, hardware, moeConfig, multimodalConfig, ttsConfig, loraRank);
  const throughput = calculateTrainingThroughput(config, training, effectiveParallel, hardware, moeConfig);
  const risks = getAllRisks(config, training, effectiveParallel, hardware, moeConfig, locale);
  const overallRisk = getOverallRiskLevel(risks);
  
  const canRun = memory.total < hardware.gpuMemoryGb;
  
  const recommendations: string[] = [];
  
  if (!canRun) {
    const isLora = training.trainingType === 'lora' || training.trainingType === 'qlora';
    
    if (isLora) {
      recommendations.push(copy.overMemory(memory.total, hardware.gpuMemoryGb));
      if (parallel.tpSize < hardware.gpusPerNode) {
        recommendations.push(copy.useTensorParallel(hardware.gpusPerNode));
      }
      recommendations.push(copy.enableActivationCheckpointing);
    } else {
      recommendations.push(copy.overMemory(memory.total, hardware.gpuMemoryGb));
      recommendations.push(copy.recommendedSetup(recommended.tpSize, recommended.ppSize));
      if (recommended.tpSize * recommended.ppSize < hardware.gpusPerNode * hardware.nodeCount) {
        recommendations.push(copy.useLora);
      }
    }
  } else {
    if (memory.activation / memory.total > 0.4) {
      recommendations.push(copy.reduceActivationMemory);
    }
    
    if (parallel.tpSize > hardware.gpusPerNode) {
      recommendations.push(copy.tpCrossNode);
    }
  }
  
  if (moeConfig && parallel.epSize === 1 && moeConfig.numLocalExperts > 8) {
    recommendations.push(copy.enableEp);
  }
  
  if (multimodalConfig && multimodalConfig.visionHiddenSize) {
    recommendations.push(copy.multimodalVision);
  }
  
  if (ttsConfig && ttsConfig.textEncoderHiddenSize) {
    recommendations.push(copy.ttsVocoder);
  }
  
  if (memory.total / hardware.gpuMemoryGb > 0.9 && canRun) {
    recommendations.push(copy.keepHeadroom);
  }
  
  if (training.trainingType === 'qlora' && training.computeDtype !== 'bf16' && training.computeDtype !== 'fp16') {
    recommendations.push(copy.qloraPrecision);
  }
  
  const explanationTags: string[] = [];
  if (overallRisk !== 'green') {
    explanationTags.push(...risks.map(r => r.title));
  }
  
  const totalTokens = training.globalBatchSize * training.seqLen * 1000;
  const estimatedTime = estimateTrainingTime(throughput.tokensPerSec, totalTokens);
  
  return {
    canRun,
    runnabilityLevel: overallRisk,
    recommendedGpuCount: effectiveParallel.tpSize * effectiveParallel.ppSize * effectiveParallel.dpSize,
    recommendedParallel: effectiveParallel,
    peakMemoryGb: Math.round(memory.total * 10) / 10,
    memParamsGb: Math.round(memory.params * 10) / 10,
    memGradsGb: Math.round(memory.grads * 10) / 10,
    memOptimizerGb: Math.round(memory.optimizer * 10) / 10,
    memActivationGb: Math.round(memory.activation * 10) / 10,
    memBufferGb: Math.round(memory.buffer * 10) / 10,
    tokensPerSec: throughput.tokensPerSec,
    tokensPerSecMin: throughput.tokensPerSecMin,
    tokensPerSecMax: throughput.tokensPerSecMax,
    estimatedTime,
    oomRiskLevel: assessOOMRisk(memory.total, hardware.gpuMemoryGb),
    communicationRiskLevel: risks.find(r => r.category === 'communication')?.level || 'green',
    explanationTags,
    recommendations,
  };
}
