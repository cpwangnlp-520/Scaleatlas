import type {
  ModelConfig,
  TrainingConfig,
  ParallelConfig,
  HardwareConfig,
  RiskLevel,
  RiskAssessment,
  MoEConfig,
  Locale,
} from '../types';
import { calculateTotalMemory } from './memory.ts';
import { RUNTIME_COPY } from '../content/runtimeCopy.ts';

export function assessOOMRisk(
  peakMemoryGb: number,
  gpuMemoryGb: number,
): RiskLevel {
  const utilization = peakMemoryGb / gpuMemoryGb;
  
  if (utilization > 0.95) return 'red';
  if (utilization > 0.90) return 'orange';
  if (utilization > 0.80) return 'yellow';
  return 'green';
}

export function assessCommunicationRisk(
  parallel: ParallelConfig,
  hardware: HardwareConfig,
  config: ModelConfig,
): RiskLevel {
  const { tpSize, ppSize, epSize } = parallel;
  const { gpusPerNode, nodeCount, interconnectType } = hardware;
  
  const gpusPerParallel = tpSize * ppSize;
  const crossesNode = gpusPerParallel > gpusPerNode;
  
  if (crossesNode && interconnectType === 'PCIe') {
    return 'red';
  }
  
  if (crossesNode && (interconnectType === 'IB' || interconnectType === 'RoCE')) {
    if (tpSize > gpusPerNode) {
      return 'orange';
    }
  }
  
  if (epSize > 1 && epSize > gpusPerNode) {
    return 'red';
  }
  
  if (tpSize > 8) {
    return 'yellow';
  }
  
  return 'green';
}

export function assessActivationRisk(
  config: ModelConfig,
  training: TrainingConfig,
): RiskLevel {
  const { seqLen, microBatchSize } = training;
  const { numHiddenLayers } = config;
  
  if (seqLen > 32768 && microBatchSize > 4) {
    return 'orange';
  }
  
  if (seqLen > 65536) {
    return 'red';
  }
  
  if (numHiddenLayers > 80 && microBatchSize > 8) {
    return 'yellow';
  }
  
  return 'green';
}

export function assessKvCacheRisk(
  config: ModelConfig,
  seqLen: number,
  concurrency: number,
): RiskLevel {
  const { hiddenSize, numHiddenLayers, numKeyValueHeads, numAttentionHeads } = config;
  
  const kvCachePerToken = 2 * hiddenSize * (numKeyValueHeads / numAttentionHeads) * numHiddenLayers * 2;
  const totalKvCache = kvCachePerToken * seqLen * concurrency;
  const kvCacheGb = totalKvCache / (1024 ** 3);
  
  if (kvCacheGb > 60) return 'red';
  if (kvCacheGb > 40) return 'orange';
  if (kvCacheGb > 20) return 'yellow';
  return 'green';
}

export function assessMoERisk(
  moeConfig?: MoEConfig,
  parallel?: ParallelConfig,
  hardware?: HardwareConfig,
): RiskLevel {
  if (!moeConfig) return 'green';
  
  const { numLocalExperts, numExpertsPerTok } = moeConfig;
  const epSize = parallel?.epSize || 1;
  const gpusPerNode = hardware?.gpusPerNode || 8;
  
  if (numExpertsPerTok > 4 && epSize > gpusPerNode) {
    return 'red';
  }
  
  if (numLocalExperts > 8 && epSize === 1) {
    return 'yellow';
  }
  
  if (numLocalExperts > 64) {
    return 'yellow';
  }
  
  return 'green';
}

export function getAllRisks(
  config: ModelConfig,
  training: TrainingConfig,
  parallel: ParallelConfig,
  hardware: HardwareConfig,
  moeConfig?: MoEConfig,
  locale: Locale = 'zh',
): RiskAssessment[] {
  const copy = RUNTIME_COPY[locale].risk;
  const risks: RiskAssessment[] = [];
  
  const memory = calculateTotalMemory(config, training, parallel, hardware, moeConfig);
  const oomRisk = assessOOMRisk(memory.total, hardware.gpuMemoryGb);
  
  if (oomRisk !== 'green') {
    risks.push({
      category: 'memory',
      level: oomRisk,
      title: copy.oomTitle,
      description: copy.oomDescription(memory.total, hardware.gpuMemoryGb),
      suggestion: copy.oomSuggestion,
      affectedComponent: 'peak_memory',
    });
  }
  
  const commRisk = assessCommunicationRisk(parallel, hardware, config);
  if (commRisk !== 'green') {
    risks.push({
      category: 'communication',
      level: commRisk,
      title: copy.commTitle,
      description: copy.commDescription(parallel.tpSize, parallel.ppSize),
      suggestion: copy.commSuggestion,
      affectedComponent: 'parallel_strategy',
    });
  }
  
  const actRisk = assessActivationRisk(config, training);
  if (actRisk !== 'green') {
    risks.push({
      category: 'memory',
      level: actRisk,
      title: copy.actTitle,
      description: copy.actDescription(training.seqLen),
      suggestion: copy.actSuggestion,
      affectedComponent: 'activation_memory',
    });
  }
  
  const moeRisk = assessMoERisk(moeConfig, parallel, hardware);
  if (moeRisk !== 'green') {
    risks.push({
      category: 'communication',
      level: moeRisk,
      title: copy.moeTitle,
      description: copy.moeDescription(moeConfig?.numLocalExperts),
      suggestion: copy.moeSuggestion,
      affectedComponent: 'moe_routing',
    });
  }
  
  return risks;
}

export type { RiskAssessment };

export function getOverallRiskLevel(risks: RiskAssessment[]): RiskLevel {
  if (risks.some(r => r.level === 'red')) return 'red';
  if (risks.some(r => r.level === 'orange')) return 'orange';
  if (risks.some(r => r.level === 'yellow')) return 'yellow';
  return 'green';
}
