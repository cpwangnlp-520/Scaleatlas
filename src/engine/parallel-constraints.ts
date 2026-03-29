import type { HardwareConfig, Locale, ModelConfig, MoEConfig, ParallelConfig } from '../types';
import { RUNTIME_COPY } from '../content/runtimeCopy.ts';

export interface ParallelCompatibilityReport {
  isValid: boolean;
  issues: string[];
  warnings: string[];
  notes: string[];
}

export function hasExplicitParallelOverride(parallel: ParallelConfig): boolean {
  return (
    parallel.tpSize > 1 ||
    parallel.ppSize > 1 ||
    parallel.dpSize > 1 ||
    parallel.cpSize > 1 ||
    parallel.epSize > 1 ||
    parallel.zeroStage !== '1'
  );
}

export function getParallelCompatibilityReport(
  config: ModelConfig,
  parallel: ParallelConfig,
  moeConfig?: MoEConfig,
  hardware?: HardwareConfig,
  locale: Locale = 'zh',
): ParallelCompatibilityReport {
  const copy = RUNTIME_COPY[locale].parallelConstraints;
  const issues: string[] = [];
  const warnings: string[] = [];
  const notes: string[] = [];

  validateTensorParallel(config, parallel.tpSize, issues, notes, copy);
  validatePipelineParallel(config, parallel.ppSize, issues, notes, copy);
  validateExpertParallel(moeConfig, parallel.epSize, issues, notes, copy);
  validateDataParallel(parallel.dpSize, issues, copy);

  if (hardware) {
    validateHardwareFit(hardware, parallel, issues, warnings, notes, copy);
  }

  return {
    isValid: issues.length === 0,
    issues,
    warnings,
    notes,
  };
}

export function isTensorParallelCompatible(config: ModelConfig, tpSize: number): boolean {
  if (!Number.isInteger(tpSize) || tpSize < 1) {
    return false;
  }

  if (config.hiddenSize % tpSize !== 0) {
    return false;
  }

  if (config.intermediateSize % tpSize !== 0) {
    return false;
  }

  if (config.numAttentionHeads % tpSize !== 0) {
    return false;
  }

  if (config.numKeyValueHeads > 0 && config.numKeyValueHeads % tpSize !== 0) {
    return false;
  }

  return true;
}

export function isPipelineParallelCompatible(config: ModelConfig, ppSize: number): boolean {
  if (!Number.isInteger(ppSize) || ppSize < 1) {
    return false;
  }

  return config.numHiddenLayers % ppSize === 0;
}

function validateTensorParallel(
  config: ModelConfig,
  tpSize: number,
  issues: string[],
  notes: string[],
  copy: any,
) {
  if (!Number.isInteger(tpSize) || tpSize < 1) {
    issues.push(copy.invalidTp);
    return;
  }

  const invalidTargets: string[] = [];

  if (config.hiddenSize % tpSize !== 0) {
    invalidTargets.push(`hidden size ${config.hiddenSize}`);
  }
  if (config.intermediateSize % tpSize !== 0) {
    invalidTargets.push(`intermediate size ${config.intermediateSize}`);
  }
  if (config.numAttentionHeads % tpSize !== 0) {
    invalidTargets.push(`attention heads ${config.numAttentionHeads}`);
  }
  if (config.numKeyValueHeads > 0 && config.numKeyValueHeads % tpSize !== 0) {
    invalidTargets.push(`KV heads ${config.numKeyValueHeads}`);
  }

  if (invalidTargets.length > 0) {
    issues.push(copy.tpMustDivide(tpSize, invalidTargets));
    return;
  }

  notes.push(copy.tpValid(tpSize, config.hiddenSize, config.numAttentionHeads, config.numKeyValueHeads));
}

function validatePipelineParallel(
  config: ModelConfig,
  ppSize: number,
  issues: string[],
  notes: string[],
  copy: any,
) {
  if (!Number.isInteger(ppSize) || ppSize < 1) {
    issues.push(copy.invalidPp);
    return;
  }

  if (config.numHiddenLayers % ppSize !== 0) {
    issues.push(copy.ppMustDivide(ppSize, config.numHiddenLayers));
    return;
  }

  notes.push(copy.ppValid(ppSize, config.numHiddenLayers));
}

function validateExpertParallel(
  moeConfig: MoEConfig | undefined,
  epSize: number,
  issues: string[],
  notes: string[],
  copy: any,
) {
  if (!Number.isInteger(epSize) || epSize < 1) {
    issues.push(copy.invalidEp);
    return;
  }

  if (epSize === 1) {
    notes.push(copy.epDisabled);
    return;
  }

  if (!moeConfig) {
    issues.push(copy.epOnlyForMoe(epSize));
    return;
  }

  if (moeConfig.numLocalExperts % epSize !== 0) {
    issues.push(copy.epMustDivide(epSize, moeConfig.numLocalExperts));
    return;
  }

  notes.push(copy.epValid(epSize, moeConfig.numLocalExperts));
}

function validateDataParallel(
  dpSize: number,
  issues: string[],
  copy: any,
) {
  if (!Number.isInteger(dpSize) || dpSize < 1) {
    issues.push(copy.invalidDp);
  }
}

function validateHardwareFit(
  hardware: HardwareConfig,
  parallel: ParallelConfig,
  issues: string[],
  warnings: string[],
  notes: string[],
  copy: any,
) {
  const totalAvailableGpus = hardware.gpusPerNode * hardware.nodeCount;
  const requestedGpus = parallel.tpSize * parallel.ppSize * parallel.dpSize;

  if (requestedGpus > totalAvailableGpus) {
    issues.push(copy.hardwareExceeded(requestedGpus, totalAvailableGpus));
  } else {
    notes.push(copy.hardwareUsage(requestedGpus, totalAvailableGpus));
  }

  if (parallel.tpSize > hardware.gpusPerNode) {
    warnings.push(copy.tpCrossNode(parallel.tpSize));
  }

  if (parallel.epSize > hardware.gpusPerNode) {
    warnings.push(copy.epCrossNode(parallel.epSize));
  }
}
