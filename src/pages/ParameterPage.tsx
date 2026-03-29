import { useMemo, useState } from 'react';
import {
  getParallelCompatibilityReport,
  isPipelineParallelCompatible,
  isTensorParallelCompatible,
} from '../engine';
import { RUNTIME_COPY } from '../content/runtimeCopy.ts';
import {
  getModelBreakdown,
  type MemoryByDtype,
} from '../engine/model-breakdown.ts';
import { ConfigImportPanel } from '../components/shared';
import { Input, SegmentedControl, Select } from '../components/shared/FormControls';
import { ParameterSkeleton } from '../features/parameter/components/ParameterSkeleton.tsx';
import {
  buildSkeletonBlocks,
  formatParams,
} from '../features/parameter/modelSkeleton.ts';
import { PARAMETER_PAGE_COPY, STEP_FLOW_COPY } from '../content/plannerCopy.ts';
import { usePlannerStore } from '../stores';
import { MODEL_PRESETS } from '../types';
import type { ModelFamily } from '../types';

export function ParameterPage() {
  const {
    model,
    moe,
    multimodal,
    parallel,
    locale,
    modelInputMode,
    lastImportedConfig,
    setModel,
    setMoe,
    setMultimodal,
    setModelInputMode,
    loadPreset,
  } = usePlannerStore();
  const copy = PARAMETER_PAGE_COPY[locale];
  const runtimeCopy = RUNTIME_COPY[locale];
  const steps = STEP_FLOW_COPY[locale].parameter.steps;
  const [presetValue, setPresetValue] = useState('');
  const [configSectionOverride, setConfigSectionOverride] = useState<'open' | 'closed' | null>(null);
  const [expandedBlockId, setExpandedBlockId] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const modelFamilyOptions = [
    { value: 'dense', label: copy.structureTypes.dense },
    { value: 'moe', label: copy.structureTypes.moe },
    { value: 'multimodal', label: copy.structureTypes.multimodal },
    { value: 'tts', label: copy.structureTypes.tts },
  ];
  const presetOptions = [
    { value: '', label: copy.presetPlaceholder },
    ...Object.keys(MODEL_PRESETS).map((presetName) => ({ value: presetName, label: presetName })),
  ];

  const canAnalyze = Boolean(
    model.hiddenSize &&
    model.numHiddenLayers &&
    model.numAttentionHeads &&
    model.intermediateSize &&
    model.vocabSize,
  );
  const activeModel = canAnalyze
    ? {
        modelName: model.modelName || 'Custom Model',
        modelFamily: (model.modelFamily || 'dense') as ModelFamily,
        sourceType: 'manual' as const,
        paramCountTotal: 0,
        hiddenSize: model.hiddenSize || 4096,
        numHiddenLayers: model.numHiddenLayers || 32,
        numAttentionHeads: model.numAttentionHeads || 32,
        numKeyValueHeads: model.numKeyValueHeads || 8,
        intermediateSize: model.intermediateSize || (model.hiddenSize || 4096) * 4,
        vocabSize: model.vocabSize || 32000,
        maxPositionEmbeddings: model.maxPositionEmbeddings || 4096,
        ropeScaling: model.ropeScaling,
        tieWordEmbeddings: model.tieWordEmbeddings,
        usesLearnedPositionEmbeddings: model.usesLearnedPositionEmbeddings,
        normType: model.normType,
        attentionType: model.attentionType,
        positionEncodingType: model.positionEncodingType,
        ffnActivation: model.ffnActivation || 'swiglu',
      }
    : null;
  const activeMoe = moe.numLocalExperts
    ? {
        numLocalExperts: moe.numLocalExperts,
        numExpertsPerTok: moe.numExpertsPerTok || 2,
        numSharedExperts: moe.numSharedExperts,
        sharedExpertIntermediateSize: moe.sharedExpertIntermediateSize,
        firstKDenseReplace: moe.firstKDenseReplace,
        moeLayerFrequency: moe.moeLayerFrequency,
        expertIntermediateSize: moe.expertIntermediateSize,
      }
    : undefined;
  const activeMultimodal = multimodal.visionHiddenSize || multimodal.audioHiddenSize
    ? multimodal
    : undefined;
  const breakdown = activeModel
    ? getModelBreakdown(activeModel, activeMoe, activeMultimodal, locale)
    : null;
  const totalParamCount = breakdown ? breakdown.totalParamCount + breakdown.auxiliaryParamCount : 0;
  const skeletonBlocks = useMemo(
    () => (breakdown && activeModel
      ? buildSkeletonBlocks(breakdown, activeModel, activeMoe, activeMultimodal, locale)
      : []),
    [activeModel, activeMoe, activeMultimodal, breakdown, locale],
  );
  const mainBlocks = skeletonBlocks.filter((block) => block.flow === 'main');
  const sideBlocks = skeletonBlocks.filter((block) => block.flow === 'side');
  const hasReadyConfig = Boolean(lastImportedConfig || canAnalyze);
  const configSectionOpen = configSectionOverride === null
    ? !hasReadyConfig
    : configSectionOverride === 'open';

  const tpCandidates = activeModel
    ? getCandidates(activeModel.hiddenSize).filter((candidate) => isTensorParallelCompatible(activeModel, candidate))
    : [];
  const ppCandidates = activeModel
    ? getCandidates(activeModel.numHiddenLayers).filter((candidate) => isPipelineParallelCompatible(activeModel, candidate))
    : [];
  const epCandidates = activeMoe
    ? getCandidates(activeMoe.numLocalExperts).filter((candidate) => activeMoe.numLocalExperts % candidate === 0)
    : [1];
  const parallelReport = activeModel
    ? getParallelCompatibilityReport(activeModel, parallel, activeMoe, undefined, locale)
    : null;
  const structureType = copy.structureTypes[(activeModel?.modelFamily || model.modelFamily || 'dense') as ModelFamily];
  const summarySource = modelInputMode === 'config' && lastImportedConfig
    ? copy.summary.imported
    : copy.summary.manual;
  const summaryName = model.modelName || lastImportedConfig?.modelConfig.modelName || runtimeCopy.common.customModel;
  const summaryFields = [
    { label: copy.configSummaryFields.layers, value: model.numHiddenLayers || '--' },
    { label: copy.configSummaryFields.hidden, value: model.hiddenSize || '--' },
    { label: copy.configSummaryFields.experts, value: activeMoe?.numLocalExperts || '-' },
    { label: copy.configSummaryFields.multimodal, value: activeMultimodal ? runtimeCopy.common.yes : runtimeCopy.common.no },
  ];

  return (
    <section className="planner-page" data-page-tone="parameter">
      <header className="planner-page-header">
        <div>
          <div className="planner-page-kicker">{STEP_FLOW_COPY[locale].parameter.pageTitle}</div>
          <h1 className="planner-page-title">{copy.pageTitle}</h1>
          <p className="planner-page-description">{copy.pageDescription}</p>
        </div>
      </header>

      <div className="planner-step-flow">
        <section className="planner-step-card" data-step="model">
          <div className="planner-step-heading">
            <div className="planner-step-index">01</div>
            <div className="planner-step-title">{steps.model}</div>
          </div>

          <div className="planner-collapsible-header">
            <div>
              <div className="planner-section-title">{copy.modelSectionTitle}</div>
              <p className="planner-section-note">{copy.modelSectionSummary}</p>
            </div>
            <button
              type="button"
              className="planner-collapse-toggle"
              onClick={() => setConfigSectionOverride(configSectionOpen ? 'closed' : 'open')}
            >
              {configSectionOpen ? copy.modelSectionToggle.close : copy.modelSectionToggle.open}
            </button>
          </div>

          <div className="planner-config-summary-row">
            <div className="planner-config-summary-chip is-strong">
              <span>{summaryName}</span>
              <strong>{structureType}</strong>
            </div>
            <div className="planner-config-summary-chip">
              <span>{copy.summary.source}</span>
              <strong>{summarySource}</strong>
            </div>
            {summaryFields.map((field) => (
              <div key={field.label} className="planner-config-summary-chip">
                <span>{field.label}</span>
                <strong>{field.value}</strong>
              </div>
            ))}
          </div>

          {configSectionOpen && (
            <div className="planner-collapsible-body">
              <SegmentedControl
                label={copy.modeLabel}
                value={modelInputMode}
                onChange={(value) => setModelInputMode(value as 'custom' | 'config')}
                options={copy.modeOptions}
              />

              {modelInputMode === 'config' && (
                <div className="mt-4">
                  <ConfigImportPanel />
                </div>
              )}

              <div className="planner-form-grid planner-form-grid-wide">
                <Select
                  label={copy.presetLabel}
                  value={presetValue}
                  onChange={(value) => {
                    setPresetValue(value);
                    if (value) {
                      loadPreset(value);
                    }
                  }}
                  options={presetOptions}
                />
                <Select
                  label={copy.modelType}
                  value={model.modelFamily || 'dense'}
                  onChange={(value) => setModel({ modelFamily: value as ModelFamily })}
                  options={modelFamilyOptions}
                />
                <Input
                  label={copy.modelName}
                  value={model.modelName || ''}
                  onChange={(value) => setModel({ modelName: value })}
                  type="text"
                  placeholder={runtimeCopy.common.customModel}
                />
                <Input
                  label={copy.hiddenSize}
                  value={model.hiddenSize || ''}
                  onChange={(value) => setModel({ hiddenSize: Number(value) })}
                  placeholder="4096"
                />
                <Input
                  label={copy.layers}
                  value={model.numHiddenLayers || ''}
                  onChange={(value) => setModel({ numHiddenLayers: Number(value) })}
                  placeholder="32"
                />
                <Input
                  label={copy.attentionHeads}
                  value={model.numAttentionHeads || ''}
                  onChange={(value) => setModel({ numAttentionHeads: Number(value) })}
                  placeholder="32"
                />
                <Input
                  label={copy.kvHeads}
                  value={model.numKeyValueHeads || ''}
                  onChange={(value) => setModel({ numKeyValueHeads: Number(value) })}
                  placeholder="8"
                />
                <Input
                  label={copy.intermediateSize}
                  value={model.intermediateSize || ''}
                  onChange={(value) => setModel({ intermediateSize: Number(value) })}
                  placeholder="14336"
                />
                <Input
                  label={copy.vocabSize}
                  value={model.vocabSize || ''}
                  onChange={(value) => setModel({ vocabSize: Number(value) })}
                  placeholder="128256"
                />
              </div>

              {(model.modelFamily === 'moe' || activeMoe) && (
                <div className="planner-inline-panel tone-train">
                  <div className="planner-inline-title">{copy.moeTitle}</div>
                  <div className="planner-form-grid">
                    <Input
                      label={copy.experts}
                      value={moe.numLocalExperts || ''}
                      onChange={(value) => setMoe({ numLocalExperts: Number(value) })}
                      placeholder="8"
                    />
                    <Input
                      label={copy.expertsPerToken}
                      value={moe.numExpertsPerTok || ''}
                      onChange={(value) => setMoe({ numExpertsPerTok: Number(value) })}
                      placeholder="2"
                    />
                  </div>
                </div>
              )}

              {(model.modelFamily === 'multimodal' || activeMultimodal) && (
                <div className="planner-inline-panel tone-infer">
                  <div className="planner-inline-title">{copy.multimodalTitle}</div>
                  <div className="planner-form-grid">
                    <Input
                      label={copy.visionHiddenSize}
                      value={multimodal.visionHiddenSize || ''}
                      onChange={(value) => setMultimodal({ visionHiddenSize: Number(value) })}
                      placeholder="1152"
                    />
                    <Input
                      label={copy.visionLayers}
                      value={multimodal.visionNumLayers || ''}
                      onChange={(value) => setMultimodal({ visionNumLayers: Number(value) })}
                      placeholder="27"
                    />
                    <Input
                      label={copy.projectorHiddenSize}
                      value={multimodal.projectorHiddenSize || ''}
                      onChange={(value) => setMultimodal({ projectorHiddenSize: Number(value) })}
                      placeholder="3072"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        <section className="planner-step-card" data-step="result">
          <div className="planner-step-heading">
            <div className="planner-step-index">02</div>
            <div className="planner-step-title">{steps.result}</div>
          </div>

          {breakdown ? (
            <>
              <div className="planner-summary-bar">
                <div className="planner-summary-meta">
                  <div className="planner-summary-name">{summaryName}</div>
                  <div className="planner-summary-note">
                    {copy.summary.source}: {summarySource}
                  </div>
                </div>
                <div className="planner-summary-meta">
                  <div className="planner-summary-label">{copy.coreMetrics.structure}</div>
                  <div className="planner-summary-note">{structureType}</div>
                </div>
              </div>

              <div className="result-summary-grid result-summary-grid-compact">
                <div className="result-summary-card tone-parameter">
                  <div className="result-summary-label">{copy.coreMetrics.params}</div>
                  <div className="result-summary-value">{formatParams(totalParamCount)}</div>
                </div>
                <div className="result-summary-card tone-parameter">
                  <div className="result-summary-label">{copy.coreMetrics.bf16}</div>
                  <div className="result-summary-value">{breakdown.totalMemoryGb.bf16.toFixed(1)} GB</div>
                </div>
                <div className="result-summary-card tone-parameter">
                  <div className="result-summary-label">{copy.coreMetrics.structure}</div>
                  <div className="result-summary-value">{structureType}</div>
                </div>
              </div>

              <div className="planner-section-block">
                <div className="planner-section-header">
                  <div>
                    <div className="planner-section-title">{copy.architectureTitle}</div>
                    <p className="planner-section-note">{copy.overviewHint}</p>
                  </div>
                  <div className="planner-skeleton-hint">{copy.clickHint}</div>
                </div>

                <ParameterSkeleton
                  copy={copy}
                  mainBlocks={mainBlocks}
                  sideBlocks={sideBlocks}
                  expandedBlockId={expandedBlockId}
                  onToggleBlock={(blockId) => setExpandedBlockId((current) => current === blockId ? null : blockId)}
                />
              </div>

              <div className="planner-section-block">
                <div className="planner-section-title">{copy.precisionTitle}</div>
                <div className="planner-metric-row">
                  {['bf16', 'fp16', 'fp8', 'int8'].map((dtype) => (
                    <div key={dtype} className="planner-mini-metric">
                      <span>{dtype.toUpperCase()}</span>
                      <strong>{breakdown.totalMemoryGb[dtype as keyof MemoryByDtype].toFixed(1)} GB</strong>
                    </div>
                  ))}
                </div>
              </div>

              <div className="planner-section-block planner-section-block-advanced">
                <button
                  type="button"
                  className="planner-advanced-toggle"
                  onClick={() => setAdvancedOpen((current) => !current)}
                >
                  <span>{copy.advancedTitle}</span>
                  <span>{advancedOpen ? copy.advancedToggle.close : copy.advancedToggle.open}</span>
                </button>

                {advancedOpen && (
                  <div className="planner-advanced-content">
                    <div className="planner-inline-panel">
                      <div className="planner-inline-title">{copy.invalidSharding}</div>
                      <div className="planner-constraint-list">
                        <div>{copy.candidateLabels.tp}: {tpCandidates.join(', ') || '-'}</div>
                        <div>{copy.candidateLabels.pp}: {ppCandidates.join(', ') || '-'}</div>
                        <div>{copy.candidateLabels.ep}: {epCandidates.join(', ') || '-'}</div>
                      </div>
                    </div>

                    {parallelReport && (
                      <div className={`planner-inline-panel${parallelReport.isValid ? '' : ' planner-inline-panel-warning'}`}>
                        <div className="planner-inline-title">{copy.invalidSharding}</div>
                        <div className="planner-constraint-list">
                          {parallelReport.issues.length > 0
                            ? parallelReport.issues.map((issue) => <div key={issue}>{issue}</div>)
                            : parallelReport.notes.map((note) => <div key={note}>{note}</div>)}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="planner-empty-copy">{copy.waitingStructure}</div>
          )}
        </section>
      </div>
    </section>
  );
}

function getCandidates(limit: number): number[] {
  return Array.from({ length: Math.max(1, limit) }, (_, index) => index + 1);
}
