import { useCallback, useState } from 'react';
import { RUNTIME_COPY } from '../../content/runtimeCopy.ts';
import {
  getParallelCompatibilityReport,
  planInference,
  recommendInferenceParallel,
} from '../../engine';
import { ConfigImportPanel, MachineSplitMap, RecommendationCard, RiskBadge } from '../shared';
import { Checkbox, Input, Select } from '../shared/FormControls';
import { INFER_PAGE_COPY, STEP_FLOW_COPY } from '../../content/plannerCopy.ts';
import { usePlannerStore } from '../../stores';
import type { Locale, ModelFamily } from '../../types';

const MODEL_FAMILY_OPTIONS: Record<Locale, Array<{ value: string; label: string }>> = {
  zh: [
    { value: 'dense', label: 'Dense' },
    { value: 'moe', label: 'MoE' },
  ],
  en: [
    { value: 'dense', label: 'Dense' },
    { value: 'moe', label: 'MoE' },
  ],
};

const GPU_OPTIONS = [
  { value: 'A100-40G', label: 'A100 40GB' },
  { value: 'A100-80G', label: 'A100 80GB' },
  { value: 'H100-SXM', label: 'H100 SXM 80GB' },
  { value: 'H100-PCIe', label: 'H100 PCIe 80GB' },
  { value: 'L40S', label: 'L40S 48GB' },
  { value: 'H20', label: 'H20 96GB' },
  { value: '4090', label: 'RTX 4090 24GB' },
];

const ENGINE_OPTIONS = [
  { value: 'vLLM', label: 'vLLM' },
  { value: 'SGLang', label: 'SGLang' },
  { value: 'TensorRT-LLM', label: 'TensorRT-LLM' },
  { value: 'TGI', label: 'TGI' },
  { value: 'custom', label: 'Custom' },
];

const PRECISION_OPTIONS = [
  { value: 'bf16', label: 'BF16' },
  { value: 'fp16', label: 'FP16' },
  { value: 'fp8', label: 'FP8' },
  { value: 'int8', label: 'INT8' },
  { value: 'int4', label: 'INT4' },
];

export function InferencePlanner() {
  const {
    locale,
    model,
    moe,
    hardware,
    inference,
    inferenceResult,
    setModel,
    setMoe,
    setHardware,
    setInference,
    setInferenceResult,
    loadPreset,
  } = usePlannerStore();
  const copy = INFER_PAGE_COPY[locale];
  const runtimeCopy = RUNTIME_COPY[locale];
  const steps = STEP_FLOW_COPY[locale].infer.steps;
  const [presetValue, setPresetValue] = useState('');
  const presetOptions = [
    { value: '', label: runtimeCopy.common.presetPlaceholder },
    { value: 'Llama-3-8B', label: 'Llama 3 8B' },
    { value: 'Llama-3-70B', label: 'Llama 3 70B' },
    { value: 'Qwen2.5-72B', label: 'Qwen2.5 72B' },
    { value: 'Mixtral-8x7B', label: 'Mixtral 8x7B (MoE)' },
    { value: 'DeepSeek-V3', label: 'DeepSeek V3 (MoE)' },
  ];
  const activeModelFamily = (model.modelFamily === 'moe' ? 'moe' : 'dense') as 'dense' | 'moe';
  const totalGpuCount = hardware.gpusPerNode * hardware.nodeCount;

  const validationModel = model.hiddenSize && model.numHiddenLayers
    ? {
        modelName: model.modelName || 'Custom Model',
        modelFamily: activeModelFamily as ModelFamily,
        sourceType: 'manual' as const,
        paramCountTotal: 0,
        hiddenSize: model.hiddenSize || 4096,
        numHiddenLayers: model.numHiddenLayers || 32,
        numAttentionHeads: model.numAttentionHeads || 32,
        numKeyValueHeads: model.numKeyValueHeads || 8,
        intermediateSize: model.intermediateSize || (model.hiddenSize || 4096) * 4,
        vocabSize: model.vocabSize || 32000,
        maxPositionEmbeddings: model.maxPositionEmbeddings || 4096,
        tieWordEmbeddings: model.tieWordEmbeddings,
        usesLearnedPositionEmbeddings: model.usesLearnedPositionEmbeddings,
        normType: model.normType,
        attentionType: model.attentionType,
        positionEncodingType: model.positionEncodingType,
        ffnActivation: model.ffnActivation || 'swiglu',
      }
    : null;
  const validationMoeConfig = moe.numLocalExperts
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

  const handleCalculate = useCallback(() => {
    if (!validationModel) {
      alert(copy.actionCopy.fillModel);
      return;
    }

    const parallel = recommendInferenceParallel(validationModel, hardware, inference, validationMoeConfig);
    const result = planInference(validationModel, inference, hardware, parallel, validationMoeConfig, locale);
    setInferenceResult(result);
  }, [
    hardware,
    inference,
    locale,
    setInferenceResult,
    validationModel,
    validationMoeConfig,
  ]);

  const resultParallelReport = validationModel && inferenceResult
    ? getParallelCompatibilityReport(
        validationModel,
        inferenceResult.recommendedParallel,
        validationMoeConfig,
        hardware,
        locale,
      )
    : null;

  return (
    <section className="planner-page" data-page-tone="infer">
      <header className="planner-page-header">
        <div>
          <div className="planner-page-kicker">{STEP_FLOW_COPY[locale].infer.pageTitle}</div>
          <h1 className="planner-page-title">{copy.pageTitle}</h1>
          <p className="planner-page-description">{copy.pageDescription}</p>
        </div>
      </header>

      <div className="planner-step-flow">
        <section className="planner-step-card" data-step="model">
          <StepHeading index="01" title={steps.model} />

          <ConfigImportPanel />

          <div className="planner-form-grid planner-form-grid-wide">
            <Select
              label={copy.preset}
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
              value={activeModelFamily}
              onChange={(value) => setModel({ modelFamily: value as ModelFamily })}
              options={MODEL_FAMILY_OPTIONS[locale]}
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
              label={copy.ffnDim}
              value={model.intermediateSize || ''}
              onChange={(value) => setModel({ intermediateSize: Number(value) })}
              placeholder="16384"
            />
            <Input
              label={copy.vocabSize}
              value={model.vocabSize || ''}
              onChange={(value) => setModel({ vocabSize: Number(value) })}
              placeholder="32000"
            />
          </div>

          {activeModelFamily === 'moe' && (
            <div className="planner-inline-panel tone-train">
              <div className="planner-inline-title">{copy.moeTitle}</div>
              <div className="planner-form-grid">
                <Input
                  label={copy.moeFields.experts}
                  value={moe.numLocalExperts || ''}
                  onChange={(value) => setMoe({ numLocalExperts: Number(value) })}
                  placeholder="8"
                />
                <Input
                  label={copy.moeFields.expertsPerToken}
                  value={moe.numExpertsPerTok || ''}
                  onChange={(value) => setMoe({ numExpertsPerTok: Number(value) })}
                  placeholder="2"
                />
              </div>
            </div>
          )}
        </section>

        <section className="planner-step-card" data-step="service">
          <StepHeading index="02" title={steps.service} />

          <div className="planner-form-grid">
            <Select
              label={copy.engine}
              value={inference.engine}
              onChange={(value) => setInference({ engine: value as typeof inference.engine })}
              options={ENGINE_OPTIONS}
            />
            <Select
              label={copy.mode}
              value={inference.inferenceMode}
              onChange={(value) => setInference({ inferenceMode: value as typeof inference.inferenceMode })}
              options={[
                { value: 'offline', label: copy.modeOptions.offline },
                { value: 'online', label: copy.modeOptions.online },
                { value: 'streaming', label: copy.modeOptions.streaming },
              ]}
            />
            <Select
              label={copy.weightDtype}
              value={inference.weightDtype}
              onChange={(value) => setInference({ weightDtype: value as typeof inference.weightDtype })}
              options={PRECISION_OPTIONS}
            />
            <Select
              label={copy.kvDtype}
              value={inference.kvCacheDtype}
              onChange={(value) => setInference({ kvCacheDtype: value as typeof inference.kvCacheDtype })}
              options={PRECISION_OPTIONS}
            />
            <Input
              label={copy.inputAvg}
              value={inference.inputTokensAvg}
              onChange={(value) => setInference({ inputTokensAvg: Number(value) })}
              min={64}
              step={64}
            />
            <Input
              label={copy.inputP95}
              value={inference.inputTokensP95}
              onChange={(value) => setInference({ inputTokensP95: Number(value) })}
              min={64}
              step={64}
            />
            <Input
              label={copy.outputAvg}
              value={inference.outputTokensAvg}
              onChange={(value) => setInference({ outputTokensAvg: Number(value) })}
              min={32}
              step={32}
            />
            <Input
              label={copy.outputP95}
              value={inference.outputTokensP95}
              onChange={(value) => setInference({ outputTokensP95: Number(value) })}
              min={32}
              step={32}
            />
            <Input
              label={copy.concurrency}
              value={inference.targetConcurrency}
              onChange={(value) => setInference({ targetConcurrency: Number(value) })}
              min={1}
            />
          </div>

          <div className="planner-inline-panel">
            <div className="planner-inline-title">{copy.options}</div>
            <div className="planner-check-grid">
              <Checkbox
                checked={inference.continuousBatching}
                onChange={(value) => setInference({ continuousBatching: value })}
                label={copy.optionLabels.continuousBatching}
              />
              <Checkbox
                checked={inference.pagedKvCache}
                onChange={(value) => setInference({ pagedKvCache: value })}
                label={copy.optionLabels.pagedKvCache}
              />
              <Checkbox
                checked={inference.speculativeDecoding}
                onChange={(value) => setInference({ speculativeDecoding: value })}
                label={copy.optionLabels.speculativeDecoding}
              />
            </div>
          </div>
        </section>

        <section className="planner-step-card" data-step="hardware">
          <StepHeading index="03" title={steps.hardware} />

          <div className="planner-form-grid">
            <Select
              label={copy.gpuType}
              value={hardware.gpuType}
              onChange={(value) => setHardware({ gpuType: value as typeof hardware.gpuType })}
              options={GPU_OPTIONS}
            />
            <Input
              label={copy.gpuMemory}
              value={hardware.gpuMemoryGb}
              onChange={(value) => setHardware({ gpuMemoryGb: Number(value) })}
            />
            <Input
              label={copy.gpusPerNode}
              value={hardware.gpusPerNode}
              onChange={(value) => setHardware({ gpusPerNode: Number(value) })}
            />
            <Input
              label={copy.nodeCount}
              value={hardware.nodeCount}
              onChange={(value) => setHardware({ nodeCount: Number(value) })}
            />
            <div className="planner-mini-stat">
              <span>{copy.totalGpu}</span>
              <strong>{totalGpuCount}</strong>
            </div>
          </div>

          <button onClick={handleCalculate} className="surface-button" type="button">
            {copy.actionCopy.run}
          </button>
        </section>

        <section className="planner-step-card planner-step-card-result" data-step="result">
          <StepHeading index="04" title={steps.result} />

          {inferenceResult ? (
            <>
              <div className="result-summary-grid">
                <div className={`result-summary-card result-summary-card-primary status-${inferenceResult.deployabilityLevel}`}>
                  <div className="result-summary-label">{copy.resultSummary.deploy}</div>
                  <div className="result-summary-value">{inferenceResult.canDeploy ? copy.statusCopy.fit : copy.statusCopy.blocked}</div>
                  <div className="mt-3">
                    <RiskBadge
                      level={inferenceResult.deployabilityLevel}
                      label={inferenceResult.canDeploy ? copy.statusCopy.badgeFit : copy.statusCopy.badgeBlocked}
                      size="sm"
                      locale={locale}
                    />
                  </div>
                </div>
                <div className="result-summary-card tone-infer">
                  <div className="result-summary-label">{copy.resultSummary.concurrency}</div>
                  <div className="result-summary-value">{inferenceResult.safeConcurrency}</div>
                </div>
                <div className="result-summary-card tone-infer">
                  <div className="result-summary-label">{copy.resultSummary.peakMemory}</div>
                  <div className="result-summary-value">{inferenceResult.peakMemoryGb.toFixed(1)} GB</div>
                  <div className="result-summary-note">{hardware.gpuMemoryGb} GB / GPU</div>
                </div>
                <div className="result-summary-card tone-infer">
                  <div className="result-summary-label">{copy.resultSummary.parallel}</div>
                  <div className="result-summary-value">
                    {inferenceResult.recommendedParallel.tpSize} / {inferenceResult.recommendedParallel.ppSize} / {inferenceResult.recommendedParallel.dpSize} / {inferenceResult.recommendedParallel.epSize}
                  </div>
                  <div className="result-summary-note">TP / PP / DP / EP</div>
                </div>
              </div>

              <div className="planner-section-block">
                <div className="planner-section-title">{copy.memoryLayers.title}</div>
                <div className="memory-composition-grid memory-composition-grid-infer">
                  {[
                    { label: copy.memoryLayers.weight, value: inferenceResult.memWeightsGb, role: 'weight' },
                    { label: copy.memoryLayers.kv, value: inferenceResult.memKvCacheGb, role: 'kv' },
                    { label: copy.memoryLayers.runtime, value: inferenceResult.memRuntimeBufferGb, role: 'runtime' },
                  ].map((item) => (
                    <div key={item.label} className={`memory-layer-card role-${item.role}`}>
                      <span>{item.label}</span>
                      <strong>{item.value.toFixed(1)} GB</strong>
                    </div>
                  ))}
                </div>
              </div>

              {resultParallelReport && !resultParallelReport.isValid && (
                <div className="planner-inline-panel planner-inline-panel-warning">
                  <div className="planner-inline-title">{copy.resultSummary.parallel}</div>
                  <div className="planner-constraint-list">
                    {resultParallelReport.issues.map((issue) => <div key={issue}>{issue}</div>)}
                  </div>
                </div>
              )}

              <MachineSplitMap
                hardware={hardware}
                parallel={inferenceResult.recommendedParallel}
                layerCount={model.numHiddenLayers || 32}
                expertCount={moe.numLocalExperts}
                mode="infer"
                title={copy.mappingTitle}
                description={copy.mappingDescription}
                locale={locale}
              />

              <RecommendationCard
                recommendations={inferenceResult.recommendations}
                risks={[]}
                title={copy.recommendationTitle}
                description=""
                locale={locale}
              />
            </>
          ) : (
            <div className="planner-empty-copy">
              {copy.actionCopy.empty}
            </div>
          )}
        </section>
      </div>
    </section>
  );
}

function StepHeading({ index, title }: { index: string; title: string }) {
  return (
    <div className="planner-step-heading">
      <div className="planner-step-index">{index}</div>
      <div className="planner-step-title">{title}</div>
    </div>
  );
}
