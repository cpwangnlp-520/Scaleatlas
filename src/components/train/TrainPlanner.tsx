import { useCallback, useState } from 'react';
import { getAllRisks, getParallelCompatibilityReport, planTraining } from '../../engine';
import { RUNTIME_COPY } from '../../content/runtimeCopy.ts';
import { hasExplicitParallelOverride } from '../../engine/parallel-constraints.ts';
import { MachineSplitMap, RecommendationCard, RiskBadge, ConfigImportPanel } from '../shared';
import { Checkbox, Input, Select } from '../shared/FormControls';
import { STEP_FLOW_COPY, TRAIN_PAGE_COPY } from '../../content/plannerCopy.ts';
import { usePlannerStore } from '../../stores';
import type {
  HardwareConfig as HardwareConfigType,
  Locale,
  ModelFamily,
  ParallelConfig as ParallelConfigType,
  TrainingConfig as TrainingConfigType,
} from '../../types';

const MODEL_FAMILY_OPTIONS: Record<Locale, Array<{ value: string; label: string }>> = {
  zh: [
    { value: 'dense', label: 'Dense' },
    { value: 'moe', label: 'MoE' },
    { value: 'multimodal', label: '多模态' },
    { value: 'tts', label: 'TTS' },
  ],
  en: [
    { value: 'dense', label: 'Dense' },
    { value: 'moe', label: 'MoE' },
    { value: 'multimodal', label: 'Multimodal' },
    { value: 'tts', label: 'TTS' },
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

const INTERCONNECT_OPTIONS = [
  { value: 'NVLink', label: 'NVLink' },
  { value: 'IB', label: 'InfiniBand' },
  { value: 'RoCE', label: 'RoCE' },
  { value: 'PCIe', label: 'PCIe' },
];

const TRAINING_TYPE_OPTIONS: Record<Locale, Array<{ value: string; label: string }>> = {
  zh: [
    { value: 'pretrain', label: '预训练' },
    { value: 'continued_pretrain', label: '增量预训练' },
    { value: 'sft', label: 'SFT' },
    { value: 'lora', label: 'LoRA' },
    { value: 'qlora', label: 'QLoRA' },
  ],
  en: [
    { value: 'pretrain', label: 'Pretrain' },
    { value: 'continued_pretrain', label: 'Continued Pretrain' },
    { value: 'sft', label: 'SFT' },
    { value: 'lora', label: 'LoRA' },
    { value: 'qlora', label: 'QLoRA' },
  ],
};

export function TrainPlanner() {
  const {
    locale,
    model,
    moe,
    multimodal,
    tts,
    hardware,
    training,
    parallel,
    loraRank,
    trainingResult,
    setModel,
    setMoe,
    setMultimodal,
    setTTS,
    setHardware,
    setTraining,
    setParallel,
    setLoraRank,
    setTrainingResult,
    loadPreset,
  } = usePlannerStore();
  const copy = TRAIN_PAGE_COPY[locale];
  const runtimeCopy = RUNTIME_COPY[locale];
  const steps = STEP_FLOW_COPY[locale].train.steps;
  const presetOptions = [
    { value: '', label: runtimeCopy.common.presetPlaceholder },
    { value: 'Llama-3-8B', label: 'Llama 3 8B' },
    { value: 'Llama-3-70B', label: 'Llama 3 70B' },
    { value: 'Qwen2.5-72B', label: 'Qwen2.5 72B' },
    { value: 'Mixtral-8x7B', label: 'Mixtral 8x7B (MoE)' },
    { value: 'DeepSeek-V3', label: 'DeepSeek V3 (MoE)' },
  ];

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [presetValue, setPresetValue] = useState('');
  const activeModelFamily = (model.modelFamily || 'dense') as ModelFamily;
  const totalGpuCount = hardware.gpusPerNode * hardware.nodeCount;
  const validationModel = model.hiddenSize && model.numHiddenLayers
    ? {
        modelName: model.modelName || 'Custom Model',
        modelFamily: activeModelFamily,
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
  const manualParallelOverrideActive = hasExplicitParallelOverride(parallel);
  const manualParallelReport = validationModel
    ? getParallelCompatibilityReport(validationModel, parallel, validationMoeConfig, hardware, locale)
    : null;

  const handleCalculate = useCallback(() => {
    if (!model.hiddenSize || !model.numHiddenLayers) {
      alert(copy.actionCopy.fillModel);
      return;
    }

    if (manualParallelOverrideActive && manualParallelReport && !manualParallelReport.isValid) {
      alert(manualParallelReport.issues.join('\n'));
      return;
    }

    const fullModel = {
      modelName: model.modelName || 'Custom Model',
      modelFamily: activeModelFamily,
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
    };
    const moeConfig = validationMoeConfig;
    const multimodalConfig = multimodal.visionHiddenSize || multimodal.audioHiddenSize ? multimodal as any : undefined;
    const ttsConfig = tts.textEncoderHiddenSize || tts.acousticDecoderHiddenSize ? tts as any : undefined;

    const result = planTraining(
      fullModel,
      training,
      hardware,
      parallel,
      moeConfig,
      multimodalConfig,
      ttsConfig,
      loraRank,
      locale,
    );
    setTrainingResult(result);
  }, [
    activeModelFamily,
    hardware,
    locale,
    loraRank,
    manualParallelOverrideActive,
    manualParallelReport,
    model.ffnActivation,
    model.hiddenSize,
    model.intermediateSize,
    model.maxPositionEmbeddings,
    model.modelName,
    model.numAttentionHeads,
    model.numHiddenLayers,
    model.numKeyValueHeads,
    model.normType,
    model.attentionType,
    model.positionEncodingType,
    model.tieWordEmbeddings,
    model.usesLearnedPositionEmbeddings,
    model.vocabSize,
    multimodal,
    parallel,
    setTrainingResult,
    training,
    tts,
    validationMoeConfig,
  ]);

  const risks = trainingResult && validationModel
    ? getAllRisks(validationModel, training, trainingResult.recommendedParallel, hardware, validationMoeConfig, locale)
    : [];

  return (
    <section className="planner-page" data-page-tone="train">
      <header className="planner-page-header">
        <div>
          <div className="planner-page-kicker">{STEP_FLOW_COPY[locale].train.pageTitle}</div>
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

          {activeModelFamily === 'multimodal' && (
            <div className="planner-inline-panel tone-infer">
              <div className="planner-inline-title">{copy.multimodalTitle}</div>
              <div className="planner-form-grid">
                <Input
                  label={copy.multimodalFields.visionHiddenSize}
                  value={multimodal.visionHiddenSize || ''}
                  onChange={(value) => setMultimodal({ visionHiddenSize: Number(value) })}
                  placeholder="1024"
                />
                <Input
                  label={copy.multimodalFields.visionLayers}
                  value={multimodal.visionNumLayers || ''}
                  onChange={(value) => setMultimodal({ visionNumLayers: Number(value) })}
                  placeholder="24"
                />
              </div>
            </div>
          )}

          {activeModelFamily === 'tts' && (
            <div className="planner-inline-panel tone-parameter">
              <div className="planner-inline-title">{copy.ttsTitle}</div>
              <div className="planner-form-grid">
                <Input
                  label={copy.ttsFields.textEncoderHidden}
                  value={tts.textEncoderHiddenSize || ''}
                  onChange={(value) => setTTS({ textEncoderHiddenSize: Number(value) })}
                  placeholder="256"
                />
                <Input
                  label={copy.ttsFields.acousticDecoderHidden}
                  value={tts.acousticDecoderHiddenSize || ''}
                  onChange={(value) => setTTS({ acousticDecoderHiddenSize: Number(value) })}
                  placeholder="1024"
                />
              </div>
            </div>
          )}
        </section>

        <section className="planner-step-card" data-step="task">
          <StepHeading index="02" title={steps.task} />

          <div className="planner-form-grid">
            <Select
              label={copy.trainingType}
              value={training.trainingType}
              onChange={(value) => setTraining({ trainingType: value as TrainingConfigType['trainingType'] })}
              options={TRAINING_TYPE_OPTIONS[locale]}
            />
            <Select
              label={copy.precision}
              value={training.computeDtype}
              onChange={(value) => setTraining({ computeDtype: value as TrainingConfigType['computeDtype'] })}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
            <Input
              label={copy.seqLen}
              value={training.seqLen}
              onChange={(value) => setTraining({ seqLen: Number(value) })}
              min={128}
              step={128}
            />
            <Input
              label={copy.globalBatch}
              value={training.globalBatchSize}
              onChange={(value) => setTraining({ globalBatchSize: Number(value) })}
              min={1}
            />
            <Input
              label={copy.microBatch}
              value={training.microBatchSize}
              onChange={(value) => setTraining({ microBatchSize: Number(value) })}
              min={1}
            />
          </div>

          {(training.trainingType === 'lora' || training.trainingType === 'qlora') && (
            <div className="planner-inline-panel">
              <div className="planner-inline-title">{copy.loraTitle}</div>
              <div className="planner-form-grid">
                <Input
                  label={copy.loraRank}
                  value={loraRank}
                  onChange={(value) => setLoraRank(Number(value))}
                  placeholder="16"
                />
              </div>
            </div>
          )}

          <div className="planner-inline-panel">
            <div className="planner-inline-title">{copy.optimizer}</div>
            <div className="planner-check-grid">
              <Checkbox
                checked={training.flashAttention}
                onChange={(value) => setTraining({ flashAttention: value })}
                label={copy.optimizerOptions.flashAttention}
              />
              <Checkbox
                checked={training.optimizerOffload}
                onChange={(value) => setTraining({ optimizerOffload: value })}
                label={copy.optimizerOptions.optimizerOffload}
              />
            </div>
            <div className="mt-4">
              <Select
                label={copy.recomputation}
                value={training.recomputation}
                onChange={(value) => setTraining({
                  recomputation: value as TrainingConfigType['recomputation'],
                  activationCheckpointing: value !== 'none',
                })}
                options={[
                  { value: 'none', label: locale === 'zh' ? '关闭' : 'Off' },
                  { value: 'selective', label: 'Selective' },
                  { value: 'full', label: 'Full' },
                ]}
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
              onChange={(value) => setHardware({ gpuType: value as HardwareConfigType['gpuType'] })}
              options={GPU_OPTIONS}
            />
            <Input
              label={copy.gpuMemory}
              value={hardware.gpuMemoryGb}
              onChange={(value) => setHardware({ gpuMemoryGb: Number(value) })}
            />
            <Select
              label={copy.interconnect}
              value={hardware.interconnectType}
              onChange={(value) => setHardware({ interconnectType: value as HardwareConfigType['interconnectType'] })}
              options={INTERCONNECT_OPTIONS}
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

          <button
            type="button"
            className="planner-advanced-toggle"
            onClick={() => setShowAdvanced((current) => !current)}
          >
            <span>{copy.parallelValidation}</span>
            <span>{showAdvanced ? copy.advancedToggle.close : copy.advancedToggle.open}</span>
          </button>

          {showAdvanced && (
            <div className="planner-advanced-content">
              <div className="planner-form-grid">
                <Input label={copy.parallelFields.tp} value={parallel.tpSize} onChange={(value) => setParallel({ tpSize: Number(value) })} />
                <Input label={copy.parallelFields.pp} value={parallel.ppSize} onChange={(value) => setParallel({ ppSize: Number(value) })} />
                <Input label={copy.parallelFields.dp} value={parallel.dpSize} onChange={(value) => setParallel({ dpSize: Number(value) })} />
                <Input label={copy.parallelFields.ep} value={parallel.epSize} onChange={(value) => setParallel({ epSize: Number(value) })} />
              </div>

              <div className="mt-4">
                <Select
                  label={copy.parallelFields.zeroStage}
                  value={parallel.zeroStage}
                  onChange={(value) => setParallel({ zeroStage: value as ParallelConfigType['zeroStage'] })}
                  options={[
                    { value: 'none', label: locale === 'zh' ? '关闭' : 'Off' },
                    { value: '1', label: 'Stage 1' },
                    { value: '2', label: 'Stage 2' },
                    { value: '3', label: 'Stage 3' },
                  ]}
                />
              </div>

              {manualParallelReport && (
                <div className={`planner-inline-panel${manualParallelReport.isValid ? '' : ' planner-inline-panel-warning'}`}>
                  <div className="planner-inline-title">{copy.parallelValidation}</div>
                  <div className="planner-constraint-list">
                    {manualParallelReport.issues.length > 0
                      ? manualParallelReport.issues.map((issue) => <div key={issue}>{issue}</div>)
                      : manualParallelReport.notes.map((note) => <div key={note}>{note}</div>)}
                  </div>
                </div>
              )}
            </div>
          )}

          <button onClick={handleCalculate} className="surface-button" type="button">
            {copy.actionCopy.run}
          </button>
        </section>

        <section className="planner-step-card planner-step-card-result" data-step="result">
          <StepHeading index="04" title={steps.result} />

          {trainingResult ? (
            <>
              <div className="result-summary-grid">
                <div className={`result-summary-card result-summary-card-primary status-${trainingResult.runnabilityLevel}`}>
                  <div className="result-summary-label">{copy.resultSummary.canRun}</div>
                  <div className="result-summary-value">{trainingResult.canRun ? copy.statusCopy.fit : copy.statusCopy.blocked}</div>
                  <div className="mt-3">
                    <RiskBadge
                      level={trainingResult.runnabilityLevel}
                      label={trainingResult.canRun ? copy.statusCopy.badgeFit : copy.statusCopy.badgeBlocked}
                      size="sm"
                      locale={locale}
                    />
                  </div>
                </div>
                <div className="result-summary-card tone-train">
                  <div className="result-summary-label">{copy.resultSummary.peakMemory}</div>
                  <div className="result-summary-value">{trainingResult.peakMemoryGb.toFixed(1)} GB</div>
                  <div className="result-summary-note">{hardware.gpuMemoryGb} GB / GPU</div>
                </div>
                <div className="result-summary-card tone-train">
                  <div className="result-summary-label">{copy.resultSummary.parallel}</div>
                  <div className="result-summary-value">
                    {trainingResult.recommendedParallel.tpSize} / {trainingResult.recommendedParallel.ppSize} / {trainingResult.recommendedParallel.dpSize} / {trainingResult.recommendedParallel.epSize}
                  </div>
                  <div className="result-summary-note">TP / PP / DP / EP</div>
                </div>
                <div className="result-summary-card tone-train">
                  <div className="result-summary-label">{copy.resultSummary.throughput}</div>
                  <div className="result-summary-value">{trainingResult.tokensPerSec.toLocaleString()}</div>
                  <div className="result-summary-note">tok/s</div>
                </div>
              </div>

              <div className="planner-section-block">
                <div className="planner-section-title">{copy.memoryBreakdown}</div>
                <div className="memory-composition-grid">
                  {[
                    { label: copy.memoryLabels.params, value: trainingResult.memParamsGb, role: 'weight' },
                    { label: copy.memoryLabels.grads, value: trainingResult.memGradsGb, role: 'grad' },
                    { label: copy.memoryLabels.optimizer, value: trainingResult.memOptimizerGb, role: 'optimizer' },
                    { label: copy.memoryLabels.activation, value: trainingResult.memActivationGb, role: 'activation' },
                    { label: runtimeCopy.common.buffer, value: trainingResult.memBufferGb, role: 'runtime' },
                  ].map((item) => (
                    <div key={item.label} className={`memory-layer-card role-${item.role}`}>
                      <span>{item.label}</span>
                      <strong>{item.value.toFixed(1)} GB</strong>
                    </div>
                  ))}
                </div>
              </div>

              <MachineSplitMap
                hardware={hardware}
                parallel={trainingResult.recommendedParallel}
                layerCount={model.numHiddenLayers || 32}
                expertCount={moe.numLocalExperts}
                mode="train"
                title={copy.mappingTitle}
                description={copy.mappingDescription}
                locale={locale}
              />

              <RecommendationCard
                recommendations={trainingResult.recommendations}
                risks={risks}
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
