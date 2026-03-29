import { useCallback, useMemo, useState } from 'react';
import { parseHFConfig } from '../../parsers';
import { usePlannerStore } from '../../stores';
import type { AttentionType, Locale, NormType, PositionEncodingType } from '../../types';

const CONFIG_COPY: Record<Locale, any> = {
  zh: {
    title: '导入 HuggingFace Config',
    hint: '默认收起原文，只保留摘要。',
    summary: 'Config 摘要',
    synced: '已同步到参数计算、训练规划和推理规划',
    recent: '最近导入，可切回复制 Config',
    unnamed: '未命名模型',
    layers: '层',
    architectureFields: {
      norm: '归一化',
      attention: '注意力',
      position: '位置编码',
    },
    emptyError: '请输入或上传 config.json 内容',
    parseFailed: '解析失败',
    viewRaw: '查看原文',
    repaste: '重新粘贴',
    clear: '清空',
    pastePlaceholder: '粘贴 config.json 内容，或拖拽文件到此处...',
    chooseFile: '选择文件',
    collapse: '收起原文',
    parse: '解析配置',
    family: {
      moe: 'MoE',
      multimodal: '多模态',
      tts: 'TTS',
      dense: 'Dense',
    },
  },
  en: {
    title: 'Import HuggingFace Config',
    hint: 'Keep the raw JSON collapsed by default.',
    summary: 'Config Summary',
    synced: 'Synced to Parameters, Training, and Inference',
    recent: 'Recently imported, switch back to Paste Config anytime',
    unnamed: 'Unnamed Model',
    layers: 'layers',
    architectureFields: {
      norm: 'Norm',
      attention: 'Attention',
      position: 'Position',
    },
    emptyError: 'Please paste or upload config.json',
    parseFailed: 'Parse failed',
    viewRaw: 'View Raw',
    repaste: 'Paste Again',
    clear: 'Clear',
    pastePlaceholder: 'Paste config.json here or drop a file...',
    chooseFile: 'Choose File',
    collapse: 'Collapse',
    parse: 'Parse Config',
    family: {
      moe: 'MoE',
      multimodal: 'Multimodal',
      tts: 'TTS',
      dense: 'Dense',
    },
  },
};

function getModelFamilyLabel(locale: Locale, modelFamily?: string) {
  const family = CONFIG_COPY[locale].family;
  switch (modelFamily) {
    case 'moe':
      return family.moe;
    case 'multimodal':
      return family.multimodal;
    case 'tts':
      return family.tts;
    default:
      return family.dense;
  }
}

function getNormLabel(normType?: NormType): string | null {
  if (normType === 'rmsnorm') {
    return 'RMSNorm';
  }
  if (normType === 'layernorm') {
    return 'LayerNorm';
  }
  return null;
}

function getAttentionLabel(attentionType?: AttentionType): string | null {
  if (!attentionType) {
    return null;
  }
  return attentionType.toUpperCase();
}

function getPositionEncodingLabel(
  positionEncodingType?: PositionEncodingType,
  usesLearnedPositionEmbeddings?: boolean,
): string {
  const resolvedType = positionEncodingType || (usesLearnedPositionEmbeddings ? 'learned' : 'rope');
  return resolvedType === 'learned' ? 'Learned' : 'RoPE';
}

export function ConfigImportPanel() {
  const {
    locale,
    importedConfigText,
    lastImportedConfig,
    modelInputMode,
    setImportedConfigText,
    applyImportedConfig,
    clearImportedConfig,
  } = usePlannerStore();
  const copy = CONFIG_COPY[locale];

  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showRawEditor, setShowRawEditor] = useState(false);

  const importedModel = lastImportedConfig?.modelConfig;
  const importSummary = useMemo(() => {
    if (!importedModel && !lastImportedConfig) {
      return null;
    }

    return [
      importedModel?.modelName || copy.unnamed,
      getModelFamilyLabel(locale, lastImportedConfig?.modelFamily),
      importedModel?.numHiddenLayers ? `${importedModel.numHiddenLayers} ${copy.layers}` : null,
      importedModel?.hiddenSize ? `Hidden ${importedModel.hiddenSize}` : null,
    ].filter(Boolean).join(' · ');
  }, [copy.layers, copy.unnamed, importedModel, lastImportedConfig, locale]);
  const architectureSummaryFields = useMemo(() => {
    if (!importedModel) {
      return [];
    }

    return [
      {
        label: copy.architectureFields.norm,
        value: getNormLabel(importedModel.normType),
      },
      {
        label: copy.architectureFields.attention,
        value: getAttentionLabel(importedModel.attentionType),
      },
      {
        label: copy.architectureFields.position,
        value: getPositionEncodingLabel(
          importedModel.positionEncodingType,
          importedModel.usesLearnedPositionEmbeddings,
        ),
      },
    ].filter((field): field is { label: string; value: string } => Boolean(field.value));
  }, [copy.architectureFields.attention, copy.architectureFields.norm, copy.architectureFields.position, importedModel]);

  const syncCopy = modelInputMode === 'config'
    ? copy.synced
    : copy.recent;

  const handleImport = useCallback(() => {
    if (!importedConfigText.trim()) {
      setError(copy.emptyError);
      return;
    }

    try {
      const parsed = parseHFConfig(importedConfigText, locale);
      setError(null);
      applyImportedConfig(importedConfigText, parsed);
      setShowRawEditor(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : copy.parseFailed);
    }
  }, [applyImportedConfig, copy.emptyError, copy.parseFailed, importedConfigText, locale]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);

    const file = e.dataTransfer.files[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setImportedConfigText(content);
      setShowRawEditor(true);
    };
    reader.readAsText(file);
  }, [setImportedConfigText]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setImportedConfigText(content);
      setShowRawEditor(true);
    };
    reader.readAsText(file);
  }, [setImportedConfigText]);

  const showCompactSummary = Boolean(lastImportedConfig) && !showRawEditor;

  return (
    <div className="space-y-3">
      <div>
        <div className="control-label">{copy.title}</div>
        <p className="control-hint mt-2">{copy.hint}</p>
      </div>

      {showCompactSummary ? (
        <div className="config-import-summary">
          <div className="config-import-summary-meta">
            <div className="config-import-summary-label">{copy.summary}</div>
            <div className="config-import-summary-title">{importSummary}</div>
            <p className="config-import-summary-note">{syncCopy}</p>
            {architectureSummaryFields.length > 0 && (
              <div className="planner-config-summary-row">
                {architectureSummaryFields.map((field) => (
                  <div key={field.label} className="planner-config-summary-chip">
                    <span>{field.label}</span>
                    <strong>{field.value}</strong>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="config-import-summary-actions">
            <button
              type="button"
              className="surface-link-button"
              onClick={() => setShowRawEditor(true)}
            >
              {copy.viewRaw}
            </button>
            <button
              type="button"
              className="surface-link-button"
              onClick={() => setShowRawEditor(true)}
            >
              {copy.repaste}
            </button>
            <button
              type="button"
              className="surface-link-button"
              onClick={() => {
                clearImportedConfig();
                setShowRawEditor(false);
                setError(null);
              }}
            >
              {copy.clear}
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`workspace-dropzone${dragActive ? ' is-active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <textarea
            value={importedConfigText}
            onChange={(e) => setImportedConfigText(e.target.value)}
            placeholder={copy.pastePlaceholder}
            className="control-area"
          />

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <label className="surface-link-button cursor-pointer">
                {copy.chooseFile}
                <input
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </label>

              {lastImportedConfig && (
                <button
                  type="button"
                  className="surface-link-button"
                  onClick={() => setShowRawEditor(false)}
                >
                  {copy.collapse}
                </button>
              )}
            </div>

            <button
              onClick={handleImport}
              disabled={!importedConfigText.trim()}
              className="surface-button surface-button-inline"
              type="button"
            >
              {copy.parse}
            </button>
          </div>
        </div>
      )}

      {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
  );
}
