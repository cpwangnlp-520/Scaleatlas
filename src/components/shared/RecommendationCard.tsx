import { RUNTIME_COPY } from '../../content/runtimeCopy.ts';
import type { ParallelCompatibilityReport, RiskAssessment } from '../../engine';
import type { Locale } from '../../types';
import { RiskBadge } from './RiskBadge';

interface RecommendationCardProps {
  recommendations: string[];
  risks: RiskAssessment[];
  title?: string;
  description?: string;
  locale?: Locale;
}

export function RecommendationCard({
  recommendations,
  risks,
  title,
  description,
  locale = 'zh',
}: RecommendationCardProps) {
  const copy = RUNTIME_COPY[locale].shared;
  const resolvedTitle = title ?? copy.recommendationTitle;
  const resolvedDescription = description ?? copy.recommendationDescription;

  if (recommendations.length === 0 && risks.length === 0) {
    return null;
  }

  return (
    <div className="workspace-panel">
      <div className="workspace-panel-header">
        <div>
          <div className="workspace-panel-title">{resolvedTitle}</div>
          {resolvedDescription && <p className="workspace-panel-description">{resolvedDescription}</p>}
        </div>
      </div>

      {risks.length > 0 && (
        <div className="mb-4 space-y-2">
          {risks.map((risk, index) => (
            <div
              key={index}
              className="workspace-subpanel flex items-start gap-3"
            >
              <RiskBadge level={risk.level} size="sm" locale={locale} />
              <div className="flex-1">
                <p className="text-sm font-medium text-[var(--text-primary)]">{risk.title}</p>
                <p className="text-xs text-[var(--text-secondary)]">{risk.description}</p>
                <p className="mt-1 text-xs text-[var(--accent-blue)]">
                  {copy.suggestionPrefix}: {risk.suggestion}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {recommendations.length > 0 && (
        <ul className="space-y-1">
          {recommendations.map((rec, index) => (
            <li key={index} className="workspace-subpanel flex items-start gap-2 text-sm text-[var(--text-secondary)]">
              <svg className="mt-0.5 h-4 w-4 flex-shrink-0 text-[var(--accent-blue)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {rec}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface ParallelStrategyCardProps {
  tpSize: number;
  ppSize: number;
  dpSize: number;
  epSize?: number;
  zeroStage: string;
  gpuCount: number;
  validation?: ParallelCompatibilityReport;
  locale?: Locale;
}

export function ParallelStrategyCard({
  tpSize,
  ppSize,
  dpSize,
  epSize = 1,
  zeroStage,
  gpuCount,
  validation,
  locale = 'zh',
}: ParallelStrategyCardProps) {
  const commonCopy = RUNTIME_COPY[locale].common;
  const sharedCopy = RUNTIME_COPY[locale].shared;

  return (
    <div className="workspace-panel">
      <div className="workspace-panel-header">
        <div>
          <div className="workspace-panel-title">{sharedCopy.parallelTitle}</div>
          <p className="workspace-panel-description">
            {sharedCopy.parallelDescription}
          </p>
        </div>
      </div>

      <div className="metric-grid">
        {[
          { label: 'TP', value: tpSize },
          { label: 'PP', value: ppSize },
          { label: 'DP', value: dpSize },
          { label: 'EP', value: epSize },
        ].map((metric) => (
          <div key={metric.label} className="metric-card">
            <div className="metric-card-label">{metric.label}</div>
            <div className="metric-card-value">{metric.value}</div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-[var(--text-secondary)]">
        <span>{sharedCopy.zeroLabel}: {zeroStage === 'none' ? commonCopy.off : `Stage ${zeroStage}`}</span>
        {epSize > 1 && <span>EP: {epSize}</span>}
        <span>{commonCopy.totalGpu} {gpuCount} GPU</span>
      </div>

      {validation && (
        <div className={`workspace-subpanel mt-4${validation.isValid ? '' : ' workspace-subpanel-amber'}`}>
          <div className="workspace-panel-title">{commonCopy.validation}</div>
          {validation.issues.length > 0 && (
            <div className="mt-3 space-y-2 text-sm text-red-600">
              {validation.issues.map((issue) => (
                <p key={issue}>{issue}</p>
              ))}
            </div>
          )}
          {validation.warnings.length > 0 && (
            <div className="mt-3 space-y-2 text-sm text-amber-700">
              {validation.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          )}
          {validation.notes.length > 0 && (
            <div className="mt-3 space-y-2 text-sm text-[var(--text-secondary)]">
              {validation.notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ThroughputCardProps {
  tokensPerSec: number;
  tokensPerSecMin?: number;
  tokensPerSecMax?: number;
  estimatedTime?: string;
  label?: string;
  locale?: Locale;
}

export function ThroughputCard({
  tokensPerSec,
  tokensPerSecMin,
  tokensPerSecMax,
  estimatedTime,
  label,
  locale = 'zh',
}: ThroughputCardProps) {
  const copy = RUNTIME_COPY[locale].shared;
  const resolvedLabel = label ?? copy.throughputLabel;

  return (
    <div className="workspace-panel">
      <div className="workspace-panel-header">
        <div>
          <div className="workspace-panel-title">{resolvedLabel}</div>
          <p className="workspace-panel-description">
            {copy.throughputDescription}
          </p>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-card-label">Throughput</div>
        <div className="metric-card-value">
          {tokensPerSec.toLocaleString()} <span className="text-base font-medium text-[var(--text-muted)]">tok/s</span>
        </div>
      </div>

      {tokensPerSecMin && tokensPerSecMax && (
        <div className="mt-3 text-sm text-[var(--text-secondary)]">
          {copy.throughputRange(tokensPerSecMin, tokensPerSecMax)}
        </div>
      )}

      {estimatedTime && (
        <div className="mt-2 text-sm text-[var(--text-secondary)]">
          {copy.estimatedTime(estimatedTime)}
        </div>
      )}
    </div>
  );
}
