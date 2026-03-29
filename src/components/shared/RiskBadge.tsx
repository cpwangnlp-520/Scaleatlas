import { RUNTIME_COPY } from '../../content/runtimeCopy.ts';
import type { Locale, RiskLevel } from '../../types';

interface RiskBadgeProps {
  level: RiskLevel;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  locale?: Locale;
}

const colorMap: Record<RiskLevel, string> = {
  green: 'bg-emerald-500 text-white',
  yellow: 'bg-amber-400 text-slate-900',
  orange: 'bg-orange-500 text-white',
  red: 'bg-rose-500 text-white',
};

const sizeMap = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-3 py-1 text-sm',
  lg: 'px-4 py-2 text-base',
};

export function RiskBadge({ level, label, size = 'md', locale = 'zh' }: RiskBadgeProps) {
  const copy = RUNTIME_COPY[locale].risk;
  const labelMap: Record<RiskLevel, string> = {
    green: copy.runnable,
    yellow: copy.atRisk,
    orange: copy.highRisk,
    red: copy.blocked,
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium shadow-sm ${colorMap[level]} ${sizeMap[size]}`}
    >
      <span className="h-2 w-2 rounded-full bg-white/80" />
      {label || labelMap[level]}
    </span>
  );
}

interface StatusCardProps {
  canRun: boolean;
  level: RiskLevel;
  title?: string;
  description?: string;
  locale?: Locale;
}

export function ResultStatusCard({ canRun, level, title, description, locale = 'zh' }: StatusCardProps) {
  const copy = RUNTIME_COPY[locale].risk;
  const statusTitle = title || (canRun ? copy.runnable : copy.blocked);
  const statusDesc = description || getDefaultDescription(level, locale);

  return (
    <div className="workspace-panel workspace-panel-accent">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="workspace-panel-title">{statusTitle}</div>
          <p className="workspace-panel-description">{statusDesc}</p>
        </div>
        <RiskBadge level={level} size="lg" locale={locale} />
      </div>
    </div>
  );
}

function getDefaultDescription(level: RiskLevel, locale: Locale): string {
  const descriptions = RUNTIME_COPY[locale].risk.descriptions as Record<RiskLevel, string>;
  return descriptions[level];
}
