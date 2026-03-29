import { RUNTIME_COPY } from '../../content/runtimeCopy.ts';
import type { Locale } from '../../types';

interface MemorySegment {
  label: string;
  value: number;
  color: string;
}

interface MemoryBarProps {
  segments: MemorySegment[];
  total: number;
  maxMemory: number;
  unit?: string;
  locale?: Locale;
}

export function MemoryBar({ segments, total, maxMemory, unit = 'GB', locale = 'zh' }: MemoryBarProps) {
  const copy = RUNTIME_COPY[locale].memoryBar;
  const safeMaxMemory = Math.max(maxMemory, 1);
  const utilization = (total / safeMaxMemory) * 100;
  const isOverCapacity = total > maxMemory;

  return (
    <div className="workspace-panel">
      <div className="workspace-panel-header">
        <div>
          <div className="workspace-panel-title">{copy.title}</div>
          <p className="workspace-panel-description">{copy.description}</p>
        </div>
        <span className={isOverCapacity ? 'text-sm font-semibold text-red-500' : 'text-sm text-[var(--text-secondary)]'}>
          {total.toFixed(1)} {unit} / {maxMemory} {unit}
        </span>
      </div>

      <div className="workspace-memory-track">
        {segments.map((segment, index) => {
          const width = (segment.value / safeMaxMemory) * 100;
          const leftOffset = segments
            .slice(0, index)
            .reduce((acc, s) => acc + (s.value / safeMaxMemory) * 100, 0);

          return (
            <div
              key={segment.label}
              className={`workspace-memory-segment ${segment.color}`}
              style={{
                left: `${leftOffset}%`,
                width: `${Math.min(width, 100 - leftOffset)}%`,
              }}
              title={`${segment.label}: ${segment.value.toFixed(1)} ${unit}`}
            />
          );
        })}

        {isOverCapacity && (
          <div
            className="workspace-memory-overflow"
            style={{ left: '100%', width: `${utilization - 100}%` }}
          />
        )}

        <div className="workspace-memory-cap" style={{ left: '100%' }} />
      </div>

      <div className="flex flex-wrap gap-3 text-xs">
        {segments.map((segment) => (
          <div key={segment.label} className="flex items-center gap-1.5">
            <span className={`h-3 w-3 rounded ${segment.color}`} />
            <span className="text-[var(--text-secondary)]">
              {segment.label}: {segment.value.toFixed(1)} {unit}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

interface MemoryBreakdownProps {
  params: number;
  grads: number;
  optimizer: number;
  activation: number;
  buffer: number;
  total: number;
  gpuMemory: number;
  locale?: Locale;
}

export function TrainingMemoryBar({
  params,
  grads,
  optimizer,
  activation,
  buffer,
  total,
  gpuMemory,
  locale = 'zh',
}: MemoryBreakdownProps) {
  const copy = RUNTIME_COPY[locale].memoryBar;
  const segments: MemorySegment[] = [
    { label: copy.params, value: params, color: 'bg-blue-500' },
    { label: copy.grads, value: grads, color: 'bg-indigo-500' },
    { label: copy.optimizer, value: optimizer, color: 'bg-purple-500' },
    { label: copy.activation, value: activation, color: 'bg-teal-500' },
    { label: copy.buffer, value: buffer, color: 'bg-gray-400' },
  ];

  return (
    <MemoryBar
      segments={segments}
      total={total}
      maxMemory={gpuMemory}
      locale={locale}
    />
  );
}

interface InferenceMemoryBreakdownProps {
  weights: number;
  kvCache: number;
  buffer: number;
  total: number;
  gpuMemory: number;
  locale?: Locale;
}

export function InferenceMemoryBar({
  weights,
  kvCache,
  buffer,
  total,
  gpuMemory,
  locale = 'zh',
}: InferenceMemoryBreakdownProps) {
  const copy = RUNTIME_COPY[locale].memoryBar;
  const segments: MemorySegment[] = [
    { label: copy.weights, value: weights, color: 'bg-blue-500' },
    { label: copy.kvCache, value: kvCache, color: 'bg-amber-500' },
    { label: copy.buffer, value: buffer, color: 'bg-gray-400' },
  ];

  return (
    <MemoryBar
      segments={segments}
      total={total}
      maxMemory={gpuMemory}
      locale={locale}
    />
  );
}
