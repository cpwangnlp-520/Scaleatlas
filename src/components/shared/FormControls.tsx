import type { CSSProperties } from 'react';

interface SelectProps {
  value: string | number;
  onChange: (value: string) => void;
  options: { value: string | number; label: string }[];
  label?: string;
  className?: string;
}

export function Select({ value, onChange, options, label, className = '' }: SelectProps) {
  return (
    <div className={`control-stack ${className}`}>
      {label && (
        <label className="control-label">{label}</label>
      )}
      <div className="control-select-wrap">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="control-select"
        >
          {options.map((opt) => (
            <option key={String(opt.value)} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <svg className="control-select-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="m7 10 5 5 5-5" />
        </svg>
      </div>
    </div>
  );
}

interface InputProps {
  value: string | number;
  onChange: (value: string) => void;
  label?: string;
  type?: 'text' | 'number';
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  className?: string;
}

export function Input({
  value,
  onChange,
  label,
  type = 'number',
  min,
  max,
  step,
  placeholder,
  className = '',
}: InputProps) {
  return (
    <div className={`control-stack ${className}`}>
      {label && (
        <label className="control-label">{label}</label>
      )}
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        className="control-field"
      />
    </div>
  );
}

interface CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  className?: string;
}

export function Checkbox({ checked, onChange, label, className = '' }: CheckboxProps) {
  return (
    <label className={`control-check ${className}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  label?: string;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  className?: string;
}

export function Slider({
  value,
  onChange,
  label,
  min,
  max,
  step = 1,
  unit = '',
  className = '',
}: SliderProps) {
  return (
    <div className={`control-slider-shell ${className}`}>
      {label && (
        <div className="control-slider-meta">
          <label className="control-label">{label}</label>
          <span className="control-slider-value">
            {value.toLocaleString()}{unit}
          </span>
        </div>
      )}
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full"
        style={{ '--value': `${((value - min) / (max - min)) * 100}%` } as CSSProperties}
      />
      <div className="flex justify-between text-xs text-[var(--text-muted)]">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

interface SegmentedControlProps {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; note?: string }[];
  label?: string;
  className?: string;
}

export function SegmentedControl({
  value,
  onChange,
  options,
  label,
  className = '',
}: SegmentedControlProps) {
  return (
    <div className={`control-stack ${className}`}>
      {label && (
        <label className="control-label">{label}</label>
      )}

      <div className="control-segmented">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`control-segmented-option${value === option.value ? ' is-active' : ''}`}
            onClick={() => onChange(option.value)}
          >
            <span>{option.label}</span>
            {option.note && (
              <small>{option.note}</small>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
