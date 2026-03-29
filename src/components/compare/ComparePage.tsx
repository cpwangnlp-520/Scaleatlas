import { usePlannerStore } from '../../stores';
import type { HardwareConfig, Locale, TrainingResult } from '../../types';
import { RiskBadge } from '../shared/RiskBadge';

interface CompareScenario {
  name: string;
  hardware: HardwareConfig;
  result: TrainingResult;
}

interface ComparePageProps {
  scenarios?: CompareScenario[];
}

const COMPARE_PAGE_COPY: Record<Locale, any> = {
  zh: {
    kicker: '方案对比',
    title: '多方案并排评估',
    description: '把不同 GPU 规格、节点规模和并行策略放到同一个视图里，直接看显存压力、吞吐和风险差异。',
    summaryTitle: '对比摘要',
    summaryDescription: '适合做容量评审时的第一轮筛选，先剔除明显不可行或性价比偏低的组合。',
    peakMemory: '显存峰值',
    singleGpuLimit: (gpuMemoryGb: number) => `单卡上限 ${gpuMemoryGb} GB`,
    parallel: '并行策略',
    throughput: '吞吐',
    estimatedTime: '预计时长',
    gpu: 'GPU',
    memoryState: '显存状态',
    overLimit: '超出上限',
    withinLimit: '在范围内',
    scenarioNames: ['方案 A', '方案 B', '方案 C'],
    scenarioTimes: ['2天 4小时', '3天 2小时', '5天 8小时'],
    scenarioExplanationTags: [
      [],
      ['显存接近上限'],
      ['显存溢出', 'TP跨节点'],
    ],
    scenarioRecommendations: [
      [],
      ['建议开启 activation checkpointing'],
      ['显存不足，需要更多 GPU', 'TP 跨节点通信开销大'],
    ],
  },
  en: {
    kicker: 'Comparison',
    title: 'Side-by-Side Scenario Review',
    description: 'Put GPU types, node counts, and parallel strategies into the same view so memory pressure, throughput, and risk differences are immediately visible.',
    summaryTitle: 'Comparison Summary',
    summaryDescription: 'Useful for a first review pass during capacity planning so obviously infeasible or weak-value combinations can be removed early.',
    peakMemory: 'Peak Memory',
    singleGpuLimit: (gpuMemoryGb: number) => `Single-GPU limit ${gpuMemoryGb} GB`,
    parallel: 'Parallel Strategy',
    throughput: 'Throughput',
    estimatedTime: 'Estimated Time',
    gpu: 'GPU',
    memoryState: 'Memory Status',
    overLimit: 'Over Limit',
    withinLimit: 'Within Limit',
    scenarioNames: ['Scenario A', 'Scenario B', 'Scenario C'],
    scenarioTimes: ['2d 4h', '3d 2h', '5d 8h'],
    scenarioExplanationTags: [
      [],
      ['Memory near the limit'],
      ['Out of memory', 'TP crosses nodes'],
    ],
    scenarioRecommendations: [
      [],
      ['Enable activation checkpointing'],
      ['Not enough memory, more GPUs are required', 'TP crossing nodes adds communication cost'],
    ],
  },
};

export function ComparePage({ scenarios: propScenarios }: ComparePageProps) {
  const locale = usePlannerStore((state) => state.locale);
  const copy = COMPARE_PAGE_COPY[locale];

  const defaultScenarios: CompareScenario[] = propScenarios || [
    {
      name: copy.scenarioNames[0],
      hardware: { gpuType: 'H100-SXM', gpuMemoryGb: 80, gpusPerNode: 8, nodeCount: 1, interconnectType: 'NVLink' },
      result: {
        canRun: true,
        runnabilityLevel: 'green',
        recommendedGpuCount: 8,
        recommendedParallel: { tpSize: 8, ppSize: 1, dpSize: 1, zeroStage: '1', cpSize: 1, epSize: 1 },
        peakMemoryGb: 45.2,
        memParamsGb: 28.5,
        memGradsGb: 14.2,
        memOptimizerGb: 0,
        memActivationGb: 2.0,
        memBufferGb: 0.5,
        tokensPerSec: 12000,
        tokensPerSecMin: 9600,
        tokensPerSecMax: 14400,
        estimatedTime: copy.scenarioTimes[0],
        oomRiskLevel: 'green',
        communicationRiskLevel: 'green',
        explanationTags: copy.scenarioExplanationTags[0],
        recommendations: copy.scenarioRecommendations[0],
      },
    },
    {
      name: copy.scenarioNames[1],
      hardware: { gpuType: 'A100-80G', gpuMemoryGb: 80, gpusPerNode: 8, nodeCount: 1, interconnectType: 'NVLink' },
      result: {
        canRun: true,
        runnabilityLevel: 'yellow',
        recommendedGpuCount: 8,
        recommendedParallel: { tpSize: 8, ppSize: 1, dpSize: 1, zeroStage: '2', cpSize: 1, epSize: 1 },
        peakMemoryGb: 68.5,
        memParamsGb: 28.5,
        memGradsGb: 14.2,
        memOptimizerGb: 21.3,
        memActivationGb: 4.0,
        memBufferGb: 0.5,
        tokensPerSec: 8500,
        tokensPerSecMin: 6800,
        tokensPerSecMax: 10200,
        estimatedTime: copy.scenarioTimes[1],
        oomRiskLevel: 'yellow',
        communicationRiskLevel: 'green',
        explanationTags: copy.scenarioExplanationTags[1],
        recommendations: copy.scenarioRecommendations[1],
      },
    },
    {
      name: copy.scenarioNames[2],
      hardware: { gpuType: 'L40S', gpuMemoryGb: 48, gpusPerNode: 8, nodeCount: 2, interconnectType: 'IB' },
      result: {
        canRun: false,
        runnabilityLevel: 'red',
        recommendedGpuCount: 16,
        recommendedParallel: { tpSize: 8, ppSize: 2, dpSize: 1, zeroStage: '3', cpSize: 1, epSize: 1 },
        peakMemoryGb: 52.8,
        memParamsGb: 18.0,
        memGradsGb: 9.0,
        memOptimizerGb: 13.5,
        memActivationGb: 10.8,
        memBufferGb: 1.5,
        tokensPerSec: 5000,
        tokensPerSecMin: 4000,
        tokensPerSecMax: 6000,
        estimatedTime: copy.scenarioTimes[2],
        oomRiskLevel: 'red',
        communicationRiskLevel: 'orange',
        explanationTags: copy.scenarioExplanationTags[2],
        recommendations: copy.scenarioRecommendations[2],
      },
    },
  ];

  return (
    <section className="space-y-8">
      <div className="workspace-page-header">
        <div>
          <div className="workspace-page-kicker">{copy.kicker}</div>
          <h1 className="workspace-page-title">{copy.title}</h1>
          <p className="workspace-page-description">{copy.description}</p>
        </div>
      </div>

      <div className="workspace-panel workspace-panel-accent">
        <div className="workspace-panel-title">{copy.summaryTitle}</div>
        <p className="workspace-panel-description">{copy.summaryDescription}</p>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {defaultScenarios.map((scenario) => (
          <div
            key={scenario.name}
            className="workspace-panel"
          >
            <div className="workspace-panel-header">
              <div>
                <div className="workspace-panel-title">{scenario.name}</div>
                <p className="workspace-panel-description">
                  {scenario.hardware.gpusPerNode * scenario.hardware.nodeCount}x {scenario.hardware.gpuType}
                </p>
              </div>
              <RiskBadge level={scenario.result.runnabilityLevel} locale={locale} />
            </div>

            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-card-label">{copy.peakMemory}</div>
                <div className="metric-card-value">{scenario.result.peakMemoryGb} GB</div>
                <div className="metric-card-note">{copy.singleGpuLimit(scenario.hardware.gpuMemoryGb)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-card-label">{copy.parallel}</div>
                <div className="metric-card-value">
                  TP{scenario.result.recommendedParallel.tpSize}
                </div>
                <div className="metric-card-note">PP{scenario.result.recommendedParallel.ppSize}</div>
              </div>
              <div className="metric-card">
                <div className="metric-card-label">{copy.throughput}</div>
                <div className="metric-card-value">{scenario.result.tokensPerSec.toLocaleString()}</div>
                <div className="metric-card-note">tok/s</div>
              </div>
              <div className="metric-card">
                <div className="metric-card-label">{copy.estimatedTime}</div>
                <div className="metric-card-value">{scenario.result.estimatedTime}</div>
              </div>
            </div>

            <div className="mt-4 space-y-2 text-sm text-[var(--text-secondary)]">
              <div className="flex justify-between">
                <span>{copy.gpu}</span>
                <span className="font-medium text-[var(--text-primary)]">
                  {scenario.hardware.gpusPerNode * scenario.hardware.nodeCount}x {scenario.hardware.gpuType}
                </span>
              </div>
              <div className="flex justify-between">
                <span>{copy.memoryState}</span>
                <span className={scenario.result.peakMemoryGb > scenario.hardware.gpuMemoryGb ? 'font-medium text-red-500' : 'font-medium text-[var(--text-primary)]'}>
                  {scenario.result.peakMemoryGb > scenario.hardware.gpuMemoryGb ? copy.overLimit : copy.withinLimit}
                </span>
              </div>
            </div>

            {scenario.result.recommendations.length > 0 && (
              <div className="workspace-subpanel workspace-subpanel-amber mt-4">
                <ul className="space-y-1 text-xs text-[var(--text-primary)]">
                  {scenario.result.recommendations.map((rec, i) => (
                    <li key={i}>• {rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
