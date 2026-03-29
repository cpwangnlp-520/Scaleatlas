import { Link } from 'react-router-dom';
import { usePlannerStore } from '../stores';
import type { Locale } from '../types';

const HOME_PAGE_COPY: Record<Locale, any> = {
  zh: {
    kicker: '桌面端双栏工作台',
    title: 'ScaleAtlas',
    description: '开源的大模型参数、训练显存和推理容量规划器。输入模型结构、任务形态和硬件约束后，即可查看峰值显存、并行建议、容量风险和吞吐区间。',
    heroTitle: '从想法到资源预算的一站式入口',
    heroDescription: '先定场景，再挑 GPU，再决定并行策略。页面信息密度为桌面端双栏工作台优化，适合在方案评审、容量规划和上线前复核时快速浏览。',
    heroPrimary: '开始训练规划',
    heroSecondary: '开始推理规划',
    previewTitle: '资源工作台预览',
    previewDescription: '右侧卡片模拟规划完成后的结果层级，让你在进入表单前先看到最终输出会聚焦哪些信号。',
    modelTypesTitle: '支持的模型类型',
    modelTypesDescription: 'Dense、MoE、多模态和 TTS 场景都可以先拿这套估算逻辑做第一轮资源定标。',
    featuresTitle: '核心功能',
    featuresDescription: '首页不只负责导航，也直接说明结果区会给你什么决策信息。',
    entryCards: [
      {
        to: '/train',
        title: '训练规划',
        description: '拆解参数、激活、优化器与并行策略，快速判断训练是否能落在目标 GPU 集群上。',
        cta: '开始训练规划',
        accent: 'text-blue-600',
      },
      {
        to: '/infer',
        title: '推理规划',
        description: '评估部署显存、KV Cache、并发上限和吞吐表现，提前排除线上容量风险。',
        cta: '开始推理规划',
        accent: 'text-orange-500',
      },
      {
        to: '/compare',
        title: '方案对比',
        description: '并排评估不同 GPU、节点数和 ZeRO/TP/PP 组合，压缩选型时间。',
        cta: '查看方案对比',
        accent: 'text-emerald-600',
      },
    ],
    previewMetrics: [
      {
        label: '训练资源概览',
        value: '8x H100 SXM',
        note: '70B SFT / ZeRO-2 / BF16',
      },
      {
        label: '推理容量速览',
        value: '64 并发',
        note: '72B / vLLM / P95 2k 上下文',
      },
      {
        label: '风险雷达',
        value: '2 个提醒',
        note: '长序列激活偏高，跨节点 TP 需谨慎',
      },
    ],
    previewFlows: [
      {
        label: '训练流',
        title: '70B / 4k 序列 / ZeRO-2 / 预估 45.2 GB 峰值',
        description: '将参数、梯度、优化器和激活单独拆解，避免只看总量却看不到真正的瓶颈。',
      },
      {
        label: '推理流',
        title: '在线服务 / 64 并发 / KV Cache 受控',
        description: '提前评估权重、KV Cache 和运行缓冲，确认吞吐与并发是否能落到同一组卡上。',
      },
    ],
    modelTypes: ['Dense LLM', 'MoE 模型', '多模态模型', 'TTS 语音模型'],
    features: [
      { title: '显存拆解', desc: '参数、梯度、优化器、KV Cache 与运行缓冲按组件拆开看。' },
      { title: '并行建议', desc: 'TP / PP / DP / ZeRO / EP 自动给出可行组合和风险权衡。' },
      { title: '风险预警', desc: 'OOM、通信瓶颈、长序列和高并发风险在结果区直接标红。' },
      { title: 'HF Config 导入', desc: '粘贴或拖入 `config.json`，自动回填主干架构字段。' },
    ],
  },
  en: {
    kicker: 'Desktop Dual-Column Workspace',
    title: 'ScaleAtlas',
    description: 'An open-source planner for LLM parameters, training memory, and inference capacity. Enter model structure, workload shape, and hardware constraints to review peak memory, parallel recommendations, risk, and throughput.',
    heroTitle: 'A single entry point from idea to resource budget',
    heroDescription: 'Choose the workload first, then the GPU, then the parallel strategy. The page density is tuned for a desktop dual-column workspace and works well for design review, capacity planning, and pre-launch validation.',
    heroPrimary: 'Start Training Planning',
    heroSecondary: 'Start Inference Planning',
    previewTitle: 'Workspace Preview',
    previewDescription: 'The right-side cards preview the result hierarchy so you can see which signals the final output emphasizes before filling the forms.',
    modelTypesTitle: 'Supported Model Types',
    modelTypesDescription: 'Dense, MoE, multimodal, and TTS workloads can all use this estimator for a first-pass sizing run.',
    featuresTitle: 'Core Features',
    featuresDescription: 'The landing page is not only navigation. It also shows the decision signals you should expect in the result area.',
    entryCards: [
      {
        to: '/train',
        title: 'Training Planning',
        description: 'Break down parameters, activations, optimizer state, and parallel strategy to quickly judge whether training fits the target GPU cluster.',
        cta: 'Start Training Planning',
        accent: 'text-blue-600',
      },
      {
        to: '/infer',
        title: 'Inference Planning',
        description: 'Evaluate deployment memory, KV cache, concurrency ceilings, and throughput before capacity risks reach production.',
        cta: 'Start Inference Planning',
        accent: 'text-orange-500',
      },
      {
        to: '/compare',
        title: 'Scenario Comparison',
        description: 'Compare GPU types, node counts, and ZeRO/TP/PP combinations side by side to shorten hardware selection.',
        cta: 'View Comparisons',
        accent: 'text-emerald-600',
      },
    ],
    previewMetrics: [
      {
        label: 'Training Snapshot',
        value: '8x H100 SXM',
        note: '70B SFT / ZeRO-2 / BF16',
      },
      {
        label: 'Inference Snapshot',
        value: '64 Concurrency',
        note: '72B / vLLM / P95 2k context',
      },
      {
        label: 'Risk Radar',
        value: '2 Alerts',
        note: 'Long-sequence activation is high and cross-node TP needs review',
      },
    ],
    previewFlows: [
      {
        label: 'Training Flow',
        title: '70B / 4k sequence / ZeRO-2 / estimated 45.2 GB peak',
        description: 'Split parameters, gradients, optimizer state, and activations apart so the real bottleneck is visible instead of being buried in one total.',
      },
      {
        label: 'Inference Flow',
        title: 'Online serving / 64 concurrency / KV cache under control',
        description: 'Estimate weights, KV cache, and runtime buffers early so throughput and concurrency are validated against the same GPU pool.',
      },
    ],
    modelTypes: ['Dense LLM', 'MoE Models', 'Multimodal Models', 'TTS Speech Models'],
    features: [
      { title: 'Memory Breakdown', desc: 'Inspect parameters, gradients, optimizer state, KV cache, and runtime buffers by component.' },
      { title: 'Parallel Guidance', desc: 'Get feasible TP / PP / DP / ZeRO / EP combinations with risk trade-offs.' },
      { title: 'Risk Warnings', desc: 'OOM, communication bottlenecks, long-sequence pressure, and high-concurrency risk show up directly in the result area.' },
      { title: 'HF Config Import', desc: 'Paste or drop `config.json` to backfill the backbone architecture automatically.' },
    ],
  },
};

export function HomePage() {
  const locale = usePlannerStore((state) => state.locale);
  const copy = HOME_PAGE_COPY[locale];

  return (
    <section className="space-y-8">
      <div className="workspace-page-header">
        <div>
          <div className="workspace-page-kicker">{copy.kicker}</div>
          <h1 className="workspace-page-title">{copy.title}</h1>
          <p className="workspace-page-description">{copy.description}</p>
        </div>
      </div>

      <div className="workspace-grid">
        <div className="workspace-form-column">
          <div className="workspace-panel workspace-panel-accent">
            <div className="workspace-panel-header">
              <div>
                <div className="workspace-panel-title">{copy.heroTitle}</div>
                <p className="workspace-panel-description">{copy.heroDescription}</p>
              </div>
            </div>

            <div className="workspace-action-row">
              <Link to="/train" className="surface-button surface-button-inline">
                {copy.heroPrimary}
              </Link>
              <Link to="/infer" className="surface-link-button">
                {copy.heroSecondary}
              </Link>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {copy.entryCards.map((card: any) => (
              <Link key={card.to} to={card.to} className="workspace-panel">
                <div className={`text-sm font-semibold ${card.accent}`}>{card.title}</div>
                <p className="mt-3 text-sm text-[var(--text-secondary)]">{card.description}</p>
                <div className={`mt-5 text-sm font-semibold ${card.accent}`}>{card.cta}</div>
              </Link>
            ))}
          </div>
        </div>

        <div className="workspace-result-column">
          <div className="workspace-panel">
            <div className="workspace-panel-header">
              <div>
                <div className="workspace-panel-title">{copy.previewTitle}</div>
                <p className="workspace-panel-description">{copy.previewDescription}</p>
              </div>
            </div>

            <div className="metric-grid">
              {copy.previewMetrics.map((item: any) => (
                <div key={item.label} className="metric-card">
                  <div className="metric-card-label">{item.label}</div>
                  <div className="metric-card-value">{item.value}</div>
                  <div className="metric-card-note">{item.note}</div>
                </div>
              ))}
            </div>

            <div className="mt-5 space-y-3">
              {copy.previewFlows.map((flow: any) => (
                <div key={flow.label} className="workspace-subpanel">
                  <div className="control-label">{flow.label}</div>
                  <div className="mt-2 text-base font-semibold text-[var(--text-primary)]">{flow.title}</div>
                  <p className="mt-2 text-sm text-[var(--text-secondary)]">{flow.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="workspace-panel">
        <div className="workspace-panel-header">
          <div>
            <div className="workspace-panel-title">{copy.modelTypesTitle}</div>
            <p className="workspace-panel-description">{copy.modelTypesDescription}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          {copy.modelTypes.map((type: string) => (
            <span key={type} className="surface-link-button">
              {type}
            </span>
          ))}
        </div>
      </div>

      <div className="workspace-panel">
        <div className="workspace-panel-header">
          <div>
            <div className="workspace-panel-title">{copy.featuresTitle}</div>
            <p className="workspace-panel-description">{copy.featuresDescription}</p>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {copy.features.map((feature: any) => (
            <div key={feature.title} className="workspace-subpanel">
              <div className="text-sm font-semibold text-[var(--text-primary)]">{feature.title}</div>
              <p className="mt-2 text-sm text-[var(--text-secondary)]">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
