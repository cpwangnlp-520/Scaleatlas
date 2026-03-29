import type { SkeletonBlock } from '../modelSkeleton.ts';
import {
  formatDetailShare,
  formatParams,
  getDetailToneClass,
} from '../modelSkeleton.ts';

interface ParameterSkeletonProps {
  copy: Record<string, any>;
  mainBlocks: SkeletonBlock[];
  sideBlocks: SkeletonBlock[];
  expandedBlockId: string | null;
  onToggleBlock: (id: string) => void;
}

export function ParameterSkeleton({
  copy,
  mainBlocks,
  sideBlocks,
  expandedBlockId,
  onToggleBlock,
}: ParameterSkeletonProps) {
  const expandedBlock = [...mainBlocks, ...sideBlocks].find((block) => block.id === expandedBlockId) || null;
  const expandedMainBlock = expandedBlock?.flow === 'main' ? expandedBlock : null;
  const expandedSideBlock = expandedBlock?.flow === 'side' ? expandedBlock : null;

  return (
    <>
      <div className="parameter-skeleton-shell">
        <div className="parameter-skeleton-lane parameter-skeleton-lane-main">
          <div className="parameter-skeleton-legend">{copy.skeletonLegend}</div>
          <div className="parameter-skeleton-main-flow">
            {mainBlocks.map((block, index) => (
              <div key={block.id} className="parameter-skeleton-item">
                <button
                  type="button"
                  className={`parameter-skeleton-node parameter-skeleton-node-tone-${block.tone}${expandedBlockId === block.id ? ' is-expanded' : ''}`}
                  onClick={() => onToggleBlock(block.id)}
                >
                  <div className="parameter-skeleton-node-header">
                    <div className="parameter-skeleton-node-title">{block.label}</div>
                  </div>
                  <div className="parameter-skeleton-node-note">{block.note}</div>
                  <div className="parameter-skeleton-node-kpis">
                    <span className="parameter-skeleton-node-kpi">{formatParams(block.paramCount)}</span>
                    <span className="parameter-skeleton-node-kpi">{block.memoryGb.bf16.toFixed(1)} GB</span>
                  </div>
                </button>
                {index < mainBlocks.length - 1 && <div className="parameter-skeleton-arrow" aria-hidden="true" />}
              </div>
            ))}
          </div>
          {expandedMainBlock && (
            <SkeletonExpansion
              copy={copy}
              block={expandedMainBlock}
            />
          )}
        </div>

        {sideBlocks.length > 0 && (
          <div className="parameter-skeleton-lane parameter-skeleton-lane-side">
            <div className="parameter-skeleton-branch-label">{copy.branchLegend}</div>
            <div className="parameter-skeleton-branch-flow">
              {sideBlocks.map((block, index) => (
                <div key={block.id} className="parameter-skeleton-item">
                  <button
                    type="button"
                    className={`parameter-skeleton-node parameter-skeleton-node-tone-${block.tone} parameter-skeleton-node-side${expandedBlockId === block.id ? ' is-expanded' : ''}`}
                    onClick={() => onToggleBlock(block.id)}
                  >
                    <div className="parameter-skeleton-node-header">
                      <div className="parameter-skeleton-node-title">{block.label}</div>
                    </div>
                    <div className="parameter-skeleton-node-note">{block.note}</div>
                    <div className="parameter-skeleton-node-kpis">
                      <span className="parameter-skeleton-node-kpi">{formatParams(block.paramCount)}</span>
                      <span className="parameter-skeleton-node-kpi">{block.memoryGb.bf16.toFixed(1)} GB</span>
                    </div>
                  </button>
                  {index < sideBlocks.length - 1 && <div className="parameter-skeleton-arrow" aria-hidden="true" />}
                </div>
              ))}
            </div>
            {expandedSideBlock && (
              <SkeletonExpansion
                copy={copy}
                block={expandedSideBlock}
              />
            )}
          </div>
        )}
      </div>

      {!expandedBlock && (
        <div className="planner-empty-copy">{copy.clickHint}</div>
      )}
    </>
  );
}

function SkeletonExpansion({
  block,
  copy,
}: {
  block: SkeletonBlock;
  copy: Record<string, any>;
}) {
  return (
    <div className={`parameter-skeleton-expanded parameter-skeleton-expanded-tone-${block.tone}`}>
      <div className="parameter-skeleton-detail-strip">
        {block.internalLabels.map((label, index) => (
          <div key={`${block.id}-${label}`} className="parameter-skeleton-detail-step">
            <span className="parameter-skeleton-detail-node">{label}</span>
            {index < block.internalLabels.length - 1 && (
              <div className="parameter-skeleton-detail-arrow" aria-hidden="true" />
            )}
          </div>
        ))}
      </div>

      <div className="parameter-skeleton-metric-grid">
        <div className="parameter-skeleton-metric-card">
          <span>{copy.coreMetrics.params}</span>
          <strong>{formatParams(block.paramCount)}</strong>
        </div>
        <div className="parameter-skeleton-metric-card">
          <span>BF16</span>
          <strong>{block.memoryGb.bf16.toFixed(1)} GB</strong>
        </div>
        <div className="parameter-skeleton-metric-card">
          <span>FP16</span>
          <strong>{block.memoryGb.fp16.toFixed(1)} GB</strong>
        </div>
      </div>

      {block.details.length > 0 && (
        <div className="parameter-skeleton-detail-list">
          <div className="parameter-skeleton-detail-table-header">
            <span>{copy.skeletonDetailHeaders.module}</span>
            <span>{copy.skeletonDetailHeaders.params}</span>
            <span>{copy.skeletonDetailHeaders.memory}</span>
            <span>{copy.skeletonDetailHeaders.share}</span>
          </div>
          {block.details.map((detail) => (
            <div
              key={detail.id}
              className={`parameter-skeleton-detail-item parameter-skeleton-detail-item-tone-${getDetailToneClass(detail.id)}`}
            >
              <div className="parameter-skeleton-detail-main">
                <div className="parameter-skeleton-detail-title">{detail.label}</div>
                {detail.note && <div className="parameter-skeleton-detail-note">{detail.note}</div>}
              </div>
              <div className="parameter-skeleton-detail-cell parameter-skeleton-detail-value">{formatParams(detail.paramCount)}</div>
              <div className="parameter-skeleton-detail-cell">{detail.memoryGb.bf16.toFixed(1)} GB</div>
              <div className="parameter-skeleton-detail-cell">{formatDetailShare(detail.paramCount, block.paramCount)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
