import { RUNTIME_COPY } from '../../content/runtimeCopy.ts';
import type { HardwareConfig, Locale, ParallelConfig } from '../../types';

interface MachineSplitMapProps {
  hardware: HardwareConfig;
  parallel: ParallelConfig;
  layerCount?: number;
  expertCount?: number;
  mode: 'train' | 'infer';
  title?: string;
  description?: string;
  locale?: Locale;
}

export function MachineSplitMap({
  hardware,
  parallel,
  layerCount,
  expertCount,
  mode,
  title,
  description,
  locale = 'zh',
}: MachineSplitMapProps) {
  const commonCopy = RUNTIME_COPY[locale].common;
  const sharedCopy = RUNTIME_COPY[locale].shared;
  const legendCopy = sharedCopy.legend;
  const resolvedTitle = title ?? sharedCopy.machineTitle;
  const resolvedDescription = description ?? sharedCopy.machineDescription;
  const totalAvailableGpus = hardware.gpusPerNode * hardware.nodeCount;
  const usedGpus = Math.min(totalAvailableGpus, parallel.tpSize * parallel.ppSize * parallel.dpSize);
  const stageSpan = Math.max(1, parallel.tpSize * parallel.ppSize);
  const layersPerStage = layerCount && parallel.ppSize > 0 ? Math.floor(layerCount / parallel.ppSize) : null;
  const nodeRows = Array.from({ length: hardware.nodeCount }, (_, nodeIndex) => {
    const gpus = Array.from({ length: hardware.gpusPerNode }, (_, localGpuIndex) => {
      const globalGpuIndex = nodeIndex * hardware.gpusPerNode + localGpuIndex;
      if (globalGpuIndex >= usedGpus) {
        return {
          globalGpuIndex,
          localGpuIndex,
          active: false,
        };
      }

      const replicaIndex = Math.floor(globalGpuIndex / stageSpan);
      const remainder = globalGpuIndex % stageSpan;
      const stageIndex = Math.floor(remainder / parallel.tpSize);
      const tpRank = remainder % parallel.tpSize;
      const crossesNode = parallel.tpSize > hardware.gpusPerNode;

      return {
        globalGpuIndex,
        localGpuIndex,
        active: true,
        replicaIndex,
        stageIndex,
        tpRank,
        carriesWeight: mode === 'infer' || parallel.zeroStage !== '3',
        carriesStage: parallel.ppSize > 1,
        carriesExpert: parallel.epSize > 1 && expertCount,
        crossesNode,
      };
    });

    return {
      nodeIndex,
      gpus,
    };
  });

  return (
    <div className="workspace-panel machine-split-card">
      <div className="workspace-panel-header">
        <div>
          <div className="workspace-panel-title">{resolvedTitle}</div>
          <p className="workspace-panel-description">{resolvedDescription}</p>
        </div>
      </div>

      <div className="machine-split-summary">
        <div className="machine-split-summary-item">
          <span>{commonCopy.cluster}</span>
          <strong>{hardware.nodeCount} x {hardware.gpusPerNode} = {totalAvailableGpus} GPU</strong>
        </div>
        <div className="machine-split-summary-item">
          <span>{commonCopy.parallel}</span>
          <strong>TP {parallel.tpSize} / PP {parallel.ppSize} / DP {parallel.dpSize} / EP {parallel.epSize}</strong>
        </div>
        <div className="machine-split-summary-item">
          <span>{commonCopy.used}</span>
          <strong>{usedGpus} GPU</strong>
        </div>
      </div>

      <div className="machine-split-legend">
        <span className="machine-split-legend-chip role-weight">{legendCopy.weight}</span>
        <span className="machine-split-legend-chip role-stage">{legendCopy.stage}</span>
        <span className="machine-split-legend-chip role-replica">{legendCopy.replica}</span>
        {parallel.epSize > 1 && <span className="machine-split-legend-chip role-expert">{legendCopy.expert}</span>}
      </div>

      <div className="machine-split-grid">
        {nodeRows.map((nodeRow) => (
          <div key={nodeRow.nodeIndex} className="machine-split-node">
            <div className="machine-split-node-title">{commonCopy.node} {nodeRow.nodeIndex}</div>
            <div className="machine-split-node-grid">
              {nodeRow.gpus.map((gpu) => (
                <div
                  key={gpu.globalGpuIndex}
                  className={`machine-split-cell${gpu.active ? '' : ' is-idle'}`}
                >
                  <div className="machine-split-cell-header">
                    <strong>GPU {gpu.localGpuIndex}</strong>
                    <span>#{gpu.globalGpuIndex}</span>
                  </div>

                  {gpu.active ? (
                    <>
                      <div className="machine-split-cell-body">
                        <div>DP {gpu.replicaIndex}</div>
                        <div>PP {gpu.stageIndex}</div>
                        <div>TP {gpu.tpRank}</div>
                      </div>

                      <div className="machine-split-cell-roles">
                        {gpu.carriesWeight && (
                          <span className="machine-split-cell-role role-weight">{legendCopy.weight}</span>
                        )}
                        {gpu.carriesStage && (
                          <span className="machine-split-cell-role role-stage">{legendCopy.stage} {gpu.stageIndex}</span>
                        )}
                        <span className="machine-split-cell-role role-replica">{legendCopy.replica} {gpu.replicaIndex}</span>
                        {gpu.carriesExpert && (
                          <span className="machine-split-cell-role role-expert">{legendCopy.expert}</span>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="machine-split-cell-idle">{commonCopy.idle.toLowerCase()}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="machine-split-notes">
        <div className="machine-split-note">
          {mode === 'train'
            ? sharedCopy.trainNote
            : sharedCopy.inferNote}
        </div>
        <div className="machine-split-note">
          {layersPerStage
            ? sharedCopy.layersPerStage(layersPerStage)
            : sharedCopy.noPipeline}
        </div>
        {parallel.epSize > 1 && expertCount ? (
          <div className="machine-split-note">
            {sharedCopy.expertsPerShard(expertCount, parallel.epSize)}
          </div>
        ) : (
          <div className="machine-split-note">{sharedCopy.noExpertParallel}</div>
        )}
      </div>
    </div>
  );
}
