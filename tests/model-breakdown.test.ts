import assert from 'node:assert/strict';
import test from 'node:test';

import { getModelBreakdown } from '../src/engine/model-breakdown.ts';
import { calculateMoEParamCount, calculateParamCount } from '../src/engine/memory.ts';
import { MODEL_PRESETS } from '../src/types/index.ts';
import type { ModelConfig } from '../src/types/index.ts';

test('dense model breakdown matches the total parameter formula', () => {
  const model = {
    sourceType: 'template',
    paramCountTotal: 0,
    ...MODEL_PRESETS['Llama-3-8B'],
  } as ModelConfig;

  const breakdown = getModelBreakdown(model);
  const total = calculateParamCount(model);
  const sum = breakdown.sections.reduce((acc, section) => acc + section.paramCount, 0);

  assert.equal(sum, total);
  assert.equal(breakdown.totalParamCount, total);
  assert.ok(breakdown.sections.some((section) => section.id === 'embedding'));
  assert.ok(breakdown.sections.some((section) => section.id === 'decoder'));
});

test('moe model breakdown isolates experts and still matches the total parameter formula', () => {
  const preset = MODEL_PRESETS['Mixtral-8x7B'];
  const model = {
    sourceType: 'template',
    paramCountTotal: 0,
    ...preset,
  } as ModelConfig;

  const breakdown = getModelBreakdown(model, {
    numLocalExperts: preset.moe?.numLocalExperts || 8,
    numExpertsPerTok: preset.moe?.numExpertsPerTok || 2,
  });
  const total = calculateMoEParamCount(model, {
    numLocalExperts: preset.moe?.numLocalExperts || 8,
    numExpertsPerTok: preset.moe?.numExpertsPerTok || 2,
  });
  const sum = breakdown.sections.reduce((acc, section) => acc + section.paramCount, 0);

  assert.equal(sum, total);
  assert.equal(breakdown.totalParamCount, total);
  assert.ok(breakdown.sections.some((section) => section.id === 'moe-experts'));
  assert.ok(breakdown.sections.some((section) => section.id === 'moe-router'));
});
