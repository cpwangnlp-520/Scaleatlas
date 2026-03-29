import assert from 'node:assert/strict';
import test from 'node:test';

import { usePlannerStore } from '../src/stores/planner.ts';
import type { ParsedHFConfig } from '../src/types/index.ts';

test('applying imported config persists raw text and parsed fields in the shared store', () => {
  const store = usePlannerStore;
  store.getState().reset();

  const parsed: ParsedHFConfig = {
    modelFamily: 'multimodal',
    modelConfig: {
      modelName: 'Qwen-Shared',
      modelFamily: 'multimodal',
      sourceType: 'hf_config_json',
      hiddenSize: 4096,
      numHiddenLayers: 48,
      numAttentionHeads: 32,
      numKeyValueHeads: 8,
      intermediateSize: 14336,
      vocabSize: 151936,
      maxPositionEmbeddings: 32768,
      normType: 'rmsnorm',
      attentionType: 'gqa',
      positionEncodingType: 'rope',
      ffnActivation: 'swiglu',
    },
    moeConfig: undefined,
    multimodalConfig: {
      visionHiddenSize: 1152,
      visionNumLayers: 27,
      projectorHiddenSize: 3072,
    },
    unrecognizedFields: [],
    warnings: [],
  };

  store.getState().applyImportedConfig('{"model_type":"qwen-shared"}', parsed);

  const state = store.getState();
  assert.equal(state.modelInputMode, 'config');
  assert.equal(state.importedConfigText, '{"model_type":"qwen-shared"}');
  assert.equal(state.lastImportedConfig?.modelConfig.modelName, 'Qwen-Shared');
  assert.equal(state.model.modelName, 'Qwen-Shared');
  assert.equal(state.model.hiddenSize, 4096);
  assert.equal(state.model.normType, 'rmsnorm');
  assert.equal(state.model.attentionType, 'gqa');
  assert.equal(state.model.positionEncodingType, 'rope');
  assert.equal(state.multimodal.visionHiddenSize, 1152);
});

test('clearing imported config keeps parsed fields but returns the editor to custom mode', () => {
  const store = usePlannerStore;
  store.getState().reset();

  const parsed: ParsedHFConfig = {
    modelFamily: 'dense',
    modelConfig: {
      modelName: 'Qwen-Clear',
      modelFamily: 'dense',
      sourceType: 'hf_config_json',
      hiddenSize: 3584,
      numHiddenLayers: 28,
      numAttentionHeads: 28,
      numKeyValueHeads: 4,
      intermediateSize: 18944,
      vocabSize: 151936,
      maxPositionEmbeddings: 32768,
      ffnActivation: 'swiglu',
    },
    unrecognizedFields: [],
    warnings: [],
  };

  store.getState().applyImportedConfig('{"model_type":"qwen-clear"}', parsed);
  store.getState().clearImportedConfig();

  const state = store.getState();
  assert.equal(state.modelInputMode, 'custom');
  assert.equal(state.importedConfigText, '');
  assert.equal(state.lastImportedConfig, null);
  assert.equal(state.model.modelName, 'Qwen-Clear');
  assert.equal(state.model.hiddenSize, 3584);
});

test('locale is stored globally and can switch between zh and en', () => {
  const store = usePlannerStore;
  store.getState().reset();

  assert.equal(store.getState().locale, 'zh');
  store.getState().setLocale('en');
  assert.equal(store.getState().locale, 'en');
  store.getState().setLocale('zh');
  assert.equal(store.getState().locale, 'zh');
});
