import type { Page } from '@playwright/test';

export async function selectPreset(page: Page, presetName: string) {
  const modelStep = page.locator('.planner-step-card[data-step="model"]');
  await modelStep.locator('select').first().selectOption(presetName);
}

export async function switchLocale(page: Page, locale: 'zh' | 'en') {
  await page.getByRole('button', { name: locale.toUpperCase() }).click();
}

export async function expandDecoderSkeleton(page: Page, layerCount: number) {
  await page.getByRole('button', { name: new RegExp(`Decoder Stack x ${layerCount}`) }).click();
}
