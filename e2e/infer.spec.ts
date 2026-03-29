import { expect, test } from '@playwright/test';
import { selectPreset, switchLocale } from './helpers';

test('inference page keeps a stable configured baseline', async ({ page }) => {
  await page.goto('/infer');
  await selectPreset(page, 'DeepSeek-V3');
  await page.getByRole('button', { name: '开始推理计算' }).click();

  await expect(page.locator('.planner-page')).toHaveScreenshot('infer-zh-deepseek-result.png', {
    animations: 'disabled',
  });
});

test('inference page keeps a stable english configured baseline', async ({ page }) => {
  await page.goto('/infer');
  await switchLocale(page, 'en');
  await selectPreset(page, 'DeepSeek-V3');
  await page.getByRole('button', { name: 'Run inference plan' }).click();

  await expect(page.locator('.planner-page')).toHaveScreenshot('infer-en-deepseek-result.png', {
    animations: 'disabled',
  });
});
