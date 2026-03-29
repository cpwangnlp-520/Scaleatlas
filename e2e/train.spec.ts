import { expect, test } from '@playwright/test';
import { selectPreset, switchLocale } from './helpers';

test('training page keeps a stable configured baseline', async ({ page }) => {
  await page.goto('/train');
  await selectPreset(page, 'DeepSeek-V3');
  await page.getByRole('button', { name: '开始训练计算' }).click();

  await expect(page.locator('.planner-page')).toHaveScreenshot('train-zh-deepseek-result.png', {
    animations: 'disabled',
  });
});

test('training page keeps a stable english configured baseline', async ({ page }) => {
  await page.goto('/train');
  await switchLocale(page, 'en');
  await selectPreset(page, 'DeepSeek-V3');
  await page.getByRole('button', { name: 'Run training plan' }).click();

  await expect(page.locator('.planner-page')).toHaveScreenshot('train-en-deepseek-result.png', {
    animations: 'disabled',
  });
});
