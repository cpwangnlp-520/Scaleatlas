import { expect, test } from '@playwright/test';
import { expandDecoderSkeleton, selectPreset, switchLocale } from './helpers';

test('parameter page keeps a stable zh expanded MoE baseline', async ({ page }) => {
  await page.goto('/');
  await selectPreset(page, 'DeepSeek-V3');
  await expandDecoderSkeleton(page, 61);

  await expect(page.locator('.planner-page')).toHaveScreenshot('parameter-zh-deepseek-expanded.png', {
    animations: 'disabled',
  });
});

test('parameter page stays fully english in the expanded baseline', async ({ page }) => {
  await page.goto('/');
  await switchLocale(page, 'en');
  await selectPreset(page, 'DeepSeek-V3');
  await expandDecoderSkeleton(page, 61);

  await expect(page.locator('.planner-page')).toHaveScreenshot('parameter-en-deepseek-expanded.png', {
    animations: 'disabled',
  });
});

test('parameter page keeps a stable 1024px expanded baseline', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 1600 });
  await page.goto('/');
  await selectPreset(page, 'DeepSeek-V3');
  await expandDecoderSkeleton(page, 61);

  await expect(page.locator('.planner-page')).toHaveScreenshot('parameter-zh-deepseek-expanded-1024.png', {
    animations: 'disabled',
  });
});
