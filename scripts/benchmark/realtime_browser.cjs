// Optional browser acceptance: requires Playwright and Chromium on the test host.
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_EXECUTABLE_PATH || undefined,
    args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      ...(process.env.TEST_WAV ? [`--use-file-for-fake-audio-capture=${process.env.TEST_WAV}`] : [])],
  });
  try {
    for (const [name, width, height] of [['desktop', 1280, 800], ['mobile', 390, 844]]) {
      const page = await browser.newPage({viewport: {width, height}});
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto((process.env.ASR_URL || 'http://localhost:4174') + '/realtime');
      if (process.env.API_KEY) await page.locator('#key').fill(process.env.API_KEY);
      for (let run = 0; run < 2; run++) {
        await page.locator('#start').click();
        await page.waitForFunction(() => !document.getElementById('stop').disabled, undefined, {timeout: 15000});
        await page.waitForTimeout(4000);
        await page.locator('#stop').click();
        await page.waitForFunction(() => document.getElementById('status').textContent === '转写完成', undefined, {timeout: 30000});
        if (process.env.TEST_WAV && !(await page.locator('#text').textContent()).trim()) throw Error('No microphone transcript');
      }
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error(`${name}: horizontal overflow`);
      if (errors.length) throw Error(errors.join('\n'));
      console.log(`${name}: recording, final transcript, repeat recording, layout OK`);
      await page.close();
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
