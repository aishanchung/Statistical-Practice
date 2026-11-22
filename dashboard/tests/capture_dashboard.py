import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/congestion_2026/outputs/workpackages_0ABCDE_v1'
DEST=OUT/'dashboard_screenshots';DEST.mkdir(exist_ok=True)
with sync_playwright() as playwright:
    browser=playwright.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1680,'height':1600},device_scale_factor=1)
    errors=[];layout_checks=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://localhost:8501',wait_until='domcontentloaded')
    page.get_by_role('button',name='產生圖表PNG',exact=True).first.wait_for(timeout=60000)
    page.locator('[data-testid="stMetricValue"]').first.wait_for(timeout=60000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(DEST/'overview_h30.png'),full_page=True)
    page.get_by_role('button',name='產生圖表PNG',exact=True).first.click()
    page.get_by_role('button',name='下載PNG',exact=True).first.wait_for(timeout=60000)
    with page.expect_download() as download:
        page.get_by_role('button',name='下載PNG',exact=True).first.click()
    download.value.save_as(str(DEST/'downloaded_probability_chart.png'))
    page.get_by_role('link',name='Event Replay',exact=True).click()
    page.get_by_text('真實歷史事件回放',exact=True).wait_for(timeout=30000)
    page.get_by_text('顯示事後答案（分析用）',exact=True).click()
    page.get_by_text('以下為事後真值',exact=False).wait_for(timeout=30000)
    page.set_viewport_size({'width':1680,'height':3000})
    for label,event in [('successful_event','2025-01-31 15:40'),('missed_event','2025-02-15 12:30')]:
        select=page.get_by_role('combobox',name='歷史Y2事件',exact=True)
        select.click();page.keyboard.type(event);page.keyboard.press('Enter')
        page.get_by_text('以下為事後真值',exact=False).wait_for(timeout=30000)
        expected_target='預測目標 15:45' if label=='successful_event' else '預測目標 12:40'
        page.get_by_text(expected_target,exact=False).first.wait_for(timeout=60000)
        page.locator('[data-testid="stStatusWidget"]').wait_for(state='hidden',timeout=60000)
        page.wait_for_load_state('networkidle')
        page.wait_for_timeout(800)
        shape_x=page.locator('.js-plotly-plot').nth(1).evaluate('el => el.layout.shapes[0].x0')
        assert str(shape_x).replace('T',' ')[:16]==event,(shape_x,event)
        layout_checks.append({'event':event,'plotted_onset':shape_x,'naive_local_ISO_alignment':True})
        page.screenshot(path=str(DEST/(label+'.png')),full_page=True)
    browser.close()
    (OUT/'dashboard_browser_tests.json').write_text(json.dumps({'page_errors':errors,'event_onset_alignment':layout_checks,'screenshots':[p.name for p in DEST.glob('*.png')],'real_data':True},ensure_ascii=False,indent=2),encoding='utf8')
    assert not errors,errors
print('Real browser screenshots and PNG download verified')
