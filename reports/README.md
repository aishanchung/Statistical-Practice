

假設下週是：
2026/10/23

先把檔案放：
reports/
├── 2026-10-23_report.pdf
└── 2026-10-23_report.pptx

然後只要在 reports.html 的報告列表加：
<div
    class="report-item"
    onclick="
        openReport(
            this,
            'reports/2026-10-23_report.pdf',
            '2026/10/23 研究進度報告',
            'VD 缺失檢查與 V2_LITE 建置'
        )
    "
>

<div class="report-date">
2026/10/23
</div>

<div class="report-name">
研究進度報告
</div>

<div class="report-download">

<a
    href="reports/2026-10-23_report.pptx"
    onclick="event.stopPropagation();"
>
下載 PPT
</a>

&nbsp;｜&nbsp;

<a
    href="reports/2026-10-23_report.pdf"
    target="_blank"
    onclick="event.stopPropagation();"
>
開啟 PDF
</a>

</div>

</div>

## 2026/11/12 第一次研究進度報告

文字版：[2026-11-12_summary.html](2026-11-12_summary.html)。

原始 35 頁簡報請放在本目錄，統一命名為 `2026-11-12_report.pdf` 和 `2026-11-12_report.pptx`。reports.html 已設定在檔案存在後自動顯示連結及 PDF 預覽；上傳前顯示文字版，不產生失效連結。
