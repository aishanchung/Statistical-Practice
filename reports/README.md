畫面會像：
┌──────────────────────────────────────────────┐
│ 2026/10/16 研究進度報告                       │
├──────────────────────────────┬───────────────┤
│                              │ 報告列表       │
│                              │               │
│         PDF 預覽             │ 10/16 報告     │
│                              │ 10/09 報告     │
│                              │ 10/02 報告     │
│                              │               │
└──────────────────────────────┴───────────────┘

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
