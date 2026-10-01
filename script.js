function filterFields() {
    const input = document
        .getElementById("fieldSearch")
        .value
        .toLowerCase();

    const rows = document.querySelectorAll(
        ".field-row"
    );

    rows.forEach(row => {
        const text = row
            .innerText
            .toLowerCase();

        if (text.includes(input)) {
            row.style.display = "";
        } else {
            row.style.display = "none";
        }
    });
}
