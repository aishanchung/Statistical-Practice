function toggleSidebar() {

    const sidebar =
        document.getElementById(
            "sidebar"
        );

    sidebar.classList.toggle(
        "open"
    );
}


document.addEventListener(
    "click",
    function (event) {

        const sidebar =
            document.getElementById(
                "sidebar"
            );

        const button =
            document.querySelector(
                ".mobile-menu-button"
            );

        if (
            window.innerWidth <= 850 &&
            sidebar &&
            sidebar.classList.contains("open") &&
            !sidebar.contains(event.target) &&
            !button.contains(event.target)
        ) {

            sidebar.classList.remove(
                "open"
            );

        }

    }
);
