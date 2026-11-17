document$.subscribe(() => {
  document
    .querySelectorAll(".md-nav--primary .md-nav__toggle.md-toggle--indeterminate")
    .forEach((el) => {
      el.checked = true;
    });
});
