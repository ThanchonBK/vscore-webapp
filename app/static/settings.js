/* settings.js -- the "defaults" button on the upload and result pages.
 *
 * Every settings control carries its default in data-default ("1"/"0" for a
 * checkbox). The button restores them all, including unticking any rejected
 * T/R matches, which are form-associated through form="rescore" even where
 * they sit outside the form element. It only resets the fields; nothing is
 * re-scored until the form is submitted.
 */
(function () {
  "use strict";
  document.querySelectorAll("[data-reset]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var form = btn.form || btn.closest("form");
      if (!form) return;
      Array.prototype.forEach.call(form.elements, function (el) {
        if (!el.dataset || !("default" in el.dataset)) return;
        if (el.type === "checkbox") el.checked = el.dataset["default"] === "1";
        else el.value = el.dataset["default"];
      });
    });
  });
})();
