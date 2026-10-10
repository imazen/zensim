// dialviz client: theme toggle, site search, sortable/filterable tables, chart tooltips.
(function () {
  "use strict";
  var root = document.documentElement;
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

  // Theme: auto -> light -> dark -> auto.
  var themeBtn = document.getElementById("theme");
  function applyTheme(t) {
    if (t === "light" || t === "dark") root.setAttribute("data-theme", t); else root.removeAttribute("data-theme");
    if (themeBtn) themeBtn.textContent = t === "light" ? "Light" : t === "dark" ? "Dark" : "Auto";
  }
  applyTheme(store("dialviz-theme") || "auto");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var cur = store("dialviz-theme") || "auto";
    var next = cur === "auto" ? "light" : cur === "light" ? "dark" : "auto";
    store("dialviz-theme", next); applyTheme(next);
  });

  // Search over the generated index (search.js sets window.DIALVIZ_INDEX).
  var q = document.getElementById("q"), box = document.getElementById("results");
  var base = (document.body.getAttribute("data-base") || "");
  var sel = -1;
  function render(hits) {
    box.innerHTML = "";
    hits.slice(0, 40).forEach(function (h, i) {
      var a = document.createElement("a");
      a.href = base + h.u;
      var k = document.createElement("span"); k.className = "k"; k.textContent = h.k;
      a.appendChild(k); a.appendChild(document.createTextNode(h.t));
      if (h.d) { var d = document.createElement("span"); d.className = "d"; d.textContent = h.d; a.appendChild(d); }
      if (i === sel) a.className = "sel";
      box.appendChild(a);
    });
    if (!hits.length) { var p = document.createElement("a"); p.textContent = "No match"; box.appendChild(p); }
    box.classList.toggle("open", true);
  }
  function search(s) {
    var idx = window.DIALVIZ_INDEX || [];
    var terms = s.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) { box.classList.remove("open"); return []; }
    var scored = [];
    for (var i = 0; i < idx.length; i++) {
      var e = idx[i], hay = (e.t + " " + (e.d || "") + " " + (e.x || "")).toLowerCase(), t = e.t.toLowerCase(), sc = 0, ok = true;
      for (var j = 0; j < terms.length; j++) {
        var p = hay.indexOf(terms[j]);
        if (p < 0) { ok = false; break; }
        sc += t.indexOf(terms[j]) === 0 ? 0 : t.indexOf(terms[j]) > 0 ? 1 : 3;
      }
      if (ok) scored.push([sc, e.t.length, e]);
    }
    scored.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    return scored.map(function (x) { return x[2]; });
  }
  var hits = [];
  if (q && box) {
    q.addEventListener("input", function () { sel = -1; hits = search(q.value); if (q.value) render(hits); });
    q.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowDown") { sel = Math.min(sel + 1, Math.min(hits.length, 40) - 1); render(hits); ev.preventDefault(); }
      else if (ev.key === "ArrowUp") { sel = Math.max(sel - 1, 0); render(hits); ev.preventDefault(); }
      else if (ev.key === "Enter" && hits.length) { location.href = base + hits[Math.max(sel, 0)].u; }
      else if (ev.key === "Escape") { box.classList.remove("open"); q.blur(); }
    });
    document.addEventListener("click", function (ev) { if (!box.contains(ev.target) && ev.target !== q) box.classList.remove("open"); });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "/" && document.activeElement !== q && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { ev.preventDefault(); q.focus(); }
    });
  }

  // Sortable tables: th.sortable; numeric when every cell parses.
  function cellVal(td) { var v = td.getAttribute("data-v"); return v !== null ? v : td.textContent.trim(); }
  document.querySelectorAll("table.sortable").forEach(function (tbl) {
    var ths = tbl.querySelectorAll("thead th");
    ths.forEach(function (th, col) {
      th.classList.add("sortable");
      th.addEventListener("click", function () {
        var body = tbl.tBodies[0], rows = Array.prototype.slice.call(body.rows);
        var dir = th.getAttribute("data-dir") === "asc" ? -1 : 1;
        ths.forEach(function (o) { o.removeAttribute("data-dir"); });
        th.setAttribute("data-dir", dir === 1 ? "asc" : "desc");
        var vals = rows.map(function (r) { return r.cells[col] ? cellVal(r.cells[col]) : ""; });
        var numeric = vals.every(function (v) { return v === "" || v === "—" || !isNaN(parseFloat(v)); });
        rows.map(function (r, i) { return [r, vals[i]]; }).sort(function (a, b) {
          if (numeric) {
            var x = parseFloat(a[1]), y = parseFloat(b[1]);
            if (isNaN(x)) return 1; if (isNaN(y)) return -1;
            return (x - y) * dir;
          }
          return a[1].localeCompare(b[1], undefined, { numeric: true }) * dir;
        }).forEach(function (p) { body.appendChild(p[0]); });
      });
    });
  });

  // Table filter: input[data-filter=<table id>] and select[data-filter-col].
  document.querySelectorAll("[data-filter]").forEach(function (inp) {
    var tbl = document.getElementById(inp.getAttribute("data-filter"));
    var counter = document.querySelector("[data-count='" + inp.getAttribute("data-filter") + "']");
    function apply() {
      var controls = document.querySelectorAll("[data-filter='" + inp.getAttribute("data-filter") + "']");
      var shown = 0;
      Array.prototype.forEach.call(tbl.tBodies[0].rows, function (r) {
        var ok = true;
        controls.forEach(function (c) {
          var v = c.value.toLowerCase(); if (!v) return;
          var col = c.getAttribute("data-filter-col");
          var text = col !== null ? (r.cells[+col] ? r.cells[+col].textContent : "") : r.textContent;
          if (c.tagName === "SELECT" ? text.trim().toLowerCase() !== v : text.toLowerCase().indexOf(v) < 0) ok = false;
        });
        r.style.display = ok ? "" : "none"; if (ok) shown++;
      });
      if (counter) counter.textContent = shown + " of " + tbl.tBodies[0].rows.length + " shown";
    }
    inp.addEventListener(inp.tagName === "SELECT" ? "change" : "input", apply);
    apply();
  });

  // Tooltips: any element with data-tip inside an svg.chart or .matrix.
  var tip = document.createElement("div"); tip.id = "tip"; document.body.appendChild(tip);
  function show(ev) {
    var t = ev.target.closest("[data-tip]"); if (!t) { tip.style.display = "none"; return; }
    tip.textContent = ""; t.getAttribute("data-tip").split("\n").forEach(function (line, i) {
      var d = document.createElement("div"); d.textContent = line; if (i === 0) d.style.fontWeight = "600"; tip.appendChild(d);
    });
    tip.style.display = "block";
    var x = ev.clientX + 14, y = ev.clientY + 14, w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > innerWidth - 8) x = ev.clientX - w - 14; if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
  }
  document.addEventListener("mousemove", show);
  document.addEventListener("touchstart", function (ev) { show(ev.touches[0] ? Object.assign({}, ev.touches[0], { target: ev.target }) : ev); }, { passive: true });
  document.addEventListener("scroll", function () { tip.style.display = "none"; }, { passive: true });
})();
