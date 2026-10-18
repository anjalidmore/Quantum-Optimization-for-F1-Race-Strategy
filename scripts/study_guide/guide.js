/* Study guide behaviour. No dependencies; everything degrades to plain
   anchors if a feature is missing. */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }
  };

  /* ---------------- Theme ---------------- */
  var root = document.documentElement;
  var savedTheme = store.get("guide-theme");
  if (savedTheme === "light" || savedTheme === "dark") root.setAttribute("data-theme", savedTheme);
  function isDark() {
    var t = root.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  $("#theme-toggle").addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    store.set("guide-theme", next);
  });

  /* ---------------- Tabs + routing ----------------
     Every page is an element with data-view; data-crumb is its breadcrumb
     label and data-parent the id of the view "above" it. The tab is read from
     the closest [data-tab]. Nothing here knows the individual pages. */
  var panels = {};
  $$("[data-tab]").forEach(function (p) { panels[p.getAttribute("data-tab")] = p; });
  var tabButtons = $$(".tabs [role=tab]");
  var lastHash = { guide: "home", diagrams: "diagrams", code: "p-home", reports: "reports-home", showcase: "slide-1" };
  var current = null, currentView = null;
  var backBtn = $("#back-btn"), crumbs = $("#breadcrumb");

  function showTab(name) {
    current = name;
    Object.keys(panels).forEach(function (k) { panels[k].hidden = k !== name; });
    tabButtons.forEach(function (b) {
      var on = b.getAttribute("data-target") === name;
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
    });
    document.body.classList.toggle("on-showcase", name === "showcase");
    document.body.classList.toggle("no-sidebar", !panels[name].querySelector(".sidebar"));
    closeNav();
    if (name === "showcase") deck.layout();
  }

  // Sibling views inside one tab: show only the active one.
  function showView(view) {
    var parent = view.parentElement;
    $$(":scope > [data-view]", parent).forEach(function (v) { v.hidden = v !== view; });
    $$("[data-doc-link], [data-practical-link]").forEach(function (a) {
      var key = a.getAttribute("data-doc-link") || a.getAttribute("data-practical-link");
      var id = a.hasAttribute("data-doc-link") ? key : "p-" + key;
      if (id === view.id) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
  }

  function chain(view) {
    var out = [], v = view, guard = 0;
    while (v && guard++ < 8) {
      out.unshift(v);
      var pid = v.getAttribute("data-parent");
      v = pid ? document.getElementById(pid) : null;
    }
    return out;
  }

  function renderCrumbs(view) {
    var items = chain(view);
    crumbs.innerHTML = "";
    items.forEach(function (v, i) {
      var li = document.createElement("li");
      var label = v.getAttribute("data-crumb") || "";
      if (i === items.length - 1) {
        var mid = v.getAttribute("data-crumb-mid");
        if (mid) { var m = document.createElement("span"); m.className = "mid"; m.textContent = mid; li.appendChild(m); }
        var cur = document.createElement("span"); cur.setAttribute("aria-current", "page"); cur.textContent = label;
        li.appendChild(cur);
      } else {
        var a = document.createElement("a"); a.href = "#" + v.id; a.textContent = label; li.appendChild(a);
      }
      crumbs.appendChild(li);
    });
    var root = !view.getAttribute("data-parent");
    var inApp = history.state && history.state.depth > 0;
    backBtn.disabled = root && !inApp;
    backBtn.title = backBtn.disabled ? "This is the start page" :
      (inApp ? "Back to the previous page" : "Up to " + (document.getElementById(view.getAttribute("data-parent")) || {}).getAttribute("data-crumb"));
  }

  // History entries are stamped with a depth: 0 for the page the visitor
  // arrived on, +1 for each navigation inside the guide. Back can then tell
  // "the previous page is ours" (history.back) from "they arrived here
  // directly" (go to the parent view instead of leaving the guide).
  var depth = 0;
  function stamped() { return history.state && typeof history.state.depth === "number"; }
  function stamp() {  // the entry the visitor arrived on (or reloaded)
    if (stamped()) { depth = history.state.depth; return; }
    try { history.replaceState({ depth: depth }, ""); } catch (e) { /* file:// in some browsers */ }
  }
  // An unstamped entry is a new navigation inside the guide; a stamped one is
  // back/forward. Chrome fires popstate before hashchange on a fragment
  // navigation, so whichever event sees the entry first stamps it.
  function onNavigate() {
    if (stamped()) { depth = history.state.depth; return; }
    depth += 1;
    try { history.replaceState({ depth: depth }, ""); } catch (e) { /* file:// in some browsers */ }
  }
  backBtn.addEventListener("click", function () {
    if (history.state && history.state.depth > 0) { history.back(); return; }
    // Arrived here directly: go up to the parent view in place (replacing this
    // entry, not pushing one), so pressing Back again keeps climbing instead of
    // bouncing between the parent and the page the visitor started on.
    var pid = currentView && currentView.getAttribute("data-parent");
    if (!pid) return;
    try { history.replaceState({ depth: depth }, "", "#" + pid); } catch (e) { location.hash = pid; return; }
    route();
  });

  function route() {
    var id = decodeURIComponent((location.hash || "").slice(1)) || "home";
    var el = document.getElementById(id);
    if (!el) { id = "home"; el = document.getElementById(id); }
    var panel = el.closest("[data-tab]");
    var tab = panel ? panel.getAttribute("data-tab") : "guide";
    if (tab !== "showcase" && deck.isFullscreen()) deck.exitFullscreen();
    showTab(tab);
    lastHash[tab] = id;
    var view = el.closest("[data-view]") || el;
    currentView = view;
    if (tab === "showcase") {
      var slide = el.closest("[data-slide]");
      deck.go(slide ? parseInt(slide.getAttribute("data-slide"), 10) : deck.index(), true);
      window.scrollTo(0, 0);
    } else {
      showView(view);
      scrollTo(el === view ? null : el);
    }
    renderCrumbs(view);
  }

  function scrollTo(el) {
    if (el) { el.scrollIntoView({ block: "start" }); }
    else { window.scrollTo(0, 0); }
  }

  tabButtons.forEach(function (b, i) {
    b.addEventListener("click", function () {
      var name = b.getAttribute("data-target");
      var target = name === "showcase" ? "slide-" + deck.index() : lastHash[name];
      if (location.hash.slice(1) === target) route(); else location.hash = target;
    });
    b.addEventListener("keydown", function (e) {
      var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      var n = tabButtons[(i + d + tabButtons.length) % tabButtons.length];
      n.focus(); n.click();
    });
  });
  window.addEventListener("popstate", onNavigate);
  window.addEventListener("hashchange", function () { onNavigate(); route(); });

  /* ---------------- Mobile navigation drawer ---------------- */
  var menu = $("#menu-toggle");
  function closeNav() { document.body.classList.remove("nav-open"); menu.setAttribute("aria-expanded", "false"); }
  menu.addEventListener("click", function (e) {
    e.stopPropagation();
    var open = !document.body.classList.contains("nav-open");
    document.body.classList.toggle("nav-open", open);
    menu.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.addEventListener("click", function (e) {
    if (!document.body.classList.contains("nav-open")) return;
    var side = e.target.closest(".sidebar");
    if (!side || e.target.closest("a")) closeNav();
  });

  /* ---------------- Search ---------------- */
  var search = $("#doc-search");
  var status = $("#search-status");
  var docText = null;
  function indexDocs() {
    docText = {};
    $$(".reports-tab .guide-main > [data-doc]").forEach(function (d) {
      docText[d.getAttribute("data-doc")] = (d.textContent || "").toLowerCase();
    });
  }
  function runSearch() {
    var q = search.value.trim().toLowerCase();
    if (!docText) indexDocs();
    var shown = 0, total = 0;
    $$(".doc-nav li").forEach(function (li) {
      var a = li.querySelector("a");
      var key = a.getAttribute("data-doc-link");
      var old = a.querySelector("mark");
      if (old) old.remove();
      total++;
      if (!q) { li.classList.remove("no-match"); return; }
      var text = docText[key] || "";
      var hits = 0, at = text.indexOf(q);
      while (at !== -1 && hits < 999) { hits++; at = text.indexOf(q, at + q.length); }
      var match = hits > 0 || a.textContent.toLowerCase().indexOf(q) !== -1;
      li.classList.toggle("no-match", !match);
      if (match) {
        shown++;
        if (hits) { var m = document.createElement("mark"); m.textContent = "(" + hits + ")"; a.appendChild(m); }
      }
    });
    $$(".nav-group").forEach(function (g) {
      var visible = g.querySelectorAll("li:not(.no-match)").length;
      g.hidden = !!q && visible === 0;
      if (q && visible) g.open = true;
    });
    status.textContent = q ? shown + " of " + total + " documents match" : "";
  }
  search.addEventListener("input", runSearch);
  search.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      var first = $(".doc-nav li:not(.no-match) a");
      if (first) { location.hash = first.getAttribute("href").slice(1); }
    }
  });

  /* ---------------- Copy buttons ---------------- */
  document.addEventListener("click", function (e) {
    var btn = e.target.closest(".copy");
    if (!btn) return;
    var pre = btn.parentNode.querySelector("pre").cloneNode(true);
    $$(".linenos", pre).forEach(function (n) { n.remove(); });
    var text = pre.textContent.replace(/\n$/, "");
    var done = function () {
      btn.textContent = "Copied"; btn.classList.add("done");
      setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("done"); }, 1400);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
    } else { fallbackCopy(text); done(); }
  });
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (err) { /* nothing more to try */ }
    ta.remove();
  }

  /* ---------------- Showcase deck ---------------- */
  var deck = (function () {
    var el = $("#deck"), stage = $("#stage"), wrap = $(".stage-wrap", el);
    var slides = $$(".slide", stage);
    var jump = $("#slide-jump"), fill = $("#progress-fill"), fsBtn = $("#fullscreen");
    var i = 1;

    function layout() {
      if (panels.showcase.hidden) return;
      var w = wrap.clientWidth, h = wrap.clientHeight;
      var reflow = w < 720 || h < 300;
      el.classList.toggle("reflow", reflow);
      if (!reflow) stage.style.setProperty("--scale", Math.min(w / 1280, h / 720).toFixed(4));
    }
    function go(n, fromRoute) {
      n = Math.max(1, Math.min(slides.length, n));
      var changed = n !== i;
      i = n;
      slides.forEach(function (s, k) {
        var on = k + 1 === n;
        s.hidden = !on;
        if (on && changed) { s.classList.remove("enter"); void s.offsetWidth; s.classList.add("enter"); }
      });
      jump.value = String(n);
      fill.style.width = (n / slides.length * 100) + "%";
      if (el.classList.contains("reflow")) wrap.scrollTop = 0;
      if (!fromRoute && history.replaceState) history.replaceState(history.state, "", "#slide-" + n);
      lastHash.showcase = "slide-" + n;
    }
    function isFullscreen() {
      return !!(document.fullscreenElement || document.webkitFullscreenElement) || el.classList.contains("pseudo-fs");
    }
    function enterFullscreen() {
      var req = el.requestFullscreen || el.webkitRequestFullscreen;
      if (req) {
        var p = req.call(el);
        if (p && p.catch) p.catch(function () { el.classList.add("pseudo-fs"); sync(); });
      } else { el.classList.add("pseudo-fs"); }
      sync();
    }
    function exitFullscreen() {
      if (el.classList.contains("pseudo-fs")) { el.classList.remove("pseudo-fs"); sync(); return; }
      var ex = document.exitFullscreen || document.webkitExitFullscreen;
      if (ex && (document.fullscreenElement || document.webkitFullscreenElement)) ex.call(document);
    }
    function sync() {
      var on = isFullscreen();
      fsBtn.textContent = on ? "✕ Exit" : "⛶ Present";
      fsBtn.setAttribute("aria-label", on ? "Exit full screen" : "Enter full screen");
      requestAnimationFrame(layout);
    }

    $("#prev-slide").addEventListener("click", function () { go(i - 1); });
    $("#next-slide").addEventListener("click", function () { go(i + 1); });
    jump.addEventListener("change", function () { go(parseInt(jump.value, 10)); jump.blur(); });
    fsBtn.addEventListener("click", function () { if (isFullscreen()) exitFullscreen(); else enterFullscreen(); });
    document.addEventListener("fullscreenchange", sync);
    document.addEventListener("webkitfullscreenchange", sync);
    window.addEventListener("resize", layout);
    if (window.ResizeObserver) new ResizeObserver(layout).observe(wrap);

    document.addEventListener("keydown", function (e) {
      if (current !== "showcase" || e.altKey || e.ctrlKey || e.metaKey) return;
      var tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      var k = e.key;
      if (tag === "button" && (k === " " || k === "Enter")) return;  // the button's own click handles it
      if (k === "ArrowRight" || k === "PageDown" || k === " " || k === "Spacebar") { e.preventDefault(); go(i + 1); }
      else if (k === "ArrowLeft" || k === "PageUp") { e.preventDefault(); go(i - 1); }
      else if (k === "Home") { e.preventDefault(); go(1); }
      else if (k === "End") { e.preventDefault(); go(slides.length); }
      else if (k === "f" || k === "F") { e.preventDefault(); if (isFullscreen()) exitFullscreen(); else enterFullscreen(); }
      else if (k === "Escape" && el.classList.contains("pseudo-fs")) { exitFullscreen(); }
    });

    var x0 = null, y0 = null;
    wrap.addEventListener("touchstart", function (e) { x0 = e.touches[0].clientX; y0 = e.touches[0].clientY; }, { passive: true });
    wrap.addEventListener("touchend", function (e) {
      if (x0 === null) return;
      var dx = e.changedTouches[0].clientX - x0, dy = e.changedTouches[0].clientY - y0;
      if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) go(i + (dx < 0 ? 1 : -1));
      x0 = y0 = null;
    }, { passive: true });

    return { go: go, layout: layout, index: function () { return i; }, isFullscreen: isFullscreen,
             exitFullscreen: exitFullscreen, count: slides.length };
  })();

  stamp();
  route();
})();
