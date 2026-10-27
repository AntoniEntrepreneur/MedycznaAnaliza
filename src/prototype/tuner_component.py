"""Streamlit component for the prototype tuner panel (TUNE.md)."""

import json
import streamlit as st

KNOBS = [
    {
        "name": "card-radius",
        "label": "Card Corner Radius",
        "type": "slider",
        "steps": [
            {"label": "none (0px)", "value": "0px"},
            {"label": "sm (6px)", "value": "6px"},
            {"label": "md (10px)", "value": "10px"},
            {"label": "lg (16px)", "value": "16px"},
            {"label": "full (24px)", "value": "24px"},
        ],
        "default": "full (24px)",
    },
    {
        "name": "card-bg",
        "label": "Card Background",
        "type": "color",
        "swatches": [
            {"label": "ice-sky", "value": "#caf0fe"},
            {"label": "slate-50", "value": "#f8fafc"},
            {"label": "white", "value": "#ffffff"},
            {"label": "sky-50", "value": "#f0f9ff"},
            {"label": "zinc-50", "value": "#fafafa"},
        ],
        "default": "ice-sky",
    },
    {
        "name": "accent-color",
        "label": "Accent Color",
        "type": "color",
        "swatches": [
            {"label": "blue-600", "value": "#2563eb"},
            {"label": "indigo-600", "value": "#4f46e5"},
            {"label": "teal-600", "value": "#0d9488"},
            {"label": "rose-600", "value": "#e11d48"},
        ],
        "default": "blue-600",
    },
    {
        "name": "card-shadow",
        "label": "Card Elevation Shadow",
        "type": "select",
        "options": [
            {"label": "flat", "value": "none"},
            {"label": "subtle", "value": "0 1px 3px rgba(0,0,0,0.06)"},
            {"label": "elevated", "value": "0 4px 12px rgba(0,0,0,0.08)"},
            {"label": "crisp-border", "value": "0 0 0 1px #e2e8f0"},
        ],
        "default": "elevated",
    },
    {
        "name": "container-padding",
        "label": "Container Padding",
        "type": "slider",
        "steps": [
            {"label": "compact (12px)", "value": "12px"},
            {"label": "comfortable (18px)", "value": "18px"},
            {"label": "spacious (26px)", "value": "26px"},
        ],
        "default": "comfortable (18px)",
    },
    {
        "name": "show-pop-badge",
        "label": "Show Population Risk Badge",
        "type": "toggle",
        "on": "inline-block",
        "off": "none",
        "default": True,
    },
    {
        "name": "dial-size",
        "label": "Risk Dial Scale",
        "type": "slider",
        "steps": [
            {"label": "compact (240px)", "value": "240px"},
            {"label": "standard (320px)", "value": "320px"},
            {"label": "prominent (400px)", "value": "400px"},
        ],
        "default": "standard (320px)",
    },
    {
        "name": "ehr-font",
        "label": "EHR Note Typography",
        "type": "select",
        "options": [
            {"label": "monospace", "value": "ui-monospace, SFMono-Regular, Menlo, monospace"},
            {"label": "system-sans", "value": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"},
        ],
        "default": "monospace",
    },
]


def render_tuner_panel():
    """Mount the floating tuner panel into the parent Streamlit window."""
    knobs_json = json.dumps(KNOBS)

    html_code = f"""
    <script>
    (function() {{
      const doc = window.parent.document;
      const target = doc.documentElement;
      const title = "Winner on Stroke Risk Calculator";
      const storageKey = "prototype-tuner:" + title;
      const knobs = {knobs_json};

      let saved = {{}};
      try {{ saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}"); }} catch(e) {{}}
      const state = {{}};

      const find = (list, label) => list.find((x) => x.label === label);
      const persist = () => {{ try {{ localStorage.setItem(storageKey, JSON.stringify(state)); }} catch(e) {{}} }};

      function cssValue(knob, v) {{
        if (knob.type === "slider") return find(knob.steps, v)?.value;
        if (knob.type === "color") return find(knob.swatches, v)?.value ?? v;
        if (knob.type === "toggle") return v ? knob.on : knob.off;
        if (knob.type === "select") return find(knob.options, v)?.value;
      }}

      function apply(knob) {{
        const v = state[knob.name];
        target.style.setProperty("--tune-" + knob.name, cssValue(knob, v));
      }}

      // Avoid duplicate panel
      const existing = doc.getElementById("prototype-tuner-panel");
      if (existing) existing.remove();

      const panel = doc.createElement("div");
      panel.id = "prototype-tuner-panel";
      panel.style.cssText =
        "position:fixed;top:60px;right:18px;z-index:9999999;width:275px;max-height:82vh;overflow-y:auto;" +
        "background:#0f172a;color:#f8fafc;font:12px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;" +
        "border-radius:12px;padding:12px;box-shadow:0 12px 32px rgba(0,0,0,.5);border:1px solid #334155;";

      const head = doc.createElement("div");
      head.style.cssText = "display:flex;gap:6px;align-items:center;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #1e293b;";
      head.innerHTML = `<span style="font-weight:700;flex:1;font-size:13px;letter-spacing:0.5px">🎛️ UI Tuner</span>`;

      const btn = (label, onClick) => {{
        const b = doc.createElement("button");
        b.textContent = label;
        b.style.cssText = "background:#1e293b;color:#f1f5f9;border:1px solid #475569;border-radius:6px;padding:3px 7px;cursor:pointer;font:inherit;font-size:11px;";
        b.onclick = onClick;
        return b;
      }};

      const body = doc.createElement("div");

      function changes() {{
        const show = (k, v) => (k.type === "toggle" ? (v ? "on" : "off") : v);
        const lines = knobs
          .filter((k) => state[k.name] !== k.default)
          .map((k) => `  ${{k.label}}: ${{show(k, k.default)}} → ${{show(k, state[k.name])}}`);
        return [`Tuned: ${{title}}`, ...(lines.length ? lines : ["  (no changes)"])].join("\\n");
      }}

      head.append(
        btn("Copy changes", () => {{
          const text = changes();
          navigator.clipboard.writeText(text);
          alert("Tuning copied to clipboard! Paste it into the chat:\\n\\n" + text);
        }}),
        btn("Reset", () => knobs.forEach(reset)),
        btn("–", () => {{ body.hidden = !body.hidden; }})
      );
      panel.append(head, body);

      const rows = {{}};
      function render(knob) {{
        let row = rows[knob.name];
        if (!row) {{
          row = doc.createElement("div");
          row.style.cssText = "margin-bottom:10px;";
          rows[knob.name] = row;
          body.append(row);
        }}
        row.innerHTML = "";

        const top = doc.createElement("div");
        top.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;";
        const changed = state[knob.name] !== knob.default;
        top.innerHTML = `<span style="font-weight:600;color:${{changed ? '#38bdf8' : '#cbd5e1'}}">${{knob.label}}${{changed ? " •" : ""}}</span>`;
        top.append(btn("↺", () => reset(knob)));
        row.append(top);

        const set = (v) => {{ state[knob.name] = v; apply(knob); persist(); render(knob); }};
        const v = state[knob.name];

        if (knob.type === "slider") {{
          const i = knob.steps.findIndex((s) => s.label === v);
          const input = Object.assign(doc.createElement("input"), {{
            type: "range",
            min: 0,
            max: knob.steps.length - 1,
            step: 1,
            value: i >= 0 ? i : 0,
          }});
          input.style.cssText = "width:100%;accent-color:#38bdf8;cursor:pointer;";
          input.oninput = () => set(knob.steps[input.value].label);
          const valLabel = doc.createElement("div");
          valLabel.style.cssText = "font-size:10px;color:#94a3b8;margin-top:2px;";
          valLabel.textContent = v;
          row.append(input, valLabel);
        }} else if (knob.type === "color") {{
          const wrap = doc.createElement("div");
          wrap.style.cssText = "display:flex;flex-wrap:wrap;gap:5px;align-items:center;margin-top:2px;";
          for (const s of knob.swatches) {{
            const sw = doc.createElement("button");
            sw.title = s.label;
            sw.style.cssText = `width:20px;height:20px;border-radius:4px;cursor:pointer;background:${{s.value}};border:2px solid ${{s.label === v ? '#38bdf8' : '#475569'}}`;
            sw.onclick = () => set(s.label);
            wrap.append(sw);
          }}
          const free = Object.assign(doc.createElement("input"), {{ type: "color", value: cssValue(knob, v) }});
          free.style.cssText = "width:24px;height:24px;border:0;padding:0;cursor:pointer;border-radius:4px;background:transparent;";
          free.oninput = () => set(free.value);
          const valLabel = doc.createElement("span");
          valLabel.style.cssText = "font-size:10px;color:#94a3b8;margin-left:4px;";
          valLabel.textContent = v;
          wrap.append(free, valLabel);
          row.append(wrap);
        }} else if (knob.type === "toggle") {{
          const wrap = doc.createElement("label");
          wrap.style.cssText = "display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11px;color:#94a3b8;";
          const input = Object.assign(doc.createElement("input"), {{ type: "checkbox", checked: v }});
          input.style.accentColor = "#38bdf8";
          input.onchange = () => set(input.checked);
          wrap.append(input, doc.createTextNode(v ? " Enabled" : " Disabled"));
          row.append(wrap);
        }} else if (knob.type === "select") {{
          const sel = doc.createElement("select");
          sel.style.cssText = "width:100%;background:#1e293b;color:#f8fafc;border:1px solid #475569;border-radius:5px;padding:3px;font-size:11px;";
          for (const o of knob.options) {{
            sel.append(new Option(o.label, o.label, false, o.label === v));
          }}
          sel.onchange = () => set(sel.value);
          row.append(sel);
        }}
      }}

      function reset(knob) {{
        state[knob.name] = knob.default;
        apply(knob);
        persist();
        render(knob);
      }}

      for (const knob of knobs) {{
        state[knob.name] = knob.name in saved ? saved[knob.name] : knob.default;
        apply(knob);
        render(knob);
      }}

      doc.body.append(panel);
    }})();
    </script>
    """
    st.components.v1.html(html_code, height=0, width=0)
