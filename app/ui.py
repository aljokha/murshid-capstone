"""Gradio interface for Murshid.

Ask anything you like — there are no canned questions. Every message goes
through the same `murshid.invoke()` used in the notebook, so the routing, the
two separate document stores, the memory, the graph trace and the approval
queue are all the real ones — nothing here is a mock-up.

    python run.py
"""

from __future__ import annotations

import html
import os
import uuid

import gradio as gr
from langgraph.types import Command

from app import murshid as core

# Gradio 6 moved `css` and `theme` from Blocks() to launch(), and dropped the
# Chatbot `type=` argument. Detect the version so this runs unchanged on both.
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_IS_G6 = GRADIO_MAJOR >= 6

VIEW_META = {
    "ask": ("01", "Ask Murshid", "One question, Arabic or English — the router decides who owns the answer"),
    "approvals": ("02", "Advisor queue", "Runs paused before anything irreversible is filed"),
    "memory": ("03", "Memory", "Long-term facts keyed by student, not by conversation"),
}

# color/label metadata per route, keyed to the CSS custom properties below
ROUTE_META = {
    "academic": {"color": "var(--mur-academic)", "on": "var(--mur-acadink)",
                "initial": "AA", "label": "academic · registrar"},
    "campus":   {"color": "var(--mur-campus)",   "on": "var(--mur-campink)",
                "initial": "CS", "label": "campus · student affairs"},
    "both":     {"color": "var(--mur-ink)",       "on": "var(--mur-card)",
                "initial": "A+C", "label": "both · merged context"},
    "action":   {"color": "var(--mur-action)",   "on": "var(--mur-actink)",
                "initial": "AC", "label": "action · needs a human"},
}

# The four destinations, exactly as the router actually classifies them —
# one deliberately Arabic-only, so the empty state proves the point live.
SEEDS = [
    {"q": "What GPA do I need to stay off academic probation?", "route": "academic",
     "why": "A regulation the Registrar owns. One store, three chunks."},
    {"q": "لا أستطيع الدخول إلى بوابة الطالب", "route": "campus",
     "why": "Not one English keyword in it — the classifier still lands it correctly."},
    {"q": "How do I appeal a grade, and where is the IT helpdesk?", "route": "both",
     "why": "Spans both offices. Two searches run concurrently, contexts merge."},
    {"q": "I want to withdraw from STAT301", "route": "action",
     "why": "Not a question — a request to file. Intent, not vocabulary."},
]

LOGO_SVG = """
<svg width="30" height="30" viewBox="0 0 34 34" fill="none">
  <circle cx="17" cy="17" r="13.5" stroke="var(--mur-rail-mut)" stroke-width="1"/>
  <path d="M17 2 L18.7 5.6 L15.3 5.6 Z" fill="var(--mur-gold)"/>
  <path d="M32 17 L28.4 18.7 L28.4 15.3 Z" fill="var(--mur-rail-mut)"/>
  <path d="M17 32 L15.3 28.4 L18.7 28.4 Z" fill="var(--mur-rail-mut)"/>
  <path d="M2 17 L5.6 15.3 L5.6 18.7 Z" fill="var(--mur-rail-mut)"/>
  <path d="M12 21.3 C11.7 16.6 15.3 11.4 22.6 7.8 L17.4 15.6 L22 17.1 Z" fill="var(--mur-gold)"/>
  <path d="M10.9 22.2 C9.2 20.1 9.3 17.3 12.1 15.2" stroke="var(--mur-rail-mut)"
        stroke-width="1.2" fill="none" stroke-linecap="round"/>
  <circle cx="10.5" cy="19.7" r="0.9" fill="var(--mur-rail-ink)"/>
</svg>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;1,6..72,300;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');

:root {
    --mur-paper: #efe9dc;
    --mur-card: #f7f3ea;
    --mur-sunk: #e5ddcc;
    --mur-ink: #12100d;
    --mur-ink2: #57503f;
    --mur-ink3: #8a8271;
    --mur-line: rgba(18,16,13,.14);
    --mur-line2: rgba(18,16,13,.08);
    --mur-chipbg: rgba(18,16,13,.06);
    --mur-gold: #c08a2e;
    --mur-academic: #1c6a53;
    --mur-campus: #2f4a9c;
    --mur-action: #a8590f;
    --mur-acadink: #f7f3ea;
    --mur-campink: #f7f3ea;
    --mur-actink: #f7f3ea;
    --mur-academic-bg: rgba(28,106,83,.11);
    --mur-campus-bg: rgba(47,74,156,.11);
    --mur-action-bg: rgba(168,89,15,.11);
    --mur-rail: #12100d;
    --mur-rail-ink: #efe9dc;
    --mur-rail-mut: rgba(239,233,220,.46);
    --mur-rail-hover: rgba(239,233,220,.09);
    --mur-disp: 'Newsreader', Georgia, serif;
}

@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --mur-paper: #12100d; --mur-card: #1b1814; --mur-sunk: #0c0b09;
        --mur-ink: #f0ebe0; --mur-ink2: #a49b88; --mur-ink3: #726a5b;
        --mur-line: rgba(240,235,224,.16); --mur-line2: rgba(240,235,224,.08);
        --mur-chipbg: rgba(240,235,224,.08);
        --mur-gold: #e0aa4c; --mur-academic: #3fb494; --mur-campus: #7e9cf0; --mur-action: #e09242;
        --mur-acadink: #08110e; --mur-campink: #080b16; --mur-actink: #160c03;
        --mur-academic-bg: rgba(63,180,148,.14); --mur-campus-bg: rgba(126,156,240,.14); --mur-action-bg: rgba(224,146,66,.14);
        --mur-rail: #080705; --mur-rail-mut: rgba(240,235,224,.44); --mur-rail-hover: rgba(240,235,224,.08);
    }
}
.dark, :root.dark {
    --mur-paper: #12100d; --mur-card: #1b1814; --mur-sunk: #0c0b09;
    --mur-ink: #f0ebe0; --mur-ink2: #a49b88; --mur-ink3: #726a5b;
    --mur-line: rgba(240,235,224,.16); --mur-line2: rgba(240,235,224,.08);
    --mur-chipbg: rgba(240,235,224,.08);
    --mur-gold: #e0aa4c; --mur-academic: #3fb494; --mur-campus: #7e9cf0; --mur-action: #e09242;
    --mur-acadink: #08110e; --mur-campink: #080b16; --mur-actink: #160c03;
    --mur-academic-bg: rgba(63,180,148,.14); --mur-campus-bg: rgba(126,156,240,.14); --mur-action-bg: rgba(224,146,66,.14);
    --mur-rail: #080705; --mur-rail-mut: rgba(240,235,224,.44); --mur-rail-hover: rgba(240,235,224,.08);
}

.gradio-container {max-width: 1680px !important; width: 97% !important; margin: 0 auto !important; background: var(--mur-paper) !important}
body, .gradio-container { font-family: 'IBM Plex Sans Arabic', ui-sans-serif, sans-serif !important; color: var(--mur-ink); }
footer {display: none !important}
* { unicode-bidi: plaintext; }
.mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }

/* ---------- shell ---------- */
.mur-shell { gap: 0 !important; border: 1px solid var(--mur-ink); border-radius: 2px; overflow: hidden; margin-top: 8px; }

/* ---------- rail ---------- */
.mur-rail { background: var(--mur-rail) !important; padding: 0 !important; position: relative;
    display: flex !important; flex-direction: column !important; }
.rail-head { padding: 22px 18px 18px; border-bottom: 1px solid var(--mur-rail-hover); }
.rail-brand { display: flex; align-items: center; gap: 10px; }
.rail-word-ar { font-size: 26px; font-weight: 600; color: var(--mur-rail-ink); line-height: 1; }
.rail-word-en { font-family: var(--mur-disp); font-size: 15px; font-style: italic; color: var(--mur-gold); margin-top: 6px; }
.rail-uni { font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; letter-spacing: .16em; text-transform: uppercase; color: var(--mur-rail-mut); margin-top: 10px; }

.nav-btn {
    display: flex !important; align-items: center; gap: 10px; width: 100% !important;
    background: none !important; border: none !important; border-radius: 0 !important;
    padding: 10px 18px !important; font-size: 13px !important; color: var(--mur-rail-mut) !important;
    text-align: start !important; justify-content: flex-start !important; box-shadow: none !important;
}
.nav-btn-active { background: var(--mur-rail-hover) !important; color: var(--mur-rail-ink) !important; font-weight: 600 !important;
    border-inline-start: 3px solid var(--mur-gold) !important; }
.nav-wrap { border-bottom: 1px solid var(--mur-rail-hover); padding: 6px 0; }

.rail-stores { padding: 16px 18px; }
.rail-stores-label { font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; letter-spacing: .16em; text-transform: uppercase; color: var(--mur-rail-mut); margin-bottom: 11px; }
.rail-store-row { margin-bottom: 10px; }
.rail-store-top { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--mur-rail-ink); }
.rail-store-bar { display: flex; gap: 2px; margin-top: 5px; }
.rail-store-bar span { flex: 1; height: 4px; }

.rail-foot { border-top: 1px solid var(--mur-rail-hover); padding: 14px 18px 16px; }
.rail-foot-label { font-family: 'IBM Plex Mono', monospace; font-size: 8px; letter-spacing: .14em; text-transform: uppercase; color: var(--mur-rail-mut); margin-bottom: 8px; }
.rail-student { display: flex; align-items: center; gap: 10px; }
.rail-student-chip { width: 30px; height: 30px; background: var(--mur-gold); color: var(--mur-rail); display: flex;
    align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex: none; }
.rail-student-name { font-size: 12px; color: var(--mur-rail-ink); }
.rail-student-id { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: var(--mur-rail-mut); margin-top: 1px; }
.mur-rail input, .mur-rail textarea {
    background: var(--mur-rail-hover) !important; border: 1px solid var(--mur-rail-hover) !important;
    color: var(--mur-rail-ink) !important; font-size: 12.5px !important;
}
.mur-rail label span { color: var(--mur-rail-mut) !important; font-size: 10px !important; }
.rail-newthread button {
    background: none !important; border: 1px solid var(--mur-rail-hover) !important; color: var(--mur-rail-mut) !important;
    font-size: 11.5px !important; box-shadow: none !important;
}

/* ---------- main header ---------- */
.mur-main { padding: 0 !important; background: var(--mur-paper) !important; }
.view-header { display: flex; align-items: baseline; justify-content: space-between; gap: 18px;
    padding: 20px 26px; border-bottom: 1px solid var(--mur-ink); }
.view-header-left { display: flex; align-items: baseline; gap: 14px; min-width: 0; }
.view-num { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .16em; color: var(--mur-gold); }
.view-title { font-family: var(--mur-disp); font-weight: 400; font-size: 21px; color: var(--mur-ink); }
.view-sub { font-size: 11.5px; color: var(--mur-ink2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.view-status { display: flex; align-items: center; gap: 8px; flex: none; }
.view-status-dot { width: 6px; height: 6px; background: var(--mur-academic); border-radius: 999px; }
.view-status-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--mur-ink2); }

/* ---------- hero (empty state) ---------- */
.mur-hero { padding: 40px 26px 6px; position: relative; overflow: hidden; }
.mur-hero-wm { position: absolute; inset-inline-end: -30px; top: 0; font-size: 150px; font-weight: 600;
    color: var(--mur-ink); opacity: .045; pointer-events: none; user-select: none; line-height: .8; }
.mur-hero-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: .2em; text-transform: uppercase; color: var(--mur-gold); margin-bottom: 16px; }
.mur-hero h1 { font-family: var(--mur-disp); font-weight: 400; font-size: 46px; line-height: 1.08; margin: 0 0 26px; max-width: 16ch; }
.mur-hero h1 em { font-style: italic; color: var(--mur-gold); }
.mur-hero-intro { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; border-top: 1px solid var(--mur-ink); padding-top: 18px; max-width: 1080px; }
.mur-hero-intro p { font-size: 13.5px; line-height: 1.7; color: var(--mur-ink2); margin: 0; }
.mur-hero-intro p.ar { line-height: 1.9; }
.mur-dest-head { display: flex; align-items: baseline; justify-content: space-between; margin: 24px 0 10px; }
.mur-dest-head span:first-child { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: .18em; text-transform: uppercase; color: var(--mur-ink3); }
.mur-dest-head span:last-child { font-size: 11px; color: var(--mur-ink3); }

.tile-row { gap: 1px !important; background: var(--mur-line); padding: 1px; margin: 0 26px 18px; }
.tile-col {
    display: flex !important; flex-direction: column; gap: 9px !important;
    background: var(--mur-paper) !important; padding: 16px !important; min-height: 128px;
}
.tile-col:hover { background: var(--mur-card) !important; }
.tile-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.tile-num { font-size: 10px; letter-spacing: .14em; }
.tile-why { font-size: 11.5px; color: var(--mur-ink2); line-height: 1.5; }
.tile-q-btn button {
    background: none !important; border: none !important; border-radius: 0 !important; padding: 0 !important;
    text-align: start !important; white-space: normal !important; box-shadow: none !important;
    color: var(--mur-ink) !important; font-family: var(--mur-disp) !important;
    font-size: 17px !important; line-height: 1.3 !important; font-weight: 400 !important; min-height: 0 !important;
}
.tile-q-btn button:hover { color: var(--mur-gold) !important; }

/* ---------- transcript ---------- */
.transcript { padding: 8px 26px 4px; display: flex; flex-direction: column; gap: 26px; min-height: 40px; }
.msg-user { font-family: var(--mur-disp); font-weight: 400; font-size: 25px; line-height: 1.25;
    padding-inline-start: 16px; border-inline-start: 3px solid var(--mur-gold); }
.msg-agent { display: grid; grid-template-columns: 38px 1fr; gap: 0; }
.msg-avatar-col { display: flex; flex-direction: column; align-items: stretch; }
.msg-avatar { height: 38px; display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; font-weight: 600; }
.msg-rule { flex: 1; width: 2px; margin-inline-start: 18px; opacity: .35; min-height: 8px; }
.msg-body { min-width: 0; padding: 0 0 4px; padding-inline-start: 16px; }
.msg-chips { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 13px; }
.msg-route-chip { font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; padding: 4px 8px; }
.msg-tool-chip { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--mur-ink2); border: 1px solid var(--mur-line); padding: 3px 7px; }
.msg-warn-chip { font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: var(--mur-actink); background: var(--mur-action); padding: 4px 8px; }
.msg-bi { display: grid; grid-template-columns: 1.05fr 1fr; gap: 0; }
.msg-bi-ar { padding-inline-end: 22px; }
.msg-bi-label { font-size: 10.5px; color: var(--mur-ink3); margin-bottom: 7px; }
.msg-bi-label.mono { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: .16em; text-transform: uppercase; }
.msg-bi-ar-text { font-size: 15px; line-height: 1.85; white-space: pre-line; }
.msg-bi-en { border-inline-start: 1px solid var(--mur-line); padding-inline-start: 22px; }
.msg-bi-en-text { font-size: 13.5px; line-height: 1.7; color: var(--mur-ink2); white-space: pre-line; }
.msg-solo { font-size: 14.5px; line-height: 1.8; white-space: pre-line; }
.msg-sources { margin-top: 16px; border-top: 1px solid var(--mur-ink); padding-top: 8px; }
.msg-sources-label { font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; letter-spacing: .18em; text-transform: uppercase; color: var(--mur-ink3); margin-bottom: 4px; }
.msg-source-row { display: flex; align-items: baseline; gap: 10px; padding: 6px 0; border-top: 1px solid var(--mur-line2); }
.msg-source-ref { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 600; flex: none; }
.msg-source-title { font-size: 12px; color: var(--mur-ink2); }
.msg-action-card { margin-top: 16px; border: 1px solid var(--mur-action); background: var(--mur-action-bg); }
.msg-action-head { display: flex; align-items: center; gap: 8px; padding: 9px 14px; background: var(--mur-action); }
.msg-action-dot { width: 6px; height: 6px; border-radius: 999px; background: var(--mur-actink); }
.msg-action-head span:last-child { font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--mur-actink); }
.msg-action-body { padding: 14px; }
.msg-action-title { font-family: var(--mur-disp); font-size: 19px; margin-bottom: 10px; }
.msg-action-grid { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; font-size: 12px; }
.msg-action-grid span:nth-child(odd) { color: var(--mur-ink3); }

.mur-thinking { display: flex; align-items: center; gap: 12px; padding: 0 26px; }
.mur-thinking-box { width: 38px; height: 38px; background: var(--mur-chipbg); display: flex; align-items: center; justify-content: center; }
.mur-thinking-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--mur-ink2); }

/* ---------- input row ---------- */
.mur-inputrow { border-top: 1px solid var(--mur-ink); background: var(--mur-card); padding: 14px 26px 16px; }
.mur-input-box { border: 1px solid var(--mur-ink) !important; background: var(--mur-paper); }
.mur-input textarea, .mur-input input { border: none !important; background: none !important; font-size: 14px !important; box-shadow: none !important; }
.mur-ask-btn button { background: var(--mur-ink) !important; color: var(--mur-paper) !important; border: none !important; border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: .12em !important; text-transform: uppercase !important; }
.mur-hint { font-size: 10.5px; color: var(--mur-ink3); margin-top: 8px; }

/* ---------- trace pane ---------- */
.mur-trace { background: var(--mur-card) !important; border-inline-start: 1px solid var(--mur-ink) !important; padding: 0 !important; }
.trace-head { padding: 18px; border-bottom: 1px solid var(--mur-line); }
.trace-title { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: .18em; text-transform: uppercase; color: var(--mur-gold); }
.trace-sub { font-size: 11px; color: var(--mur-ink2); margin-top: 5px; line-height: 1.5; }
.trace-empty { margin: 18px; font-size: 11.5px; color: var(--mur-ink3); line-height: 1.6; padding: 14px; border: 1px dashed var(--mur-line); }
.trace-steps { padding: 14px 18px 0; }
.trace-step { display: grid; grid-template-columns: 9px 1fr; gap: 10px; }
.trace-step-rail { display: flex; flex-direction: column; align-items: center; padding-top: 4px; }
.trace-step-dot { width: 6px; height: 6px; background: var(--mur-ink3); flex: none; }
.trace-step-line { width: 1px; flex: 1; background: var(--mur-line); min-height: 10px; }
.trace-step-body { padding-bottom: 14px; min-width: 0; }
.trace-step-name { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; font-weight: 500; color: var(--mur-ink); }
.trace-step-detail { font-size: 10px; color: var(--mur-ink2); line-height: 1.5; margin-top: 3px; overflow-wrap: break-word; }
.trace-foot { margin: 0 18px; border-top: 1px solid var(--mur-ink); padding: 12px 0 18px; display: grid;
    grid-template-columns: auto 1fr; gap: 6px 12px; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: var(--mur-ink3); }
.trace-foot span:nth-child(even) { color: var(--mur-ink); }
.trace-foot .traced { color: var(--mur-academic); }

/* ---------- approvals ---------- */
.approvals-wrap { padding: 24px 26px; display: flex; flex-direction: column; gap: 16px; max-width: 900px; }
.ticket-card { background: var(--mur-card); border: 1px solid var(--mur-ink); }
.ticket-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 9px 16px; }
.ticket-status { display: flex; align-items: center; gap: 8px; }
.ticket-dot { width: 6px; height: 6px; border-radius: 999px; }
.ticket-status-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
.ticket-thread { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; opacity: .75; }
.ticket-body { padding: 18px; }
.ticket-title { font-family: var(--mur-disp); font-weight: 400; font-size: 26px; line-height: 1.1; }
.ticket-tool { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--mur-ink2); margin: 6px 0 18px; overflow-wrap: break-word; }
.ticket-grid { display: grid; grid-template-columns: auto 1fr; gap: 9px 20px; font-size: 12.5px; padding-bottom: 14px; border-bottom: 1px solid var(--mur-line); }
.ticket-grid span:nth-child(odd) { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--mur-ink3); }
.ticket-filed-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: .14em; text-transform: uppercase; color: var(--mur-ink3); margin: 14px 0 7px; }
.ticket-filed-text { font-size: 13px; line-height: 1.65; }
.ticket-receipt { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: var(--mur-ink3); margin-top: 12px; overflow-wrap: break-word; }
.ticket-note { font-size: 10.5px; color: var(--mur-ink3); padding: 12px 18px 16px; }
.approvals-empty { font-family: var(--mur-disp); font-size: 17px; line-height: 1.5; color: var(--mur-ink2); max-width: 50ch; }

.decide-panel { border: 1px solid var(--mur-ink); background: var(--mur-card); padding: 4px 4px 16px; margin-top: -4px; }
.decide-panel-head { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--mur-ink3); padding: 12px 16px 4px; }
.decide-panel textarea { border: 1px solid var(--mur-ink) !important; background: var(--mur-paper) !important; border-radius: 0 !important; }
.decide-btn-approve button { background: var(--mur-academic) !important; color: var(--mur-acadink) !important; border: none !important; border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
.decide-btn-edit button { background: var(--mur-ink) !important; color: var(--mur-paper) !important; border: none !important; border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
.decide-btn-reject button { background: var(--mur-paper) !important; color: var(--mur-ink2) !important; border: 1px solid var(--mur-ink) !important; border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: .1em !important; text-transform: uppercase !important; }

/* ---------- memory ---------- */
.memory-wrap { padding: 26px 26px 34px; display: flex; flex-direction: column; gap: 30px; max-width: 900px; }
.memory-headline { font-family: var(--mur-disp); font-weight: 400; font-size: 34px; line-height: 1.1; max-width: 18ch; }
.memory-headline em { font-style: italic; color: var(--mur-gold); }
.memory-storekey { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--mur-ink2); margin-top: 10px; }
.memory-tiles { display: grid; grid-template-columns: repeat(3,1fr); gap: 1px; background: var(--mur-line); margin-top: 18px; border: 1px solid var(--mur-ink); }
.memory-tile { background: var(--mur-card); padding: 16px; }
.memory-tile-key { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: .1em; color: var(--mur-ink3); margin-bottom: 10px; }
.memory-tile-val { font-family: var(--mur-disp); font-size: 25px; line-height: 1.1; }
.memory-tile-note { font-size: 11px; color: var(--mur-ink2); line-height: 1.55; margin-top: 9px; }
.memory-section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; border-bottom: 1px solid var(--mur-ink); padding-bottom: 9px; }
.memory-section-title { font-family: var(--mur-disp); font-weight: 400; font-size: 20px; }
.memory-section-sub { font-size: 11px; color: var(--mur-ink3); }
.memory-table { display: grid; grid-template-columns: 120px 1fr 1fr; font-size: 12px; }
.memory-table-head { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: var(--mur-ink3); padding: 10px 14px; }
.memory-table-head.gold { color: var(--mur-gold); }
.memory-row-label { padding: 11px 0; border-top: 1px solid var(--mur-line); font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: .08em; text-transform: uppercase; color: var(--mur-ink3); }
.memory-row-short { padding: 11px 14px; border-top: 1px solid var(--mur-line); color: var(--mur-ink2); }
.memory-row-long { padding: 11px 14px; border-top: 1px solid var(--mur-line); background: var(--mur-chipbg); }
.thread-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 13px 0; border-bottom: 1px solid var(--mur-line); }
.thread-n { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: var(--mur-ink3); flex: none; }
.thread-q { font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 46ch; }
.thread-id { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--mur-ink3); margin-top: 3px; }
.thread-badge { font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; padding: 3px 7px; flex: none; }
.memory-empty { font-size: 12.5px; color: var(--mur-ink3); }

/* ---------- footer ---------- */
.mur-footer { text-align: center; color: var(--mur-ink3); font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .04em; padding: 14px 0 4px; }
"""


def _theme():
    return gr.themes.Soft(
        primary_hue="emerald", secondary_hue="blue", neutral_hue="stone",
        font=[gr.themes.GoogleFont("IBM Plex Sans Arabic"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(block_radius="0px", block_shadow="none", input_radius="0px",
          button_large_radius="0px", button_small_radius="0px")


def _dir(text: str) -> str:
    return "rtl" if any("؀" <= c <= "ۿ" for c in text) else "ltr"


def _e(text) -> str:
    return html.escape(str(text if text is not None else ""))


# ─────────────────────────── rendering: header / rail ─────────────────────
def render_header(view: str) -> str:
    num, title, sub = VIEW_META[view]
    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    status = f"{core.llm.model_name if core.llm else '—'} · {'traced' if tracing_on else 'local'}"
    return (
        '<div class="view-header"><div class="view-header-left">'
        f'<span class="view-num mono">{_e(num)}</span>'
        f'<span class="view-title">{_e(title)}</span>'
        f'<span class="view-sub">{_e(sub)}</span></div>'
        '<div class="view-status"><span class="view-status-dot"></span>'
        f'<span class="view-status-label mono">{_e(status)}</span></div></div>'
    )


def render_store_bars(info: dict) -> str:
    counts = info.get("counts", {})
    a_docs = counts.get("academic", (0, 0))[0]
    c_docs = counts.get("campus", (0, 0))[0]

    def row(label, n, color):
        bars = "".join(f'<span style="background:{color}"></span>' for _ in range(min(n, 10)))
        return (f'<div class="rail-store-row"><div class="rail-store-top">'
                f'<span>{label}</span><span class="mono">{n:02d}</span></div>'
                f'<div class="rail-store-bar">{bars}</div></div>')

    return (
        '<div class="rail-stores"><div class="rail-stores-label">Two stores, kept apart</div>'
        + row("academic", a_docs, "var(--mur-academic)")
        + row("campus", c_docs, "var(--mur-campus)") + "</div>"
    )


def render_rail_head() -> str:
    return (
        '<div class="rail-head"><div class="rail-brand">' + LOGO_SVG +
        '<div><div class="rail-word-ar">مرشد</div>'
        '<div class="rail-word-en">Murshid</div></div></div>'
        '<div class="rail-uni">Al-Noor University</div></div>'
    )


# ─────────────────────────── rendering: hero (empty state) ────────────────
def render_hero() -> str:
    return (
        '<div class="mur-hero"><div class="mur-hero-wm">نور</div>'
        '<div class="mur-hero-eyebrow mono">Student services · one door</div>'
        '<h1>One question.<br>The right <em>office</em>.</h1>'
        '<div class="mur-hero-intro">'
        '<p>Ask in Arabic or English. Murshid decides whether the Registrar or Student '
        "Affairs owns your answer, searches only that office's documents, and stops "
        'for a human before filing anything it cannot undo.</p>'
        '<p class="ar" dir="rtl">اسأل بالعربية أو الإنجليزية. يحدد مُرشد الجهة المسؤولة عن '
        'سؤالك، ويبحث في وثائقها فقط، ويتوقف لمراجعة بشرية قبل تنفيذ أي طلب لا يمكن '
        'الرجوع عنه.</p></div>'
        '<div class="mur-dest-head"><span class="mono">Four destinations</span>'
        '<span>The router picks one. Try any of them.</span></div></div>'
    )


# ─────────────────────────── rendering: transcript ─────────────────────────
def _msg_user_html(m: dict) -> str:
    return f'<div class="msg-user" dir="{m["dir"]}">{_e(m["text"])}</div>'


def _msg_agent_html(m: dict) -> str:
    meta = ROUTE_META.get(m["route"], ROUTE_META["academic"])
    chips = (f'<span class="msg-route-chip" style="color:{meta["on"]};background:{meta["color"]}">'
             f'{_e(meta["label"])}</span>')
    for t in m.get("tools", []):
        chips += f'<span class="msg-tool-chip">{_e(t)}</span>'
    if m.get("warn"):
        chips += '<span class="msg-warn-chip">confidence=low → escalated</span>'

    if m.get("two"):
        content = (
            '<div class="msg-bi"><div class="msg-bi-ar" dir="rtl">'
            '<div class="msg-bi-label">العربية</div>'
            f'<div class="msg-bi-ar-text">{_e(m["ar"])}</div></div>'
            '<div class="msg-bi-en" dir="ltr"><div class="msg-bi-label mono">English</div>'
            f'<div class="msg-bi-en-text">{_e(m["en"])}</div></div></div>'
        )
    else:
        content = f'<div class="msg-solo" dir="{m.get("dir", "ltr")}">{_e(m.get("solo", ""))}</div>'

    sources = ""
    if m.get("sources"):
        rows = "".join(
            f'<div class="msg-source-row"><span class="msg-source-ref" '
            f'style="color:{ROUTE_META.get(s["collection"], ROUTE_META["academic"])["color"]}">'
            f'{_e(s["source"])}</span></div>' for s in m["sources"])
        sources = (f'<div class="msg-sources"><div class="msg-sources-label">Retrieved · k=3</div>{rows}</div>')

    action = ""
    if m.get("is_action"):
        action = (
            '<div class="msg-action-card">'
            '<div class="msg-action-head"><span class="msg-action-dot"></span>'
            '<span>interrupt() · paused</span></div>'
            f'<div class="msg-action-body"><div class="msg-action-title">{_e(m["act_title"])}</div>'
            '<div class="msg-action-grid">'
            f'<span>Course</span><span class="mono">{_e(m["act_course"])}</span>'
            f'<span>Policy</span><span>{_e(m["act_policy"])}</span>'
            f'<span>Reversible</span><span style="color:var(--mur-action);font-weight:500">No — filing is final</span>'
            '</div></div></div>'
        )

    return (
        f'<div class="msg-agent"><div class="msg-avatar-col">'
        f'<div class="msg-avatar" style="background:{meta["color"]};color:{meta["on"]}">{_e(meta["initial"])}</div>'
        f'<div class="msg-rule" style="background:{meta["color"]}"></div></div>'
        f'<div class="msg-body"><div class="msg-chips">{chips}</div>{content}{sources}{action}</div></div>'
    )


def render_transcript(msgs: list[dict]) -> str:
    if not msgs:
        return ""
    parts = [_msg_user_html(m) if m["role"] == "user" else _msg_agent_html(m) for m in msgs]
    return f'<div class="transcript">{"".join(parts)}</div>'


# ─────────────────────────── rendering: trace panel ────────────────────────
def render_trace(steps: list[dict] | None, thread_id: str, latency) -> str:
    head = ('<div class="trace-head"><div class="trace-title mono">Graph trace</div>'
            '<div class="trace-sub">Every box in the entrypoint, in the order it ran.</div></div>')
    if not steps:
        return head + ('<div class="trace-empty">Ask something and the router, the retrieval '
                       'and the memory writes appear here.</div>')
    rows = "".join(
        f'<div class="trace-step"><div class="trace-step-rail"><span class="trace-step-dot"></span>'
        f'<span class="trace-step-line"></span></div>'
        f'<div class="trace-step-body"><div class="trace-step-name">{_e(s["name"])}</div>'
        f'<div class="trace-step-detail">{_e(s["detail"])}</div></div></div>' for s in steps)
    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    foot = (
        '<div class="trace-foot"><span>thread</span><span>' + _e(thread_id) + '</span>'
        '<span>latency</span><span>' + _e(f"{latency}s" if latency is not None else "—") + '</span>'
        '<span>langsmith</span><span class="' + ("traced" if tracing_on else "") + '">'
        + ("traced" if tracing_on else "off") + '</span></div>'
    )
    return head + f'<div class="trace-steps">{rows}</div>' + foot


# ─────────────────────────── rendering: approvals ──────────────────────────
def _ticket_html(t: dict, active: bool) -> str:
    status = t["status"]
    bg = {"pending": "var(--mur-action)", "approved": "var(--mur-academic)", "rejected": "var(--mur-chipbg)"}[status]
    ink = {"pending": "var(--mur-actink)", "approved": "var(--mur-acadink)", "rejected": "var(--mur-ink2)"}[status]
    label = {"pending": "interrupt() · awaiting decision" + (" — acting on this one below" if active else " — queued"),
             "approved": "approved · filed with the registrar",
             "rejected": "rejected · nothing filed"}[status]
    title = ("Grade appeal" if t["action_type"] == "appeal" else "Course withdrawal")
    tool = (f'submit_withdrawal(student_id="{t["student_id"]}", course="{t["course_code"]}")'
            if t["action_type"] == "withdrawal" else
            f'file_appeal(student_id="{t["student_id"]}", course="{t["course_code"]}")')

    body = (
        f'<div class="ticket-body"><div class="ticket-title">{_e(title)} — {_e(t["course_code"])}</div>'
        f'<div class="ticket-tool mono">{_e(tool)}</div>'
        '<div class="ticket-grid">'
        f'<span>Student</span><span>{_e(t["student_id"])}</span>'
        f'<span>Course</span><span>{_e(t["course_code"])}</span>'
        f'<span>Stated reason</span><span>{_e(t["student_stated_reason"])}</span>'
        f'<span>Policy</span><span>{_e(t["policy_note"])}</span>'
        '</div>'
    )
    if status != "pending":
        body += (f'<div class="ticket-filed-label mono">Filed reason</div>'
                 f'<div class="ticket-filed-text">{_e(t["filed_reason"])}</div>')
        if t.get("receipt"):
            body += f'<div class="ticket-receipt">reference: {_e(t["receipt"])}</div>'
    body += '</div>'

    return (
        f'<div class="ticket-card"><div class="ticket-head" style="background:{bg}">'
        f'<div class="ticket-status"><span class="ticket-dot" style="background:{ink}"></span>'
        f'<span class="ticket-status-label" style="color:{ink}">{_e(label)}</span></div>'
        f'<span class="ticket-thread mono" style="color:{ink}">thread: {_e(t["thread_id"])}</span></div>'
        f'{body}</div>'
    )


def oldest_pending() -> dict | None:
    pending = [t for t in core.APPROVALS.values() if t["status"] == "pending"]
    return pending[0] if pending else None


def render_approvals() -> tuple[str, str]:
    tickets = list(core.APPROVALS.values())
    active = oldest_pending()
    tickets_sorted = sorted(tickets, key=lambda t: (t["status"] != "pending",))
    if not tickets_sorted:
        html_out = ('<div class="approvals-empty">A gate that can only say yes is not a gate. '
                    'Nothing is waiting on a person right now — file a withdrawal or an appeal '
                    'from Ask Murshid to see one land here.</div>')
    else:
        cards = "".join(_ticket_html(t, active is not None and t["thread_id"] == active["thread_id"])
                        for t in tickets_sorted[:8])
        note = ('<div class="ticket-note">Only the oldest pending ticket has live decision '
                'controls below — work the queue one at a time, oldest first.</div>'
                if len([t for t in tickets_sorted if t["status"] == "pending"]) > 1 else "")
        html_out = cards + note
    return html_out, (active["thread_id"] if active else "")


def render_decide_panel(active: dict | None) -> tuple:
    """Returns (visible, reason_value, header_html)."""
    if not active:
        return gr.update(visible=False), "", ""
    title = "Grade appeal" if active["action_type"] == "appeal" else "Course withdrawal"
    header = (f'<div class="decide-panel-head mono">Deciding · {_e(title)} — '
              f'{_e(active["course_code"])} · {_e(active["student_id"])}</div>')
    return gr.update(visible=True), active["student_stated_reason"], header


# ─────────────────────────── rendering: memory ──────────────────────────────
def render_memory(student_id: str) -> str:
    profile = core.load_student_profile(student_id)
    lang_label = "العربية" if profile["preferred_language"] == "ar" else \
        ("English" if profile["preferred_language"] == "en" else "—")
    tiles = [
        ("preferred_language", lang_label,
         "Inferred once, applied in every later conversation without being asked again."),
        ("questions_asked", profile["questions_asked"],
         "Counted across threads — which is how a Store differs from chat history."),
        ("last_topic", profile["last_topic"] or "—",
         "Keeps follow-ups in context after a thread has ended."),
    ]
    tiles_html = "".join(
        f'<div class="memory-tile"><div class="memory-tile-key mono">{_e(k)}</div>'
        f'<div class="memory-tile-val">{_e(v)}</div>'
        f'<div class="memory-tile-note">{_e(note)}</div></div>' for k, v, note in tiles)

    rows = [
        ("Object", "InMemorySaver", "InMemoryStore"),
        ("Key", "thread_id", '("students", student_id)'),
        ("Holds", "the in-progress run, paused interrupts",
         "preferred_language, questions_asked, last_topic, thread_history"),
        ("Lifetime", "one conversation", "across all conversations"),
    ]
    table_rows = "".join(
        f'<div class="memory-row-label">{_e(a)}</div>'
        f'<div class="memory-row-short">{_e(b)}</div>'
        f'<div class="memory-row-long">{_e(c)}</div>' for a, b, c in rows)

    threads = profile["thread_history"]
    if threads:
        thread_rows = "".join(
            f'<div class="thread-row"><div style="min-width:0;display:flex;align-items:baseline;gap:12px">'
            f'<span class="thread-n mono">#{i + 1}</span><div style="min-width:0">'
            f'<div class="thread-q" dir="{_dir(t["question"])}">{_e(t["question"])}</div>'
            f'<div class="thread-id mono">{_e(t["thread_id"])}</div></div></div>'
            f'<span class="thread-badge" style="color:{ROUTE_META.get(t["route"], ROUTE_META["academic"])["on"]};'
            f'background:{ROUTE_META.get(t["route"], ROUTE_META["academic"])["color"]}">{_e(t["route"])}</span></div>'
            for i, t in enumerate(reversed(threads)))
    else:
        thread_rows = '<div class="memory-empty">No conversations yet for this student.</div>'

    return (
        '<div class="memory-wrap">'
        f'<div><div class="memory-headline">What Murshid remembers about <em>you</em></div>'
        f'<div class="memory-storekey mono">("students", "{_e(student_id)}") — outlives every conversation</div>'
        f'<div class="memory-tiles">{tiles_html}</div></div>'
        '<div><div class="memory-section-head"><span class="memory-section-title">Two kinds of memory</span>'
        '<span class="memory-section-sub">Why an Arabic answer follows you into next week</span></div>'
        '<div class="memory-table">'
        '<div class="memory-table-head"></div>'
        '<div class="memory-table-head">Short-term</div>'
        '<div class="memory-table-head gold">Long-term</div>'
        f'{table_rows}</div></div>'
        '<div><div class="memory-section-head"><span class="memory-section-title">Your conversations</span>'
        '<span class="memory-section-sub">Separate threads, one profile</span></div>'
        f'{thread_rows}</div>'
        '</div>'
    )


# ─────────────────────────── message-building helpers ─────────────────────
def _build_agent_message(result: dict) -> dict:
    route = result["routed_to"]
    if route == "action":
        return {}  # handled separately by _interrupt_action_message
    tools = []
    if route in ("academic", "both"):
        tools.append("search_academic")
    if route in ("campus", "both"):
        tools.append("search_campus")
    ar, en = result.get("answer_ar", ""), result.get("answer_en", "")
    two = bool(ar and en)
    return {
        "role": "agent", "route": route, "tools": tools,
        "warn": result.get("confidence") == "low",
        "two": two, "ar": ar, "en": en,
        "one": not two, "solo": en or ar, "dir": "ltr",
        "sources": result.get("sources", []),
    }


def _interrupt_action_message(p: dict) -> dict:
    return {
        "role": "agent", "route": "action", "tools": ["transfer_to_action_agent", "interrupt()"],
        "warn": False, "two": False, "one": True,
        "solo": ("This is a request to file something, so it has stopped for a student advisor. "
                 "Nothing has been filed yet — see the Advisor queue."),
        "dir": "ltr", "sources": [], "is_action": True,
        "act_title": ("Grade appeal — awaiting advisor" if p.get("type") == "appeal"
                      else "Course withdrawal — awaiting advisor"),
        "act_course": p.get("course_code") or "—",
        "act_policy": core.APPROVALS.get(p.get("thread_id", ""), {}).get("policy_note", "—"),
    }


# ─────────────────────────── event handlers ────────────────────────────────
def ask(question, student_id, thread_id, msgs, pending):
    """`pending` holds a paused "which course code?" interrupt, if the last
    turn ended on one — in which case this message answers it (resumed on
    the same thread) rather than starting a new question."""
    question = (question or "").strip()
    if not question:
        return (msgs, render_transcript(msgs), gr.update(visible=len(msgs) == 0), "",
                render_trace(None, thread_id, None), gr.update(), pending)

    student_id = (student_id or "").strip() or "guest001"
    msgs = msgs + [{"role": "user", "text": question, "dir": _dir(question)}]
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        if pending and pending.get("field") == "course_code":
            result = core.murshid.invoke(Command(resume=question), cfg)
        else:
            result = core.murshid.invoke(
                {"student_id": student_id, "question": question, "thread_id": thread_id}, cfg)
    except Exception as e:
        msgs.append({"role": "agent", "route": "academic", "tools": [], "warn": True,
                    "two": False, "one": True, "dir": "ltr", "sources": [],
                    "solo": f"Something went wrong: {type(e).__name__} — {e}"})
        return (msgs, render_transcript(msgs), gr.update(visible=False), "",
                render_trace(None, thread_id, None), nav_label(), None)

    if "__interrupt__" in result:
        p = result["__interrupt__"][0].value
        if p.get("field") == "course_code":
            msgs.append({"role": "agent", "route": "academic", "tools": [], "warn": False,
                        "two": False, "one": True, "dir": "ltr", "sources": [],
                        "solo": p["message"]})
            return (msgs, render_transcript(msgs), gr.update(visible=False), "",
                    render_trace(None, thread_id, None), nav_label(), p)

        p = {**p, "thread_id": thread_id}
        msgs.append(_interrupt_action_message(p))
        return (msgs, render_transcript(msgs), gr.update(visible=False), "",
                render_trace(None, thread_id, None), nav_label(), None)

    msgs.append(_build_agent_message(result))
    trace_html = render_trace(result.get("steps"), thread_id, result.get("latency_seconds"))
    return (msgs, render_transcript(msgs), gr.update(visible=False), "",
            trace_html, nav_label(), None)


def new_thread(student_id):
    return ([], "", str(uuid.uuid4())[:8], gr.update(visible=True),
            render_trace(None, "", None), nav_label(), None)


def switch_student(new_id, thread_id):
    return new_thread(new_id)


def nav_label() -> str:
    n = sum(1 for t in core.APPROVALS.values() if t["status"] == "pending")
    return f"02 · Advisor queue · {n}" if n else "02 · Advisor queue"


NAV_ACTIVE = ["nav-btn", "nav-btn-active"]
NAV_INACTIVE = ["nav-btn"]


def go_view(view, student_id):
    updates = {v: gr.update(visible=(v == view)) for v in ("ask", "approvals", "memory")}
    nav_updates = {v: gr.update(elem_classes=NAV_ACTIVE if v == view else NAV_INACTIVE)
                  for v in ("ask", "approvals", "memory")}
    approvals_html, active_id = render_approvals()
    active = core.APPROVALS.get(active_id)
    panel_visible, reason_val, panel_head = render_decide_panel(active)
    memory_html = render_memory(student_id) if view == "memory" else gr.update()
    return (view, updates["ask"], updates["approvals"], updates["memory"],
            nav_updates["ask"], nav_updates["approvals"], nav_updates["memory"],
            render_header(view), approvals_html, active_id, panel_visible, reason_val,
            panel_head, memory_html)


def decide_ticket(kind, reason_text, note_text, active_id, thread_id, msgs, student_id):
    if not active_id:
        approvals_html, new_active = render_approvals()
        panel_visible, reason_val, panel_head = render_decide_panel(core.APPROVALS.get(new_active))
        return (approvals_html, new_active, panel_visible, reason_val, panel_head,
                msgs, render_transcript(msgs), nav_label())

    ticket = core.APPROVALS.get(active_id)
    cfg = {"configurable": {"thread_id": active_id}}
    if kind == "reject":
        payload = {"approved": False, "note": (note_text or "").strip() or "Declined by the advisor."}
    else:
        payload = {"approved": True, "course_code": ticket["course_code"] if ticket else None,
                  "edited_reason": (reason_text or "").strip()
                                    or (ticket["student_stated_reason"] if ticket else "")}

    try:
        result = core.murshid.invoke(Command(resume=payload), cfg)
    except Exception:
        result = {}

    # If this ticket belongs to the browser's own open thread, drop the
    # outcome into that conversation's transcript too.
    if active_id == thread_id and "answer" in result:
        outcome = result.get("outcome", "")
        msgs = msgs + [{"role": "agent", "route": "action", "tools": [], "warn": False,
                        "two": False, "one": True, "dir": "ltr", "sources": [],
                        "solo": result["answer"]}]
        _ = outcome  # outcome already folded into the message text above

    approvals_html, new_active = render_approvals()
    panel_visible, reason_val, panel_head = render_decide_panel(core.APPROVALS.get(new_active))
    return (approvals_html, new_active, panel_visible, reason_val, panel_head,
            msgs, render_transcript(msgs), nav_label())


def registry_view():
    if not core.REGISTRY_LOG:
        return "*Nothing has been filed yet.*"
    rows = ["| Reference | Course | Recorded reason |", "|---|---|---|"]
    rows += [f"| `{e['ref']}` | {e['course']} | {e['reason']} |" for e in core.REGISTRY_LOG]
    return "\n".join(rows)


# ─────────────────────────────── the app ───────────────────────────────────
def create_app(info: dict):
    blocks_kwargs = {"title": "مُرشد — Murshid"}
    if not _IS_G6:
        blocks_kwargs["css"] = CSS
        blocks_kwargs["theme"] = _theme()

    with gr.Blocks(**blocks_kwargs) as demo:

        view_state = gr.State("ask")
        thread_state = gr.State(str(uuid.uuid4())[:8])
        msgs_state = gr.State([])
        active_ticket_state = gr.State("")
        pending_state = gr.State(None)

        with gr.Row(elem_classes="mur-shell"):
            with gr.Column(scale=0, min_width=250, elem_classes="mur-rail"):
                gr.HTML(render_rail_head())
                with gr.Column(elem_classes="nav-wrap"):
                    nav_ask = gr.Button("01 · Ask Murshid", elem_classes=NAV_ACTIVE)
                    nav_approvals = gr.Button(nav_label(), elem_classes=NAV_INACTIVE)
                    nav_memory = gr.Button("03 · Memory", elem_classes=NAV_INACTIVE)
                gr.HTML(render_store_bars(info))
                gr.HTML('<div style="flex:1;min-height:12px"></div>')
                with gr.Column(elem_classes="rail-foot"):
                    gr.HTML('<div class="rail-foot-label mono">Student</div>')
                    sid = gr.Textbox(value="s2201", show_label=False,
                                     placeholder="student id", container=False)
                    newconv = gr.Button("New thread", size="sm", elem_classes="rail-newthread")

            with gr.Column(scale=1, elem_classes="mur-main"):
                header = gr.HTML(render_header("ask"))

                with gr.Column(visible=True) as view_ask:
                    with gr.Column(visible=True) as hero:
                        gr.HTML(render_hero())
                        tile_buttons = []
                        with gr.Row(elem_classes="tile-row"):
                            for i, sd in enumerate(SEEDS):
                                meta = ROUTE_META[sd["route"]]
                                with gr.Column(elem_classes="tile-col"):
                                    gr.HTML(
                                        f'<div class="tile-top">'
                                        f'<span class="mono tile-num" style="color:{meta["color"]}">0{i + 1}</span>'
                                        f'<span class="msg-route-chip" style="color:{meta["on"]};'
                                        f'background:{meta["color"]}">{_e(sd["route"])}</span></div>')
                                    tile_buttons.append(gr.Button(sd["q"], elem_classes="tile-q-btn"))
                                    gr.HTML(f'<div class="tile-why">{_e(sd["why"])}</div>')

                    with gr.Row():
                        with gr.Column(scale=3):
                            transcript = gr.HTML(render_transcript([]))
                            with gr.Column(elem_classes="mur-inputrow"):
                                with gr.Row(elem_classes="mur-input-box"):
                                    box = gr.Textbox(placeholder="Ask about a regulation or a "
                                                                 "service — أو اسأل بالعربية",
                                                     show_label=False, scale=8, autofocus=True,
                                                     lines=1, max_lines=4, elem_classes="mur-input",
                                                     container=False)
                                    send = gr.Button("Ask", scale=1, min_width=80, elem_classes="mur-ask-btn")
                                gr.HTML('<div class="mur-hint">Answers come from Al-Noor\'s own '
                                       'documents. Nothing is filed without an advisor.</div>')
                        with gr.Column(scale=1, elem_classes="mur-trace"):
                            trace = gr.HTML(render_trace(None, "", None))

                with gr.Column(visible=False) as view_approvals:
                    with gr.Column(elem_classes="approvals-wrap"):
                        approvals_html, _init_active = render_approvals()
                        approvals_view = gr.HTML(approvals_html)
                        with gr.Column(visible=bool(_init_active), elem_classes="decide-panel") as decide_panel:
                            decide_head = gr.HTML("")
                            reason_box = gr.Textbox(label="Stated reason — editable",
                                                    show_label=False, lines=3,
                                                    placeholder="The reason that reaches the Registrar")
                            with gr.Row():
                                note_box = gr.Textbox(label="If rejecting, why?",
                                                      placeholder="e.g. Past the end-of-week-10 deadline")
                            with gr.Row():
                                approve_btn = gr.Button("Approve as written", elem_classes="decide-btn-approve")
                                edit_btn = gr.Button("Approve with edit", elem_classes="decide-btn-edit")
                                reject_btn = gr.Button("Reject with note", elem_classes="decide-btn-reject")

                with gr.Column(visible=False) as view_memory:
                    memory_view = gr.HTML("")

        with gr.Accordion("Registrar log — what has actually been filed", open=False):
            reg = gr.Markdown(registry_view(), elem_classes="registrar-log")
            gr.Button("Refresh", size="sm").click(registry_view, outputs=reg)

        gr.HTML(
            f'<div class="mur-footer">model · {_e(info.get("model", "?"))} &nbsp;·&nbsp; '
            'two separate document stores &nbsp;·&nbsp; github.com/aljokha/murshid-capstone</div>')

        # ---- wiring ----
        ask_args = dict(fn=ask, inputs=[box, sid, thread_state, msgs_state, pending_state],
                        outputs=[msgs_state, transcript, hero, box, trace, nav_approvals, pending_state])
        send.click(**ask_args)
        box.submit(**ask_args)
        for btn, sd in zip(tile_buttons, SEEDS):
            btn.click(lambda sid_, th, m, p, q=sd["q"]: ask(q, sid_, th, m, p),
                      inputs=[sid, thread_state, msgs_state, pending_state],
                      outputs=[msgs_state, transcript, hero, box, trace, nav_approvals, pending_state])

        newconv.click(new_thread, inputs=[sid],
                      outputs=[msgs_state, box, thread_state, hero, trace, nav_approvals, pending_state]
                      ).then(lambda: render_transcript([]), outputs=transcript)
        sid.submit(switch_student, inputs=[sid, thread_state],
                  outputs=[msgs_state, box, thread_state, hero, trace, nav_approvals, pending_state]
                  ).then(lambda: render_transcript([]), outputs=transcript)

        nav_out = [view_state, view_ask, view_approvals, view_memory,
                  nav_ask, nav_approvals, nav_memory, header,
                  approvals_view, active_ticket_state, decide_panel, reason_box,
                  decide_head, memory_view]
        nav_ask.click(lambda sid_: go_view("ask", sid_), inputs=[sid], outputs=nav_out)
        nav_approvals.click(lambda sid_: go_view("approvals", sid_), inputs=[sid], outputs=nav_out)
        nav_memory.click(lambda sid_: go_view("memory", sid_), inputs=[sid], outputs=nav_out)

        decide_out = [approvals_view, active_ticket_state, decide_panel, reason_box,
                      decide_head, msgs_state, transcript, nav_approvals]
        approve_btn.click(lambda r, n, a, t, m, s: decide_ticket("approve", r, n, a, t, m, s),
                          inputs=[reason_box, note_box, active_ticket_state, thread_state, msgs_state, sid],
                          outputs=decide_out).then(registry_view, outputs=reg)
        edit_btn.click(lambda r, n, a, t, m, s: decide_ticket("edit", r, n, a, t, m, s),
                       inputs=[reason_box, note_box, active_ticket_state, thread_state, msgs_state, sid],
                       outputs=decide_out).then(registry_view, outputs=reg)
        reject_btn.click(lambda r, n, a, t, m, s: decide_ticket("reject", r, n, a, t, m, s),
                         inputs=[reason_box, note_box, active_ticket_state, thread_state, msgs_state, sid],
                         outputs=decide_out).then(registry_view, outputs=reg)

    return demo


def launch_kwargs() -> dict:
    """Arguments that belong on launch() for this Gradio version."""
    if _IS_G6:
        return {"css": CSS, "theme": _theme()}
    return {}
