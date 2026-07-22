"""
Gradio demo for MathPaper AI — live multi-agent RAG over a research paper.

Answers are grounded in BOTH the paper (retrieved chunks) and external references
(Wikipedia / Encyclopedia of Mathematics / ProofWiki / MathWorld), rendered in LaTeX.

The agent pipeline runs in a background thread while the UI polls it, so you can
watch each agent fire live instead of staring at a blank panel.

Colab:
    !pip install -q gradio pymupdf
    import os; os.environ["LLM_PROVIDER"] = "ollama"
    !python app.py
"""

import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr

from mathpaper import llm, PlanningAgent, HybridRetriever, load_demo_corpus
from mathpaper.ingest import pdf_to_corpus, corpus_summary
from mathpaper.evaluation import score_answer, composite


def _post_long(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())
llm._post = _post_long

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

if PROVIDER == "ollama":
    MODEL_CHOICES = [
        "qwen2.5:7b", "llama3.2:3b", "qwen2.5:3b", "qwen2.5:14b",
        "deepseek-r1:7b", "deepseek-r1:8b", "deepseek-r1:14b",
        "gemma3:4b", "gemma3:12b", "phi4-mini", "mistral", "llama3.1:8b",
    ]
    DEFAULT_A, DEFAULT_B = "qwen2.5:7b", "deepseek-r1:7b"
else:
    MODEL_CHOICES = [llm.PROVIDERS[PROVIDER]["strong"], llm.PROVIDERS[PROVIDER]["small"]]
    DEFAULT_A, DEFAULT_B = MODEL_CHOICES[0], MODEL_CHOICES[-1]

SAMPLES = [
    "Why is KL divergence minimized in Equation (5)?",
    "How is Equation (5) derived from the ELBO?",
    "Why use cross entropy instead of mean squared error?",
    "What does lambda represent?",
]

LATEX = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]

ALL_AGENTS = ["Query Analyzer", "Planner", "Memory", "Paper Retrieval",
              "Evidence Verifier", "Math Knowledge", "Explanation Generator",
              "Citation Validator"]

STATE = {"corpus": load_demo_corpus(), "name": "Built-in demo (VAE paper)"}


# ---------------- output saving ----------------
OUTPUT_DIR = None
def _output_dir():
    global OUTPUT_DIR
    if OUTPUT_DIR:
        return OUTPUT_DIR
    drive = "/content/drive/MyDrive/Maths_Rag output"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive, exist_ok=True); OUTPUT_DIR = drive
    else:
        OUTPUT_DIR = os.path.abspath("Maths_Rag_output"); os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _save_run(rec):
    with open(os.path.join(_output_dir(), "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _planner_for(model_tag):
    if PROVIDER == "ollama":
        llm.PROVIDERS["ollama"]["small"] = model_tag
        llm.PROVIDERS["ollama"]["strong"] = model_tag
    return PlanningAgent(HybridRetriever(STATE["corpus"]))


# ---------------- paper handling ----------------
def load_pdf(f):
    if f is None:
        return f"Using: **{STATE['name']}**"
    try:
        STATE["corpus"] = pdf_to_corpus(f.name)
        STATE["name"] = os.path.basename(f.name)
        return f"Loaded **{STATE['name']}** — {corpus_summary(STATE['corpus'])}"
    except Exception as e:
        return f"**Could not read PDF:** {e}"


def use_demo():
    STATE["corpus"] = load_demo_corpus()
    STATE["name"] = "Built-in demo (VAE paper)"
    return f"Using: **{STATE['name']}**"


# ---------------- rendering ----------------
def _status_md(done, current, elapsed, model, note=""):
    """Live backend view: which agents have run, which is running now."""
    lines = []
    for a in ALL_AGENTS:
        if a in done:
            lines.append(f'<div class="ag ok">✓ {a}</div>')
        elif a == current:
            lines.append(f'<div class="ag run">▶ {a} <span class="dots">…</span></div>')
        else:
            lines.append(f'<div class="ag idle">· {a}</div>')
    head = (f'<div class="stat-head">backend · <b>{model}</b> · {elapsed:.0f}s</div>')
    return f'<div class="statbox">{head}{"".join(lines)}'\
           f'<div class="stat-note">{note}</div></div>'


def _tools_md(state):
    md = "### Tool-sourced background\n"
    if not state.external_knowledge:
        return md + "\n_No external lookup was needed for this question._"
    for k in state.external_knowledge:
        md += f"\n**{k.get('concept','')}** — *{k.get('source_name','external')}*  \n"
        md += f"{k.get('text','')[:320]}\n"
        u = k.get("source", "")
        if str(u).startswith("http"):
            md += f"\n[{u}]({u})\n"
    return md


def _evidence_md(state):
    if not state.evidence:
        return "### Paper evidence\n\n_none retrieved_"
    md = "### Paper evidence (retrieved chunks)\n"
    for c in state.evidence:
        md += f"\n**`{c['id']}`** *({c.get('section','')})*  \n{c['text'][:280]}\n"
    return md


def _force_tool(state, question):
    from mathpaper.agents import MathKnowledgeAgent, ExplanationGeneratorAgent
    c = re.sub(r"(?i)^(why|how|what)\s+(is|are|does|do|use)\s+", "", question)
    c = re.sub(r"(?i)\s*(in|from)\s+equation.*$", "", c).strip(" ?.")
    if not c:
        return state
    state.missing = [c]
    MathKnowledgeAgent().run(state)
    ExplanationGeneratorAgent().run(state)
    return state


# ---------------- tab 1: ask (streaming) ----------------
def run_ask(question, model_tag, force_tool):
    q = (question or "").strip()
    if not q:
        yield "_Enter a question._", "", "", ""
        return

    shared = {"done": [], "current": None, "state": None, "error": None, "note": ""}

    def on_step(label):
        if shared["current"]:
            shared["done"].append(shared["current"])
        shared["current"] = label

    def worker():
        try:
            planner = _planner_for(model_tag)
            st = planner.run(q, on_step=on_step)
            if force_tool and not st.external_knowledge:
                shared["current"] = "Math Knowledge"
                shared["note"] = "forcing external reference lookup…"
                st = _force_tool(st, q)
                shared["note"] = ""
            if shared["current"]:
                shared["done"].append(shared["current"])
            shared["current"] = None
            shared["state"] = st
        except Exception as e:
            shared["error"] = (e, traceback.format_exc())

    t0 = time.time()
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    # poll while the pipeline runs so the UI shows live progress
    while th.is_alive():
        yield ("⏳ *working…*",
               _status_md(shared["done"], shared["current"], time.time() - t0,
                          model_tag, shared["note"]),
               "", "")
        time.sleep(0.4)
    th.join()
    dt = time.time() - t0

    if shared["error"]:
        e, tb = shared["error"]
        yield (f"### Error\n```\n{e}\n```\n<details><summary>traceback</summary>\n\n"
               f"```\n{tb[-1500:]}\n```\n</details>",
               _status_md(shared["done"], None, dt, model_tag, "failed"), "", "")
        return

    st = shared["state"]
    if not (st.answer or "").strip():
        yield ("### No answer produced\nThe generator returned empty text. "
               "This usually means the model is still warming up — try again, or "
               "check the model is pulled (`!ollama list`).",
               _status_md(shared["done"], None, dt, model_tag, "empty answer"),
               _tools_md(st), _evidence_md(st))
        return

    _save_run({"time_iso": time.strftime("%Y-%m-%d %H:%M:%S"), "paper": STATE["name"],
               "model": model_tag, "question": q, "answer": st.answer,
               "trace": st.trace, "n_agents": len(st.trace), "latency_s": round(dt, 1)})

    detail = "<br>".join(f"· {x}" for x in st.trace)
    yield (st.answer,
           _status_md(shared["done"], None, dt, model_tag, "done") +
           f'<div class="tracebox"><b>trace</b><br>{detail}<br><br>'
           f'<b>paper</b>: {STATE["name"]}</div>',
           _tools_md(st), _evidence_md(st))


# ---------------- tab 2: compare ----------------
def run_compare(question, model_a, model_b, force_tool):
    q = (question or "").strip()
    if not q:
        yield "_Enter a question._", "", "_Enter a question._", ""
        return
    outs = {}
    for slot, tag in (("A", model_a), ("B", model_b)):
        yield (outs.get("A", "⏳ *waiting…*"), outs.get("At", ""),
               outs.get("B", "⏳ *waiting…*"), outs.get("Bt", ""))
        t0 = time.time()
        try:
            planner = _planner_for(tag)
            st = planner.run(q)
            if force_tool and not st.external_knowledge:
                st = _force_tool(st, q)
            dt = time.time() - t0
            _save_run({"time_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "paper": STATE["name"], "model": tag, "question": q,
                       "answer": st.answer or "", "trace": st.trace,
                       "n_agents": len(st.trace), "latency_s": round(dt, 1)})
            outs[slot] = st.answer or "_No answer._"
            outs[slot + "t"] = (f'<div class="tracebox"><b>{tag}</b> · {dt:.1f}s · '
                                f'{len(st.trace)} agents<br>'
                                + "<br>".join(f"· {x}" for x in st.trace)
                                + "</div>") + "\n\n" + _tools_md(st)
        except Exception as e:
            outs[slot] = f"### Error\n```\n{e}\n```"
            outs[slot + "t"] = ""
    yield outs.get("A", ""), outs.get("At", ""), outs.get("B", ""), outs.get("Bt", "")


# ---------------- tab 3: evaluate ----------------
def run_evaluation(questions_text, selected_models, progress=gr.Progress()):
    if not selected_models:
        return "_Pick at least one model._", None
    questions = [x.strip() for x in (questions_text or "").splitlines() if x.strip()]
    if not questions:
        return "_Enter at least one question (one per line)._", None

    rows, errors = [], []
    total, done = len(selected_models) * len(questions), 0
    for model in selected_models:
        for q in questions:
            progress(done / total, desc=f"{model} · {q[:30]}…")
            try:
                planner = _planner_for(model)
                t0 = time.time()
                st = planner.run(q)
                dt = time.time() - t0
                sc = score_answer(st.answer, st.evidence)
                rows.append({"model": model, "question": q[:40],
                             "composite": composite(sc),
                             "cite_valid": sc["citation_validity"],
                             "grounded": sc["groundedness"],
                             "halluc": sc["hallucination_flag"],
                             "agents": len(st.trace), "latency_s": round(dt, 1)})
            except Exception as e:
                errors.append(f"{model} / {q[:30]}: {e}")
            done += 1

    if not rows:
        return "### Evaluation failed\n" + "\n".join(f"- {e}" for e in errors), None

    summary = {}
    for r in rows:
        s = summary.setdefault(r["model"], {k: [] for k in
                ("composite", "grounded", "cite_valid", "halluc", "latency_s", "agents")})
        for k in s:
            s[k].append(r[k])
    avg = lambda xs: round(sum(xs) / len(xs), 3)

    md = "### Model leaderboard (averaged over questions)\n\n"
    md += "| Model | Composite | Grounded | Cite-valid | Halluc. | Avg agents | Avg latency |\n|---|---|---|---|---|---|---|\n"
    for model, s in sorted(summary.items(), key=lambda kv: -avg(kv[1]["composite"])):
        md += (f"| `{model}` | **{avg(s['composite'])}** | {avg(s['grounded'])} | "
               f"{avg(s['cite_valid'])} | {avg(s['halluc'])} | {avg(s['agents'])} | "
               f"{avg(s['latency_s'])}s |\n")
    md += ("\n*Composite blends citation validity, coverage and groundedness minus a "
           "hallucination penalty (reference-free, 0–1). Higher is better.*")
    if errors:
        md += "\n\n**Errors:**\n" + "\n".join(f"- {e}" for e in errors[:5])

    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(_output_dir(), f"evaluation_{ts}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md += f"\n\n💾 Saved to `{path}`"

    cols = ["model", "question", "composite", "cite_valid", "grounded",
            "halluc", "agents", "latency_s"]
    return md, [[r[c] for c in cols] for r in rows]


# ---------------- theme (matches the original demo palette) ----------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');
.gradio-container, .gradio-container * { font-family: 'STIX Two Text', Georgia, serif !important; }
.gradio-container { background: #16211b !important; color: #ece7da !important; max-width: 1180px !important; }
.gradio-container h1 { font-size: 2.4rem !important; font-weight: 600 !important; color: #ece7da !important; }
.gradio-container h2, .gradio-container h3, .gradio-container h4 { color: #ece7da !important; }
.gradio-container p, .gradio-container li, .gradio-container span, .gradio-container label { color: #ece7da !important; }
.gradio-container em { color: #a9b3a8 !important; }
.block, .form, .gr-box, .gr-panel { background: #19251f !important; border-color: #3a4a40 !important; border-radius: 2px !important; }
input, textarea, .gr-input, .gr-text-input { background: #1e2b24 !important; color: #ece7da !important; border-color: #3a4a40 !important; border-radius: 2px !important; }
button.primary, .gr-button-primary { background: #e6c76d !important; color: #16211b !important; border: none !important; border-radius: 2px !important; font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; }
button.secondary, .gr-button { background: transparent !important; color: #a9b3a8 !important; border: 1px solid #3a4a40 !important; border-radius: 2px !important; font-family: 'IBM Plex Mono', monospace !important; }
.tab-nav button { color: #a9b3a8 !important; font-family: 'IBM Plex Mono', monospace !important; }
.tab-nav button.selected { color: #e6c76d !important; border-bottom-color: #e6c76d !important; }
a { color: #a3c6d8 !important; }
code, pre, .gradio-container code { font-family: 'IBM Plex Mono', monospace !important; background: #1e2b24 !important; color: #a3c6d8 !important; }
table { border-color: #3a4a40 !important; }
th { background: #1e2b24 !important; color: #e6c76d !important; font-family: 'IBM Plex Mono', monospace !important; }
td { border-color: #3a4a40 !important; color: #ece7da !important; }
/* live backend status box */
.statbox { border: 1px solid #3a4a40; border-radius: 2px; padding: 12px 14px; background: #19251f; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
.stat-head { color: #e6c76d; text-transform: uppercase; letter-spacing: .12em; font-size: 10px; margin-bottom: 9px; }
.ag { padding: 2px 0; }
.ag.ok { color: #7fd1a3; }
.ag.run { color: #e6c76d; animation: blink 1s infinite; }
.ag.idle { color: #4a5a50; }
.stat-note { color: #d99a86; margin-top: 8px; font-size: 11px; }
.tracebox { border: 1px dashed #3a4a40; border-radius: 2px; padding: 10px 12px; margin-top: 10px; background: #16211b; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #a9b3a8; line-height: 1.65; }
@keyframes blink { 0%,100% { opacity: 1 } 50% { opacity: .45 } }
"""

with gr.Blocks(title="MathPaper AI") as demo:
    gr.HTML(f"<style>{CSS}</style>")   # theme, version-independent
    gr.Markdown(
        "# MathPaper AI\n"
        "*Experimental approach to math intuition* — an agentic RAG system that "
        "explains concepts, derivations and proofs from research papers.\n\n"
        "Answers are grounded in **the paper** (retrieved chunks) *and* **external "
        "references** (Wikipedia · Encyclopedia of Mathematics · ProofWiki · "
        "MathWorld), rendered in **LaTeX**. Runs save to `Maths_Rag output` on Drive."
    )

    with gr.Accordion("📄 Paper — upload your own, or use the demo", open=False):
        with gr.Row():
            pdf = gr.File(label="Research paper (PDF)", file_types=[".pdf"])
            demo_btn = gr.Button("Use built-in demo paper")
        paper_status = gr.Markdown(f"Using: **{STATE['name']}**")
        pdf.change(load_pdf, pdf, paper_status)
        demo_btn.click(use_demo, None, paper_status)

    with gr.Tab("Ask (single model)"):
        with gr.Row():
            q1 = gr.Textbox(label="Your question", value=SAMPLES[0], scale=3)
            m1 = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_A, label="Language model", scale=1)
            f1 = gr.Checkbox(label="force tool lookup", value=True, scale=1)
            b1 = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(SAMPLES, inputs=q1)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## Answer")
                ans1 = gr.Markdown(latex_delimiters=LATEX)
            with gr.Column(scale=2):
                stat1 = gr.HTML()
                tools1 = gr.Markdown()
        with gr.Accordion("Retrieved paper evidence", open=False):
            ev1 = gr.Markdown()
        b1.click(run_ask, [q1, m1, f1], [ans1, stat1, tools1, ev1])
        q1.submit(run_ask, [q1, m1, f1], [ans1, stat1, tools1, ev1])

    with gr.Tab("Compare two models"):
        with gr.Row():
            q2 = gr.Textbox(label="Your question", value=SAMPLES[0], scale=2)
            ma = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_A, label="Model A", scale=1)
            mb = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_B, label="Model B", scale=1)
            f2 = gr.Checkbox(label="force tool lookup", value=True, scale=1)
            b2 = gr.Button("Compare", variant="primary", scale=1)
        gr.Examples(SAMPLES, inputs=q2)
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model A")
                ansA = gr.Markdown(latex_delimiters=LATEX)
                trA = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### Model B")
                ansB = gr.Markdown(latex_delimiters=LATEX)
                trB = gr.Markdown()
        b2.click(run_compare, [q2, ma, mb, f2], [ansA, trA, ansB, trB])

    with gr.Tab("📊 Evaluate models"):
        gr.Markdown("Score several language models on a question set with "
                    "reference-free metrics, then rank them.")
        with gr.Row():
            eq = gr.Textbox(label="Questions (one per line)",
                            value="\n".join(SAMPLES), lines=5, scale=2)
            em = gr.CheckboxGroup(MODEL_CHOICES, value=[DEFAULT_A, DEFAULT_B],
                                  label="Models to evaluate", scale=1)
        b3 = gr.Button("Run evaluation", variant="primary")
        board = gr.Markdown()
        table = gr.Dataframe(
            headers=["model", "question", "composite", "cite_valid",
                     "grounded", "halluc", "agents", "latency_s"],
            label="Per-question detail", wrap=True)
        b3.click(run_evaluation, [eq, em], [board, table])


if __name__ == "__main__":
    # Gradio 6 wants css on launch(); older versions accept it on Blocks.
    try:
        demo.queue().launch(share=True, css=CSS)
    except TypeError:
        demo.css = CSS
        demo.queue().launch(share=True)
