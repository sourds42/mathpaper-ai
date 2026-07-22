"""
Gradio demo for MathPaper AI — a live, shareable web UI for the multi-agent RAG
pipeline. Host it from Colab (prints a public *.gradio.live URL).

Tabs:
  1. Ask (single model)  — answer + agent trace + tool sources + paper evidence
  2. Compare two models  — same question through two LLMs, side by side
  3. Evaluate models     — score several LLMs on a question set, ranked leaderboard

Answers are grounded in BOTH the paper (retrieved chunks) and external references
(Wikipedia / Encyclopedia of Mathematics / ProofWiki / MathWorld) and rendered in
LaTeX.

Colab usage:
    !pip install -q gradio pymupdf
    import os; os.environ["LLM_PROVIDER"] = "ollama"
    !python app.py
"""

import json
import os
import sys
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


# ---- local models cold-start slowly: raise the HTTP timeout -----------
def _post_long(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())
llm._post = _post_long

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# Language models offered in the UI. For Ollama these are tags you have pulled.
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

STATE = {"corpus": load_demo_corpus(), "name": "Built-in demo (VAE paper)"}


# ---- output saving: Drive folder if mounted, else local ----------------
OUTPUT_DIR = None
def _output_dir():
    global OUTPUT_DIR
    if OUTPUT_DIR:
        return OUTPUT_DIR
    drive = "/content/drive/MyDrive/Maths_Rag output"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive, exist_ok=True)
        OUTPUT_DIR = drive
    else:
        OUTPUT_DIR = os.path.abspath("Maths_Rag_output")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _save_run(record):
    d = _output_dir()
    with open(os.path.join(d, "runs.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    return d


def _planner_for(model_tag):
    """Point both agent roles at one model so comparisons are clean."""
    if PROVIDER == "ollama":
        llm.PROVIDERS["ollama"]["small"] = model_tag
        llm.PROVIDERS["ollama"]["strong"] = model_tag
    return PlanningAgent(HybridRetriever(STATE["corpus"]))


# ---- paper handling ----------------------------------------------------
def load_pdf(pdf_file):
    if pdf_file is None:
        return f"Using: **{STATE['name']}**"
    try:
        STATE["corpus"] = pdf_to_corpus(pdf_file.name)
        STATE["name"] = os.path.basename(pdf_file.name)
        return f"Loaded **{STATE['name']}** — {corpus_summary(STATE['corpus'])}"
    except Exception as e:
        return f"**Could not read PDF:** {e}"


def use_demo():
    STATE["corpus"] = load_demo_corpus()
    STATE["name"] = "Built-in demo (VAE paper)"
    return f"Using: **{STATE['name']}**"


# ---- shared rendering helpers -----------------------------------------
def _trace_md(state, model_tag, dt):
    return ("### Agent trace\n" + "\n".join(f"- {t}" for t in state.trace)
            + f"\n\n**Model:** `{model_tag}` · **Time:** {dt:.1f}s"
            + f"\n\n**Paper:** {STATE['name']}")


def _tools_md(state):
    md = "### Tool-sourced background\n"
    if not state.external_knowledge:
        return md + ("\n_No external lookup needed — the verifier judged the paper "
                     "evidence sufficient. Tick **force tool lookup** to see the "
                     "reference tool fire anyway._")
    for k in state.external_knowledge:
        md += f"\n**{k.get('concept','')}** — *{k.get('source_name','external')}*  \n"
        md += f"{k.get('text','')[:320]}\n"
        url = k.get("source", "")
        if str(url).startswith("http"):
            md += f"\n[{url}]({url})\n"
    return md


def _evidence_md(state):
    if not state.evidence:
        return "### Paper evidence\n\n_none retrieved_"
    md = "### Paper evidence (retrieved chunks)\n"
    for c in state.evidence:
        md += f"\n**`{c['id']}`** *({c.get('section','')})*  \n{c['text'][:280]}\n"
    return md


def _force_tool(state, question):
    """Run the reference tool even if the verifier didn't ask for it, then
    regenerate so the answer actually uses the fetched definition."""
    import re
    from mathpaper.agents import MathKnowledgeAgent, ExplanationGeneratorAgent
    concept = re.sub(r"(?i)^(why|how|what)\s+(is|are|does|do|use)\s+", "", question)
    concept = re.sub(r"(?i)\s*(in|from)\s+equation.*$", "", concept).strip(" ?.")
    if not concept:
        return state
    state.missing = [concept]
    MathKnowledgeAgent().run(state)
    ExplanationGeneratorAgent().run(state)
    return state


# ---- tab 1: ask --------------------------------------------------------
def run_ask(question, model_tag, force_tool):
    q = (question or "").strip()
    if not q:
        return "_Enter a question._", "", "", ""
    try:
        planner = _planner_for(model_tag)
        t0 = time.time()
        state = planner.run(q)
        if force_tool and not state.external_knowledge:
            state = _force_tool(state, q)
        dt = time.time() - t0
        _save_run({"time_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "paper": STATE["name"], "model": model_tag, "question": q,
                   "answer": state.answer or "", "trace": state.trace,
                   "n_agents": len(state.trace), "latency_s": round(dt, 1)})
        return (state.answer or "_No answer produced._",
                _trace_md(state, model_tag, dt), _tools_md(state), _evidence_md(state))
    except Exception as e:
        # surface errors in the UI instead of failing silently
        return (f"### Error\n```\n{e}\n```\n<details><summary>traceback</summary>\n\n"
                f"```\n{traceback.format_exc()[-1500:]}\n```\n</details>",
                "", "", "")


# ---- tab 2: compare ----------------------------------------------------
def run_compare(question, model_a, model_b, force_tool):
    q = (question or "").strip()
    if not q:
        return "_Enter a question._", "", "_Enter a question._", ""
    out = []
    for tag in (model_a, model_b):
        try:
            planner = _planner_for(tag)
            t0 = time.time()
            state = planner.run(q)
            if force_tool and not state.external_knowledge:
                state = _force_tool(state, q)
            dt = time.time() - t0
            _save_run({"time_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "paper": STATE["name"], "model": tag, "question": q,
                       "answer": state.answer or "", "trace": state.trace,
                       "n_agents": len(state.trace), "latency_s": round(dt, 1)})
            meta = _trace_md(state, tag, dt) + "\n\n" + _tools_md(state)
            out.append((state.answer or "_No answer._", meta))
        except Exception as e:
            out.append((f"### Error\n```\n{e}\n```", ""))
    return out[0][0], out[0][1], out[1][0], out[1][1]


# ---- tab 3: evaluate ---------------------------------------------------
def run_evaluation(questions_text, selected_models, progress=gr.Progress()):
    if not selected_models:
        return "_Pick at least one model._", None
    questions = [q.strip() for q in (questions_text or "").splitlines() if q.strip()]
    if not questions:
        return "_Enter at least one question (one per line)._", None

    rows, errors = [], []
    total = len(selected_models) * len(questions)
    done = 0
    for model in selected_models:
        for q in questions:
            progress(done / total, desc=f"{model} · {q[:30]}…")
            try:
                planner = _planner_for(model)
                t0 = time.time()
                state = planner.run(q)
                dt = time.time() - t0
                sc = score_answer(state.answer, state.evidence)
                rows.append({"model": model, "question": q[:40],
                             "composite": composite(sc),
                             "cite_valid": sc["citation_validity"],
                             "grounded": sc["groundedness"],
                             "halluc": sc["hallucination_flag"],
                             "agents": len(state.trace),
                             "latency_s": round(dt, 1)})
            except Exception as e:
                errors.append(f"{model} / {q[:30]}: {e}")
            done += 1

    if not rows:
        return ("### Evaluation failed\n" + "\n".join(f"- {e}" for e in errors)), None

    summary = {}
    for r in rows:
        s = summary.setdefault(r["model"], {k: [] for k in
                ("composite", "grounded", "cite_valid", "halluc", "latency_s", "agents")})
        for k in s:
            s[k].append(r[k])
    avg = lambda xs: round(sum(xs) / len(xs), 3)

    board = sorted(summary.items(), key=lambda kv: -avg(kv[1]["composite"]))
    md = "### Model leaderboard (averaged over questions)\n\n"
    md += "| Model | Composite | Grounded | Cite-valid | Halluc. | Avg agents | Avg latency |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for model, s in board:
        md += (f"| `{model}` | **{avg(s['composite'])}** | {avg(s['grounded'])} | "
               f"{avg(s['cite_valid'])} | {avg(s['halluc'])} | {avg(s['agents'])} | "
               f"{avg(s['latency_s'])}s |\n")
    md += ("\n*Composite blends citation validity, coverage, and groundedness minus a "
           "hallucination penalty (reference-free, 0–1). Higher is better.*")
    if errors:
        md += "\n\n**Errors:**\n" + "\n".join(f"- {e}" for e in errors[:5])

    d = _output_dir()
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(d, f"evaluation_{ts}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md += f"\n\n💾 Detailed results saved to `{path}`"

    cols = ["model", "question", "composite", "cite_valid", "grounded",
            "halluc", "agents", "latency_s"]
    return md, [[r[c] for c in cols] for r in rows]


# ---- UI ----------------------------------------------------------------
with gr.Blocks(title="MathPaper AI") as demo:
    gr.Markdown(
        "# MathPaper AI\n"
        "*Experimental approach to math intuition* — an agentic RAG system that "
        "explains concepts, derivations, and proofs from research papers.\n\n"
        "Answers are grounded in **the paper** (retrieved chunks) *and* **external "
        "references** (Wikipedia · Encyclopedia of Mathematics · ProofWiki · "
        "MathWorld), and rendered in **LaTeX**. Runs are saved to "
        "`Maths_Rag output` on Drive."
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
            ask1 = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(SAMPLES, inputs=q1)
        gr.Markdown("## Answer")
        ans1 = gr.Markdown(latex_delimiters=LATEX)
        with gr.Row():
            trace1 = gr.Markdown()
            tools1 = gr.Markdown()
        with gr.Accordion("Retrieved paper evidence", open=False):
            ev1 = gr.Markdown()
        ask1.click(run_ask, [q1, m1, f1], [ans1, trace1, tools1, ev1])
        q1.submit(run_ask, [q1, m1, f1], [ans1, trace1, tools1, ev1])

    with gr.Tab("Compare two models"):
        with gr.Row():
            q2 = gr.Textbox(label="Your question", value=SAMPLES[0], scale=2)
            ma = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_A, label="Model A", scale=1)
            mb = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_B, label="Model B", scale=1)
            f2 = gr.Checkbox(label="force tool lookup", value=True, scale=1)
            ask2 = gr.Button("Compare", variant="primary", scale=1)
        gr.Examples(SAMPLES, inputs=q2)
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model A")
                ansA = gr.Markdown(latex_delimiters=LATEX)
                traceA = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### Model B")
                ansB = gr.Markdown(latex_delimiters=LATEX)
                traceB = gr.Markdown()
        ask2.click(run_compare, [q2, ma, mb, f2], [ansA, traceA, ansB, traceB])

    with gr.Tab("📊 Evaluate models"):
        gr.Markdown(
            "Score several language models on a question set with reference-free "
            "metrics (citation validity / groundedness / hallucination), then rank them."
        )
        with gr.Row():
            eq = gr.Textbox(label="Questions (one per line)",
                            value="\n".join(SAMPLES), lines=5, scale=2)
            em = gr.CheckboxGroup(MODEL_CHOICES, value=[DEFAULT_A, DEFAULT_B],
                                  label="Models to evaluate", scale=1)
        eb = gr.Button("Run evaluation", variant="primary")
        board = gr.Markdown()
        table = gr.Dataframe(
            headers=["model", "question", "composite", "cite_valid",
                     "grounded", "halluc", "agents", "latency_s"],
            label="Per-question detail", wrap=True)
        eb.click(run_evaluation, [eq, em], [board, table])


if __name__ == "__main__":
    demo.queue().launch(share=True)
