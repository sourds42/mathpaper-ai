"""
Gradio demo for MathPaper AI — a live, shareable web UI for the multi-agent RAG
pipeline. Host it from Colab (prints a public *.gradio.live URL).

Three features:
  1. Upload your own paper (PDF) — replaces the built-in demo corpus.
  2. Live agent status — watch each agent fire in the backend as it runs.
  3. Multi-model comparison — run the same question through two models
     side-by-side and compare answers, agent traces, and timing.

Colab usage:
    !pip install -q gradio pymupdf
    import os; os.environ["LLM_PROVIDER"] = "ollama"
    !python app.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import json
import urllib.request

import gradio as gr

from mathpaper import llm, PlanningAgent, HybridRetriever, load_demo_corpus
from mathpaper.ingest import pdf_to_corpus, corpus_summary
from mathpaper.evaluation import score_answer, composite

# Longer timeout — local models cold-start slowly on the first call.
def _post_long(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())
llm._post = _post_long

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# Model choices offered in the UI. For Ollama these are tags you've pulled.
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
    "What does lambda represent?",
    "Why use cross entropy instead of mean squared error?",
    "How is Equation (5) derived from the ELBO?",
]

# ---- shared corpus state: starts as the demo paper, swapped by PDF upload ----
STATE = {"corpus": load_demo_corpus(), "name": "Built-in demo (VAE paper)"}


def _planner_for(model_tag):
    """Build a planner whose strong+small roles both point at one model tag,
    so the whole pipeline runs on the chosen model (clean for comparison)."""
    if PROVIDER == "ollama":
        llm.PROVIDERS["ollama"]["small"] = model_tag
        llm.PROVIDERS["ollama"]["strong"] = model_tag
    return PlanningAgent(HybridRetriever(STATE["corpus"]))


def load_pdf(pdf_file):
    if pdf_file is None:
        return f"Using: **{STATE['name']}**"
    try:
        corpus = pdf_to_corpus(pdf_file.name)
    except Exception as e:
        return f"**Could not read PDF:** {e}"
    STATE["corpus"] = corpus
    STATE["name"] = os.path.basename(pdf_file.name)
    return f"Loaded **{STATE['name']}** — {corpus_summary(corpus)}"


def use_demo():
    STATE["corpus"] = load_demo_corpus()
    STATE["name"] = "Built-in demo (VAE paper)"
    return f"Using: **{STATE['name']}**"


# ---- single-model run with live status streaming ----
# ---- output saving: Google Drive folder if mounted, else local ----
OUTPUT_DIR = None
def _output_dir():
    """Resolve the save folder once. Prefers Drive: /content/drive/MyDrive/Maths_Rag output.
    Falls back to ./Maths_Rag_output if Drive isn't mounted."""
    global OUTPUT_DIR
    if OUTPUT_DIR:
        return OUTPUT_DIR
    drive = "/content/drive/MyDrive/Maths_Rag output"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive, exist_ok=True)
        OUTPUT_DIR = drive
    else:
        local = os.path.abspath("Maths_Rag_output")
        os.makedirs(local, exist_ok=True)
        OUTPUT_DIR = local
    return OUTPUT_DIR


def _save_run(record: dict):
    """Append one run to a JSONL log and also write a readable per-run .md file."""
    d = _output_dir()
    # append to a single JSONL log (easy to load later for analysis)
    with open(os.path.join(d, "runs.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    # human-readable copy
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_model = record["model"].replace(":", "-").replace("/", "-")
    fname = f"{ts}_{safe_model}.md"
    with open(os.path.join(d, fname), "w") as f:
        f.write(f"# MathPaper AI run\n\n")
        f.write(f"- **Time:** {record['time_iso']}\n")
        f.write(f"- **Paper:** {record['paper']}\n")
        f.write(f"- **Model:** `{record['model']}`\n")
        f.write(f"- **Latency:** {record['latency_s']:.1f}s\n")
        f.write(f"- **Agents fired:** {record['n_agents']}\n\n")
        f.write(f"## Question\n{record['question']}\n\n")
        f.write("## Agent trace\n" + "\n".join(f"- {t}" for t in record["trace"]) + "\n\n")
        f.write(f"## Answer\n{record['answer']}\n")
    return os.path.join(d, fname)


def run_streaming(question, model_tag):
    if not question or not question.strip():
        yield "_Enter a question._", ""
        return
    planner = _planner_for(model_tag)

    yield "⏳ Running agents…", "### Agent status\n- starting…"

    t0 = time.time()
    state = planner.run(question.strip(), on_step=lambda l: None)
    dt = time.time() - t0

    saved = _save_run({
        "time_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper": STATE["name"], "model": model_tag,
        "question": question.strip(), "answer": state.answer or "",
        "trace": state.trace, "n_agents": len(state.trace), "latency_s": dt,
    })
    status = ("### Agent status\n" + "\n".join(f"- ✓ {s}" for s in state.trace)
              + f"\n\n**Model:** `{model_tag}` · **Time:** {dt:.1f}s"
              + f"\n\n💾 Saved to `{saved}`")
    yield (state.answer or "_No answer produced._"), status


# ---- two-model comparison ----
def run_compare(question, model_a, model_b):
    if not question or not question.strip():
        return "_Enter a question._", "", "_Enter a question._", ""
    results = []
    for tag in (model_a, model_b):
        planner = _planner_for(tag)
        t0 = time.time()
        state = planner.run(question.strip())
        dt = time.time() - t0
        _save_run({
            "time_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paper": STATE["name"], "model": tag,
            "question": question.strip(), "answer": state.answer or "",
            "trace": state.trace, "n_agents": len(state.trace), "latency_s": dt,
        })
        trace = "\n".join(f"- {s}" for s in state.trace)
        meta = f"### `{tag}`\n{trace}\n\n**Time:** {dt:.1f}s · **Agents:** {len(state.trace)}"
        results.append((state.answer or "_No answer._", meta))
    return results[0][0], results[0][1], results[1][0], results[1][1]


# ---- evaluation: score selected models across a set of questions ----
def run_evaluation(questions_text, selected_models, progress=gr.Progress()):
    if not selected_models:
        return "_Pick at least one model._", None
    questions = [q.strip() for q in (questions_text or "").splitlines() if q.strip()]
    if not questions:
        return "_Enter at least one question (one per line)._", None

    rows = []
    total = len(selected_models) * len(questions)
    done = 0
    for model in selected_models:
        for q in questions:
            progress(done / total, desc=f"{model} · {q[:30]}…")
            planner = _planner_for(model)
            t0 = time.time()
            state = planner.run(q)
            dt = time.time() - t0
            sc = score_answer(state.answer, state.evidence)
            rows.append({
                "model": model,
                "question": q[:40] + ("…" if len(q) > 40 else ""),
                "composite": composite(sc),
                "cite_valid": sc["citation_validity"],
                "grounded": sc["groundedness"],
                "halluc": sc["hallucination_flag"],
                "agents": len(state.trace),
                "latency_s": round(dt, 1),
            })
            done += 1

    # per-model averages
    summary = {}
    for r in rows:
        m = r["model"]
        s = summary.setdefault(m, {"composite": [], "grounded": [], "cite_valid": [],
                                    "halluc": [], "latency_s": [], "agents": []})
        for k in s:
            s[k].append(r[k])
    avg = lambda xs: round(sum(xs) / len(xs), 3)

    # build a markdown leaderboard sorted by composite
    board = sorted(summary.items(), key=lambda kv: -avg(kv[1]["composite"]))
    md = "### Model leaderboard (averaged over questions)\n\n"
    md += "| Model | Composite | Grounded | Cite-valid | Halluc. | Avg agents | Avg latency |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for model, s in board:
        md += (f"| `{model}` | **{avg(s['composite'])}** | {avg(s['grounded'])} | "
               f"{avg(s['cite_valid'])} | {avg(s['halluc'])} | {avg(s['agents'])} | "
               f"{avg(s['latency_s'])}s |\n")
    md += ("\n*Composite blends citation validity, coverage, and groundedness "
           "minus a hallucination penalty (reference-free, 0–1). Higher is better.*")

    # save the full per-run detail to Drive
    d = _output_dir()
    ts = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(d, f"evaluation_{ts}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md += f"\n\n💾 Detailed results saved to `{os.path.join(d, f'evaluation_{ts}.jsonl')}`"

    # dataframe wants list-of-lists in header order
    cols = ["model", "question", "composite", "cite_valid",
            "grounded", "halluc", "agents", "latency_s"]
    table = [[r[c] for c in cols] for r in rows]
    return md, table


LATEX = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]

with gr.Blocks(title="MathPaper AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# MathPaper AI\n"
        "*Experimental approach to math intuition* — an agentic RAG system that "
        "explains concepts, derivations, and proofs from research papers by "
        "coordinating specialized agents.\n\n"
        "*Every run is saved to `Maths_Rag output` on your Drive (if mounted).*"
    )

    with gr.Accordion("📄 Paper: upload your own, or use the demo", open=False):
        with gr.Row():
            pdf = gr.File(label="Upload a research paper (PDF)", file_types=[".pdf"])
            with gr.Column():
                demo_btn = gr.Button("Use built-in demo paper")
        paper_status = gr.Markdown(f"Using: **{STATE['name']}**")
        pdf.change(load_pdf, inputs=pdf, outputs=paper_status)
        demo_btn.click(use_demo, outputs=paper_status)

    with gr.Tab("Ask (single model)"):
        with gr.Row():
            q1 = gr.Textbox(label="Your question", value=SAMPLES[0], scale=3)
            model1 = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_A, label="Model", scale=1)
            ask1 = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(SAMPLES, inputs=q1)
        with gr.Row():
            ans1 = gr.Markdown(latex_delimiters=LATEX)
            status1 = gr.Markdown()
        ask1.click(run_streaming, inputs=[q1, model1], outputs=[ans1, status1])
        q1.submit(run_streaming, inputs=[q1, model1], outputs=[ans1, status1])

    with gr.Tab("Compare two models"):
        with gr.Row():
            q2 = gr.Textbox(label="Your question", value=SAMPLES[0], scale=2)
            model_a = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_A, label="Model A", scale=1)
            model_b = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_B, label="Model B", scale=1)
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
        ask2.click(run_compare, inputs=[q2, model_a, model_b],
                   outputs=[ansA, traceA, ansB, traceB])

    with gr.Tab("📊 Evaluate models"):
        gr.Markdown(
            "Score multiple models on a set of questions using reference-free "
            "metrics (faithfulness / groundedness / hallucination), then rank them. "
            "Results save to your Drive folder."
        )
        with gr.Row():
            eval_questions = gr.Textbox(
                label="Questions (one per line)",
                value="\n".join(SAMPLES), lines=5, scale=2)
            eval_models = gr.CheckboxGroup(
                MODEL_CHOICES, value=[DEFAULT_A, DEFAULT_B],
                label="Models to evaluate", scale=1)
        eval_btn = gr.Button("Run evaluation", variant="primary")
        eval_board = gr.Markdown()
        eval_table = gr.Dataframe(
            headers=["model", "question", "composite", "cite_valid",
                     "grounded", "halluc", "agents", "latency_s"],
            label="Per-question detail", wrap=True)
        eval_btn.click(run_evaluation, inputs=[eval_questions, eval_models],
                       outputs=[eval_board, eval_table])

if __name__ == "__main__":
    demo.queue().launch(share=True)
