"""
Gradio demo for MathPaper AI — host a live, shareable web UI for the multi-agent
pipeline straight from Colab (or anywhere).

Colab usage:
    !pip install -q gradio
    # (repo already cloned + installed, Ollama running with models pulled)
    import os
    os.environ["LLM_PROVIDER"] = "ollama"   # or "gemini"
    !python app.py                          # prints a public *.gradio.live URL

The public link stays live while the notebook session runs.
"""

import os
import sys

# make the package importable whether or not `pip install -e` registered the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr

from mathpaper import llm, PlanningAgent, HybridRetriever, load_demo_corpus

# ----------------------------------------------------------------------
# Model config: default to whatever the env says; make local names match
# the models people actually pull in Colab (llama3.2:3b + qwen2.5:7b).
# ----------------------------------------------------------------------
if os.environ.get("LLM_PROVIDER", "ollama") == "ollama":
    llm.PROVIDERS["ollama"]["small"] = os.environ.get("OLLAMA_SMALL", "llama3.2:3b")
    llm.PROVIDERS["ollama"]["strong"] = os.environ.get("OLLAMA_STRONG", "qwen2.5:7b")

# Longer timeout — local models cold-start slowly on first call.
import json, urllib.request
def _post_long(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())
llm._post = _post_long

# One planner instance, reused across requests (keeps conversation memory).
PLANNER = PlanningAgent(HybridRetriever(load_demo_corpus()))

SAMPLES = [
    "Why is KL divergence minimized in Equation (5)?",
    "What does lambda represent?",
    "Why use cross entropy instead of mean squared error?",
    "How is Equation (5) derived from the ELBO?",
]


def answer_question(question):
    if not question or not question.strip():
        return "_Enter a question about the paper._", ""
    try:
        state = PLANNER.run(question.strip())
    except Exception as e:
        msg = (f"**Error:** {e}\n\n"
               "(If this is a timeout, the model is still loading — "
               "try again in a few seconds.)")
        return msg, ""
    trace = "\n".join(f"- {t}" for t in state.trace)
    trace_md = f"### Agent trace\n{trace}"
    answer_md = state.answer or "_No answer produced._"
    return answer_md, trace_md


with gr.Blocks(title="MathPaper AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# MathPaper AI\n"
        "*Experimental approach to math intuition* — an agentic RAG system that "
        "explains concepts, derivations, and proofs from a research paper "
        "(*a VAE for molecular generation*) by coordinating specialized agents.\n\n"
        "Ask a question below. The **agent trace** shows which agents fired — "
        "simple questions skip agents they don't need (dynamic orchestration)."
    )
    with gr.Row():
        q = gr.Textbox(label="Your question", value=SAMPLES[0], scale=4)
        btn = gr.Button("Ask", variant="primary", scale=1)
    gr.Examples(examples=SAMPLES, inputs=q)
    with gr.Row():
        with gr.Column(scale=3):
            answer = gr.Markdown(label="Answer", latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "\\[", "right": "\\]", "display": True},
                {"left": "$", "right": "$", "display": False},
                {"left": "\\(", "right": "\\)", "display": False},
            ])
        with gr.Column(scale=2):
            trace = gr.Markdown()

    btn.click(answer_question, inputs=q, outputs=[answer, trace])
    q.submit(answer_question, inputs=q, outputs=[answer, trace])


if __name__ == "__main__":
    # share=True gives a public *.gradio.live URL (works from Colab)
    demo.launch(share=True)
