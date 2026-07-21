import { useState, useRef, useEffect } from "react";

// ---------------- toy paper corpus, now in LaTeX ----------------
const CORPUS = [
  { id: "chunk_0", section: "abstract", text: "We propose a variational autoencoder for molecular generation trained with a composite objective balancing reconstruction and regularization." },
  { id: "chunk_1", section: "intro", text: "Generative models for molecules must produce valid structures. Prior work uses GANs but suffers from mode collapse." },
  { id: "chunk_2", section: "method", text: "Equation (3): $\\mathcal{L}_{\\mathrm{rec}} = -\\mathbb{E}_{q}[\\log p(x \\mid z)]$ is the reconstruction loss, the expected negative log-likelihood of the data under the decoder." },
  { id: "chunk_3", section: "method", text: "Equation (5): $\\mathcal{L} = \\mathcal{L}_{\\mathrm{rec}} + \\beta \\, D_{\\mathrm{KL}}\\big(q(z \\mid x) \\,\\|\\, p(z)\\big)$. We minimize KL divergence between the approximate posterior $q(z \\mid x)$ and the prior $p(z)$ to regularize the latent space." },
  { id: "chunk_4", section: "method", text: "The hyperparameter $\\beta$ controls the trade-off between reconstruction fidelity and latent regularization. We set $\\beta = 0.5$ by validation." },
  { id: "chunk_5", section: "method", text: "The symbol $\\lambda$ denotes the learning-rate decay coefficient in the scheduler, set to $0.95$ per epoch." },
  { id: "chunk_6", section: "method", text: "Equation (8): $z = \\mu + \\sigma \\odot \\epsilon$, $\\epsilon \\sim \\mathcal{N}(0, I)$. The reparameterization trick allows gradients to flow through the sampling step." },
  { id: "chunk_7", section: "method", text: "In Equation (8) the second term is $\\sigma \\odot \\epsilon$; sigma is squared in the KL term of Equation (5) because the KL between Gaussians depends on the variance $\\sigma^2$, not the standard deviation." },
  { id: "chunk_8", section: "training", text: "We use cross entropy for atom-type prediction rather than mean squared error because atom types are categorical; cross entropy matches the multinomial likelihood." },
  { id: "chunk_9", section: "training", text: "Training runs for 200 epochs with Adam, batch size 128, on a single A100." },
  { id: "chunk_10", section: "results", text: "Our model achieves 94.2% validity, beating the GAN baseline at 87.1%." },
  { id: "chunk_11", section: "results", text: "Ablation: removing the KL term of Equation (5) collapses the latent space and validity drops to 71%." },
  { id: "chunk_12", section: "related", text: "beta-VAE introduced the weighting of the divergence term to encourage disentangled representations." },
  { id: "chunk_13", section: "appendix", text: "Derivation of Equation (5): starting from the ELBO, $\\log p(x) \\geq \\mathbb{E}_{q}[\\log p(x \\mid z)] - D_{\\mathrm{KL}}\\big(q(z \\mid x) \\,\\|\\, p(z)\\big)$; maximizing the ELBO equals minimizing $\\mathcal{L}$." },
  { id: "chunk_14", section: "appendix", text: "For Gaussian $q$ and $p$, the divergence term has closed form $\\tfrac{1}{2} \\sum_{i} \\left( \\sigma_i^2 + \\mu_i^2 - 1 - \\log \\sigma_i^2 \\right)$." },
];

const AGENTS = [
  { key: "analyze", name: "Query Analyzer", job: "classify intent & expertise" },
  { key: "plan", name: "Planner", job: "choose minimal agent pipeline" },
  { key: "memory", name: "Memory", job: "resolve follow-up references" },
  { key: "retrieve", name: "Paper Retrieval", job: "hybrid search over chunks" },
  { key: "verify", name: "Evidence Verifier", job: "block answers lacking support" },
  { key: "math", name: "Math Knowledge", job: "fill prerequisite gaps" },
  { key: "generate", name: "Explanation Generator", job: "write grounded answer" },
  { key: "cite", name: "Citation Validator", job: "reject unsupported claims" },
];

const PIPELINES = {
  variable_lookup: ["analyze", "plan", "retrieve", "generate"],
  summary: ["analyze", "plan", "retrieve", "generate"],
  equation_explanation: ["analyze", "plan", "memory", "retrieve", "verify", "generate"],
  concept_comparison: ["analyze", "plan", "memory", "retrieve", "verify", "generate", "cite"],
  derivation: ["analyze", "plan", "memory", "retrieve", "verify", "generate", "cite"],
};

const SAMPLES = [
  "Why is KL divergence minimized in Equation (5)?",
  "What does lambda represent?",
  "Why use cross entropy instead of mean squared error?",
  "How is Equation (5) derived from the ELBO?",
];

// ---------------- lexical retrieval (LaTeX-aware tokenizer) ----------------
function tokenize(s) {
  return s.toLowerCase()
    .replace(/\\(mathcal|mathrm|mathbb|big|tfrac|sum|left|right|odot|sim|mid|geq|,|;)/g, " ")
    .replace(/\\([a-z]+)/g, " $1 ")            // \beta -> beta, \sigma -> sigma
    .replace(/[^a-z0-9_()]+/g, " ")
    .split(" ").filter(Boolean);
}
function retrieve(query, k = 5) {
  const q = tokenize(query);
  const scored = CORPUS.map((c) => {
    const t = new Set(tokenize(c.text));
    let s = 0;
    q.forEach((w) => { if (t.has(w)) s += w.length > 3 ? 2 : 1; });
    return { c, s };
  }).filter((x) => x.s > 0).sort((a, b) => b.s - a.s);
  return scored.slice(0, k).map((x) => x.c);
}

// ---------------- Claude call ----------------
async function ask(system, prompt) {
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-6", max_tokens: 1000,
      system, messages: [{ role: "user", content: prompt }],
    }),
  });
  const d = await r.json();
  return (d.content || []).filter((b) => b.type === "text").map((b) => b.text).join("\n");
}
function parseJSON(s) {
  try { return JSON.parse(s.replace(/```json|```/g, "").trim()); }
  catch { const m = s.match(/\{[\s\S]*\}/); return m ? JSON.parse(m[0]) : null; }
}

// ---------------- KaTeX loader + math renderer ----------------
function useKatex() {
  const [ready, setReady] = useState(!!window.katex);
  useEffect(() => {
    if (window.katex) return;
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css";
    document.head.appendChild(css);
    const js = document.createElement("script");
    js.src = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js";
    js.onload = () => setReady(true);
    document.head.appendChild(js);
  }, []);
  return ready;
}

// Splits text on $$display$$ and $inline$ delimiters and renders each math
// segment with KaTeX; plain segments stay as text nodes.
function MathText({ text, katexReady }) {
  if (!katexReady || !window.katex) return <span>{text}</span>;
  const parts = text.split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g);
  return (
    <>
      {parts.map((p, i) => {
        const display = p.startsWith("$$") && p.endsWith("$$");
        const inline = !display && p.startsWith("$") && p.endsWith("$") && p.length > 2;
        if (!display && !inline) return <span key={i}>{p}</span>;
        const tex = p.slice(display ? 2 : 1, display ? -2 : -1);
        let html;
        try {
          html = window.katex.renderToString(tex, { displayMode: display, throwOnError: false });
        } catch { return <span key={i}>{p}</span>; }
        return <span key={i} style={display ? { display: "block", margin: "10px 0", overflowX: "auto" } : undefined} dangerouslySetInnerHTML={{ __html: html }} />;
      })}
    </>
  );
}

// ---------------- component ----------------
export default function MathPaperDemo() {
  const katexReady = useKatex();
  const [question, setQuestion] = useState(SAMPLES[0]);
  const [status, setStatus] = useState({});
  const [answer, setAnswer] = useState("");
  const [running, setRunning] = useState(false);
  const [openChunk, setOpenChunk] = useState(null);
  const [evidence, setEvidence] = useState([]);
  const historyRef = useRef([]);

  const set = (key, state, note = "") =>
    setStatus((p) => ({ ...p, [key]: { state, note } }));

  async function run() {
    if (!question.trim() || running) return;
    setRunning(true); setAnswer(""); setStatus({}); setEvidence([]); setOpenChunk(null);
    let ev = [], external = [];
    try {
      set("analyze", "run");
      const a = parseJSON(await ask(
        'Classify a question about a research paper. Reply ONLY JSON: {"intent": one of ["variable_lookup","equation_explanation","concept_comparison","derivation","summary"], "expertise": one of ["beginner","undergraduate","researcher"]}',
        question)) || { intent: "equation_explanation", expertise: "undergraduate" };
      set("analyze", "done", `${a.intent} · ${a.expertise}`);

      set("plan", "run");
      const plan = PIPELINES[a.intent] || PIPELINES.equation_explanation;
      AGENTS.forEach((ag) => { if (!plan.includes(ag.key)) set(ag.key, "skip", "not needed for this intent"); });
      set("plan", "done", plan.filter((s) => !["analyze", "plan"].includes(s)).join(" → "));

      let resolved = question;
      if (plan.includes("memory")) {
        set("memory", "run");
        if (historyRef.current.length) {
          resolved = (await ask(
            "Rewrite the follow-up question so it is fully self-contained using the history. Reply with the question only.",
            `History: ${JSON.stringify(historyRef.current.slice(-4))}\n\nFollow-up: ${question}`)).trim();
          set("memory", "done", `resolved: "${resolved.slice(0, 60)}"`);
        } else set("memory", "done", "no prior turns — passthrough");
      }

      set("retrieve", "run");
      ev = retrieve(resolved);
      setEvidence(ev);
      set("retrieve", "done", ev.map((c) => c.id).join(", ") || "no hits");

      if (plan.includes("verify")) {
        for (let cycle = 0; cycle < 2; cycle++) {
          set("verify", "run");
          const v = parseJSON(await ask(
            'Decide if evidence suffices to answer without guessing (symbols defined? prerequisites present?). Reply ONLY JSON: {"sufficient": bool, "missing_concepts": []}',
            `Question: ${resolved}\nEvidence: ${JSON.stringify(ev.map((c) => c.text))}\nExternal: ${JSON.stringify(external.map((e) => e.text))}`)) || { sufficient: true, missing_concepts: [] };
          if (v.sufficient) { set("verify", "done", "evidence sufficient"); break; }
          set("verify", "done", `missing: ${v.missing_concepts.join(", ")}`);
          if (v.missing_concepts.length) {
            set("math", "run");
            for (const concept of v.missing_concepts.slice(0, 2)) {
              const def = await ask(
                "Give a precise 3-sentence textbook definition. Write all mathematics in LaTeX using $...$ for inline math and $$...$$ for the key formula on its own line.",
                concept);
              external.push({ concept, text: def });
            }
            set("math", "done", `${external.length} definition(s) fetched`);
          } else break;
        }
      }

      set("generate", "run");
      const out = await ask(
        `Explain for a ${a.expertise}-level reader using ONLY the evidence. Cite chunk ids inline like [chunk_3]. Write ALL mathematical notation in LaTeX: $...$ for inline math, $$...$$ for display equations on their own line. Never write math as plain text like sigma^2 — always LaTeX, e.g. $\\sigma^2$. If evidence is insufficient for a claim, say so. Keep it under 250 words.`,
        `Question: ${resolved}\nEvidence: ${JSON.stringify(ev)}\nBackground: ${JSON.stringify(external)}`);
      set("generate", "done");

      if (plan.includes("cite")) {
        set("cite", "run");
        const c = parseJSON(await ask(
          'Check every claim is supported by the cited chunks. Reply ONLY JSON: {"valid": bool, "unsupported": []}',
          `Answer: ${out}\nEvidence: ${JSON.stringify(ev)}`)) || { valid: true };
        set("cite", "done", c.valid ? "all claims supported" : `flagged: ${(c.unsupported || []).join("; ").slice(0, 80)}`);
      }

      setAnswer(out);
      historyRef.current.push({ q: resolved, a: out.slice(0, 400) });
    } catch (e) {
      setAnswer("Pipeline error: " + e.message);
    }
    setRunning(false);
  }

  // answer -> citation chips, math rendered inside text segments
  const renderAnswer = () => {
    const parts = answer.split(/(\[chunk_\d+\])/g);
    return parts.map((p, i) => {
      const m = p.match(/\[(chunk_\d+)\]/);
      if (m) return (
        <button key={i} onClick={() => setOpenChunk(openChunk === m[1] ? null : m[1])} style={st.chip}>
          {m[1].replace("chunk_", "§")}
        </button>
      );
      return <MathText key={i} text={p} katexReady={katexReady} />;
    });
  };
  const open = CORPUS.find((c) => c.id === openChunk);

  return (
    <div style={st.page}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');
        @keyframes pulse { 0%,100% { border-color:#e6c76d } 50% { border-color:#5f6f65 } }
        button:focus-visible { outline: 2px solid #e6c76d; outline-offset: 2px; }
        .katex { font-size: 1.05em; }
        @media (prefers-reduced-motion: reduce) { * { animation: none !important } }
      `}</style>

      <header style={st.header}>
        <div style={st.eyebrow}>seminar demo · agentic RAG</div>
        <h1 style={st.title}>MathPaper <em>AI</em></h1>
        <p style={st.sub}>Ask about the loaded paper — <span style={{ fontStyle: "italic" }}>"A VAE for Molecular Generation"</span> — and watch eight specialist agents decide, retrieve, verify, and answer. Skipped agents are struck through: that's dynamic orchestration. Equations render in LaTeX.</p>
      </header>

      <div style={st.inputRow}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="Ask about the paper…" style={st.input} disabled={running} />
        <button onClick={run} disabled={running} style={{ ...st.runBtn, opacity: running ? 0.5 : 1 }}>
          {running ? "working…" : "Ask"}
        </button>
      </div>
      <div style={st.samples}>
        {SAMPLES.map((s) => (
          <button key={s} style={st.sample} disabled={running} onClick={() => setQuestion(s)}>{s}</button>
        ))}
      </div>

      <section style={st.board}>
        {AGENTS.map((ag) => {
          const s = status[ag.key] || { state: "idle" };
          const style = {
            idle: { borderColor: "#3a4a40", color: "#5f6f65" },
            run: { borderColor: "#e6c76d", color: "#ece7da", animation: "pulse 1.1s infinite" },
            done: { borderColor: "#a3c6d8", color: "#ece7da" },
            skip: { borderColor: "#2b382f", color: "#4a5a50", textDecoration: "line-through" },
          }[s.state];
          return (
            <div key={ag.key} style={{ ...st.agent, ...style }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                <span style={{ fontWeight: 500 }}>{ag.name}</span>
                <span style={st.stateTag}>{s.state === "idle" ? "" : s.state}</span>
              </div>
              <div style={st.note}>{s.note || ag.job}</div>
            </div>
          );
        })}
      </section>

      {answer && (
        <section style={st.answer}>
          <div style={st.answerLabel}>Answer</div>
          <div style={st.answerBody}>{renderAnswer()}</div>
          {open && (
            <div style={st.chunkView}>
              <strong>{open.id}</strong> · {open.section} — <MathText text={open.text} katexReady={katexReady} />
            </div>
          )}
        </section>
      )}

      {evidence.length > 0 && !answer && (
        <div style={st.retrNote}>retrieved: {evidence.map((c) => c.id).join(" · ")}</div>
      )}
    </div>
  );
}

const st = {
  page: { minHeight: "100vh", background: "#16211b", color: "#ece7da", fontFamily: "'STIX Two Text', Georgia, serif", padding: "28px 16px 60px", maxWidth: 780, margin: "0 auto" },
  header: { borderBottom: "1px solid #3a4a40", paddingBottom: 18, marginBottom: 20 },
  eyebrow: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, letterSpacing: "0.14em", color: "#e6c76d", textTransform: "uppercase", marginBottom: 8 },
  title: { fontSize: 40, fontWeight: 600, margin: 0, lineHeight: 1.05 },
  sub: { color: "#a9b3a8", fontSize: 15, lineHeight: 1.55, marginTop: 10, maxWidth: 640 },
  inputRow: { display: "flex", gap: 8 },
  input: { flex: 1, background: "#1e2b24", border: "1px solid #3a4a40", color: "#ece7da", padding: "12px 14px", fontSize: 15, fontFamily: "inherit", borderRadius: 2 },
  runBtn: { background: "#e6c76d", color: "#16211b", border: "none", padding: "0 22px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 14, fontWeight: 500, cursor: "pointer", borderRadius: 2 },
  samples: { display: "flex", flexWrap: "wrap", gap: 6, margin: "10px 0 24px" },
  sample: { background: "transparent", border: "1px solid #3a4a40", color: "#a9b3a8", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, padding: "5px 10px", cursor: "pointer", borderRadius: 2, textAlign: "left" },
  board: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 8 },
  agent: { border: "1px solid", borderRadius: 2, padding: "10px 12px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, background: "#19251f", transition: "border-color .3s" },
  stateTag: { fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase", color: "#e6c76d" },
  note: { marginTop: 5, fontSize: 11, color: "#8a978c", lineHeight: 1.45, minHeight: 16 },
  answer: { marginTop: 26, borderTop: "1px solid #3a4a40", paddingTop: 18 },
  answerLabel: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: "#a3c6d8", marginBottom: 10 },
  answerBody: { fontSize: 16, lineHeight: 1.7, whiteSpace: "pre-wrap" },
  chip: { background: "#243329", border: "1px solid #a3c6d8", color: "#a3c6d8", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, padding: "1px 7px", margin: "0 3px", cursor: "pointer", borderRadius: 10, verticalAlign: "baseline" },
  chunkView: { marginTop: 14, border: "1px dashed #3a4a40", padding: "10px 12px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: "#a9b3a8", lineHeight: 1.6 },
  retrNote: { marginTop: 16, fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: "#8a978c" },
};
