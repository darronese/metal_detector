import type { Step } from "../engine/messages";
import { esc, fmt, heat, pct, plainText, tokenText, tokenTitle, type Tokens } from "./format";

// pure html builders: data in, markup out. no DOM access, no state: view.ts decides where the markup goes

function tokenCell(id: number, col: number, tokens: Tokens, cls = "") {
  const info = tokens.get(id);
  return `<div class="cell tok ${info?.special ? "special" : ""} ${cls}" data-col="${col}" title="${esc(tokenTitle(id, info))}">
    <span class="id">${id}</span><span class="txt">${esc(tokenText(info))}</span></div>`;
}

function grid(cols: number, body: string, cls = "") {
  return `<div class="grid ${cls}" style="--cols:${cols}">${body}</div>`;
}

function label(main: string, sub = "") {
  return `<div class="label"><span class="main">${main}</span>${sub ? `<span class="sub">${sub}</span>` : ""}</div>`;
}

/** the [1, seq] input_ids before a pass, plus an empty slot where the new token will go */
export function before(ids: number[], tokens: Tokens, sel: number, slot: boolean) {
  const n = ids.length;
  const cols = n + (slot ? 1 : 0);
  let html = label("position", "");
  for (let j = 0; j < cols; j++) html += `<div class="cell pos ${j === sel ? "sel" : ""}" data-col="${j}">${j}</div>`;
  html += label("<code>input_ids[0]</code>", `shape [1, ${n}]`);
  ids.forEach((id, j) => html += tokenCell(id, j, tokens, j === sel ? "sel" : ""));
  if (slot) html += `<div class="cell slot">?</div>`;
  return `<h2 class="block-title"><span class="num">1</span> input_ids <em>before</em> the forward pass</h2>` + grid(cols, html);
}

/** logits[0] drawn as positions (columns) x vocab ids (rows), with skipped id ranges collapsed */
export function logits(s: Step, tokens: Tokens, sel: number, enter: boolean) {
  const n = s.input_ids.length;
  const last = n - 1;
  let html = "";
  const gap = (from: number, to: number) => {
    if (to <= from) return;
    html += `<div class="label gap-label">⋮ <span class="sub">${fmt(to - from)} ids (${fmt(from)}–${fmt(to - 1)})</span></div><div class="gap"></div>`;
  };

  let prev = 0;
  s.rows.forEach((id, r) => {
    gap(prev, id);
    prev = id + 1;
    const info = tokens.get(id);
    html += `<div class="label vocab ${id === s.winner ? "win" : ""}" title="${esc(tokenTitle(id, info))}">
      <span class="id">${id}</span><span class="txt">${esc(tokenText(info))}</span></div>`;
    for (let j = 0; j < n; j++) {
      const p = s.probs[j][r];
      const l = s.logits[j][r];
      const cls = [j === last && "last", j === last && id === s.winner && "win", j === sel && "sel", enter && j === last && "enter"].filter(Boolean).join(" ");
      html += `<div class="cell heat ${cls}" data-col="${j}" style="--t:${heat(p).toFixed(3)}"
        title="logits[0, ${j}, ${id}] = ${l.toFixed(3)}\nsoftmax p = ${pct(p)}"><span class="logit">${l.toFixed(1)}</span><span class="prob">${pct(p)}</span></div>`;
    }
    html += `<div class="cell empty"></div>`;
  });
  gap(prev, s.vocab);

  return `<h2 class="block-title"><span class="num">2</span> <code>model(inputs)</code> → logits
      <span class="shape">[1, ${n}, ${fmt(s.vocab)}]</span>
      <span class="hint">showing <code>logits[0]</code>: the only batch. Columns are positions, rows are vocab ids.
      Column ${last} (highlighted) is the slice <code>logits[0, ${last}, :]</code>. Its argmax becomes the next token.</span></h2>`
    + grid(n + 1, html, "logits");
}

/** the [1, seq + 1] input_ids after cat */
export function after(s: Step, tokens: Tokens, sel: number, enter: boolean) {
  const n = s.input_ids.length;
  let html = label("<code>input_ids[0]</code>", `shape [1, ${n + 1}]`);
  s.input_ids.forEach((id, j) => html += tokenCell(id, j, tokens, `dim ${j === sel ? "sel" : ""}`));
  html += tokenCell(s.winner, n, tokens, `new ${s.eos ? "eos" : ""} ${enter ? "enter" : ""} ${n === sel ? "sel" : ""}`);
  return `<h2 class="block-title"><span class="num">3</span> <code>argmax</code> → <code>cat(dim=1)</code> → input_ids <em>after</em>
      <span class="shape">[1, ${n}] + [1, 1] → [1, ${n + 1}]</span></h2>` + grid(n + 1, html);
}

/** right-hand panel: what one position's row of logits predicts */
export function inspector(s: Step | null, tokens: Tokens, sel: number) {
  if (!s) return `<p class="muted">Once a pass finishes, click any column to see what that position predicted.</p>`;
  const n = s.input_ids.length;
  if (sel >= n) {
    return `<h3>position ${sel} <span class="tag new">new</span></h3>
      <div class="big-tok">${esc(tokenText(tokens.get(s.winner)))}</div>
      <p class="muted">id ${s.winner}: argmax of <code>logits[0, ${n - 1}, :]</code>, appended by <code>cat</code>.
      It has no logits yet; the next forward pass produces them.</p>`;
  }
  const col = s.columns[sel];
  const actual = sel + 1 < n ? s.input_ids[sel + 1] : s.winner;
  const at = tokens.get(s.input_ids[sel]);
  const rank = col.top.findIndex(c => c.id === actual);
  const maxP = col.top[0].prob;

  const rows = col.top.map((c, i) => `<li class="${i === 0 ? "argmax" : ""} ${c.id === actual ? "actual" : ""}">
      <span class="bar" style="--w:${(c.prob / maxP * 100).toFixed(1)}%"></span>
      <span class="txt" title="${esc(tokenTitle(c.id, tokens.get(c.id)))}">${esc(tokenText(tokens.get(c.id)))}</span>
      <span class="id">${c.id}</span><span class="p">${pct(c.prob)}</span><span class="l">${c.logit.toFixed(2)}</span></li>`).join("");

  const verdict = sel === n - 1
    ? `argmax → <b>${esc(tokenText(tokens.get(s.winner)))}</b> becomes position ${n}`
    : rank === 0
      ? `the prompt's next token <b>${esc(tokenText(tokens.get(actual)))}</b> was the model's top pick`
      : `the prompt's next token <b>${esc(tokenText(tokens.get(actual)))}</b> was ${rank > 0 ? `rank ${rank + 1}` : `not in the top ${col.top.length}`}`;

  return `<h3>position ${sel} ${sel === n - 1 ? "<span class=\"tag\">last</span>" : ""}</h3>
    <div class="big-tok ${at?.special ? "special" : ""}">${esc(tokenText(at))}</div>
    <p class="muted"><code>logits[0, ${sel}, :]</code> predicts position ${sel + 1}</p>
    <ol class="cands">${rows}</ol>
    <p class="verdict">${verdict}</p>
    <dl class="stats">
      <div><dt>entropy</dt><dd>${col.entropy.toFixed(2)} nats</dd></div>
      <div><dt>effective choices</dt><dd title="exp(entropy): roughly how many tokens it's torn between">${Math.exp(col.entropy).toFixed(1)}</dd></div>
      ${sel === n - 1 ? `<div><dt>p(stop)</dt><dd title="probability of any stop token here: how close the model came to ending">${pct(s.pStop)}</dd></div>` : ""}
      <div><dt>pass time</dt><dd>${s.ms.toFixed(0)} ms</dd></div>
    </dl>`;
}

/**
 * the generated text, one chip per token so you can see where each pass's addition starts and ends.
 * chip i was added by forward pass i; the one added by the pass being viewed is highlighted, later ones don't exist yet
 */
export function generated(s: Step | null, promptLength: number, tokens: Tokens, enter: boolean) {
  if (!s) return `<p class="muted">nothing yet. Each forward pass adds one token here.</p>`;
  const ids = [...s.input_ids.slice(promptLength), s.winner];
  const chips = ids.map((id, i) => {
    const info = tokens.get(id);
    const latest = i === ids.length - 1;
    // real text so it reads like the answer; special tokens and newlines get a visible marker
    const text = !info ? "…" : info.special ? info.raw || "∅" : info.text.replace(/\n/g, "↵\n");
    return `<span class="gen ${info?.special ? "special" : ""} ${latest ? "latest" : ""} ${latest && enter ? "enter" : ""}" data-step="${i}"
      title="${esc(`added by forward pass ${i + 1}\n${tokenTitle(id, info)}`)}">${esc(text)}</span>`;
  }).join("");
  const added = tokens.get(s.winner);
  return `<p class="gen-added">forward pass ${s.step + 1} added
      <span class="gen latest">${esc(tokenText(added))}</span> <span class="id">id ${s.winner}</span>
      ${s.eos ? "<span class=\"tag stop\">stop token: generation ends here</span>" : `<span class="p-stop" title="probability of any stop token at this pass">p(stop) ${pct(s.pStop)}</span>`}</p>
    <div class="gen-text">${chips}</div>`;
}

/** shown instead of the board while a run is in progress */
export function generating(passes: number, max: number, promptLength: number) {
  return `<div class="generating">
    <p><b>Generating…</b> forward pass ${passes + 1} <span class="muted">· prompt ${promptLength} tokens + ${passes} generated</span></p>
    <div class="meter"><div style="width:${(passes / max * 100).toFixed(1)}%"></div></div>
    <p class="muted">Every pass is being kept. The board appears when the run ends: a stop token, the ${max}-pass limit, or Stop.</p>
  </div>`;
}

// the three numbers the overview plots for each pass, all from the LAST position (the one argmax picks from)
const winnerP = (s: Step) => s.columns[s.columns.length - 1].top[0].prob;
const entropy = (s: Step) => s.columns[s.columns.length - 1].entropy;

/**
 * the whole run at a glance: one column per forward pass, three small-multiple strips sharing the pass axis.
 * separate strips, not one chart: the three measures have different scales (never two y-axes)
 */
export function overview(steps: Step[], view: number, running: boolean) {
  if (running) return `<p class="muted ov-empty">The run overview appears when the run is over.</p>`;
  if (!steps.length) return `<p class="muted ov-empty">After you Generate, every forward pass gets a column here: how sure the model was, how torn, and how close it came to stopping. Click one to jump to it.</p>`;

  const maxH = Math.max(...steps.map(entropy), 0.1);
  // which pass is the extreme of each strip: labelled in the strip's caption instead of on the bars
  const extreme = (f: (s: Step) => number, pick: (a: number, b: number) => boolean) =>
    steps.reduce((best, s, i) => pick(f(s), f(steps[best])) ? i : best, 0);
  const loP = extreme(winnerP, (a, b) => a < b);
  const hiH = extreme(entropy, (a, b) => a > b);
  const hiStop = extreme(s => s.pStop, (a, b) => a > b);

  const strips = [
    { label: "winner probability", scale: "0–100%", note: `lowest ${pct(winnerP(steps[loP]))} · pass ${loP + 1}`, cls: "", h: winnerP },
    { label: "entropy", scale: `0–${maxH.toFixed(1)} nats`, note: `highest ${entropy(steps[hiH]).toFixed(2)} · pass ${hiH + 1}`, cls: "", h: (s: Step) => entropy(s) / maxH },
    { label: "p(stop)", scale: "log scale, 1e-6–100%", note: `highest ${pct(steps[hiStop].pStop)} · pass ${hiStop + 1}`, cls: "stop", h: (s: Step) => heat(s.pStop) },
  ];

  const labels = strips.map(st => `<div><span>${st.label}</span><small class="scale">${st.scale}</small><small class="note">${st.note}</small></div>`).join("");
  const columns = steps.map((s, i) => {
    const aria = `pass ${i + 1}: winner ${pct(winnerP(s))}, entropy ${entropy(s).toFixed(2)} nats, p(stop) ${pct(s.pStop)}`;
    const cells = strips.map(st => `<span class="ov-cell ${st.cls}"><i style="height:${(st.h(s) * 100).toFixed(1)}%"></i></span>`).join("");
    return `<button class="ov-pass ${i === view ? "cur" : ""}" data-step="${i}" aria-label="${aria}">${cells}</button>`;
  }).join("");

  return `<div class="ov" style="--n:${steps.length}">
      <div class="ov-labels">${labels}</div>
      <div class="ov-plot ${steps.length > 120 ? "dense" : ""}">${columns}</div>
      <div class="ov-axis"><span>pass 1</span><span>pass ${steps.length}</span></div>
    </div>`;
}

/** hover card for one pass of the overview */
export function overviewTip(s: Step, tokens: Tokens) {
  return `<b>pass ${s.step + 1}</b> added <code>${esc(tokenText(tokens.get(s.winner)))}</code>
    <dl>
      <dt>winner probability</dt><dd>${pct(winnerP(s))}</dd>
      <dt>entropy</dt><dd>${entropy(s).toFixed(2)} nats</dd>
      <dt>p(stop)</dt><dd>${pct(s.pStop)}</dd>
    </dl>`;
}

/** every generated id of a finished run: what the last pass had, plus the token it appended */
export const generatedIds = (last: Step, promptLength: number) => [...last.input_ids.slice(promptLength), last.winner];

/**
 * the whole answer as readable text, whichever pass is being viewed
 *
 * @param steps - every pass of the run
 * @param promptLength - how many input_ids were the prompt
 * @param maxSteps - the pass limit, to say why it ended
 */
export function finalOutput(steps: Step[], promptLength: number, tokens: Tokens, maxSteps: number) {
  const last = steps.at(-1);
  if (!last) return "";
  const text = plainText(generatedIds(last, promptLength), tokens);
  const why = last.eos ? "ended with a stop token"
    : steps.length >= maxSteps ? `hit the ${maxSteps}-pass limit`
    : "stopped before the model finished";
  return `<p class="final-text">${text.trim() ? esc(text) : `<span class="muted">(only special tokens)</span>`}</p>
    <p class="final-meta">${steps.length} tokens · ${why}</p>`;
}
