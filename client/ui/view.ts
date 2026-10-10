import { DEFAULT_SYSTEM } from "../../src/hood/prompt";
import { loadPrefs } from "../loadPrefs";
import { MODELS } from "../models";
import { canGenerate, currentStep, MAX_STEPS, selectedColumn, type State } from "../state";
import { dom } from "./dom";
import { after, before, generated, generating, inspector, logits, overview, overviewTip } from "./templates";

// state -> screen. reads State, writes the DOM, never changes State

export type RenderOptions = { enter?: boolean, follow?: boolean };

/** one-time setup of controls that start with a value */
export function initView() {
  dom.system.value = DEFAULT_SYSTEM;
  // first entry is selected by default: the small one, so a first visit isn't a 786 MB download
  dom.model.innerHTML = MODELS.map(m =>
    `<option value="${m.id}">${m.name} · ${m.mb.webgpu} MB · ${m.note}</option>`).join("");
}

export function render(s: State, { enter = false, follow = false }: RenderOptions = {}) {
  renderModel(s);
  renderControls(s);
  renderBoard(s, enter);
  renderTimeline(s);
  dom.status.textContent = s.status;
  if (follow) followLatest();
}

/** step 1: what's loaded, what will load, and the download */
function renderModel(s: State) {
  const { status, id, device, dtype } = s.model;
  const name = (modelId?: string) => MODELS.find(m => m.id === modelId)?.name ?? modelId ?? "";
  const pickedOther = status === "ready" && id !== dom.model.value;
  dom.model.disabled = status === "loading" || s.running;

  // the badge names the loaded model, so switching the dropdown can't look like it's already ready
  dom.badge.className = `badge ${pickedOther ? "" : device ?? ""}`;
  dom.badge.textContent =
    status === "loading" ? `loading ${name(id)}…`
    : pickedOther ? `${name(id)} loaded · this one loads on Generate`
    : status === "ready" ? `${name(id)} · ${device} · ${dtype}`
    : "not loaded";
  dom.badge.title = device === "webgpu" ? "running on the GPU" : device === "wasm" ? "no usable WebGPU, running on the CPU (wasm)" : "";

  // there's no clean way to abort a download in transformers.js, so say how to get out
  // plus why it's on the CPU (phone) or a forced device/dtype (test link), when that applies
  const { why } = loadPrefs();
  dom.modelHint.textContent = (status === "loading"
    ? "Downloading and loading. To cancel, reload the page."
    : "Downloads the first time you generate, then your browser keeps it.") + (why ? ` ${why}` : "");

  dom.progress.hidden = !s.progress;
  if (s.progress) {
    dom.progress.querySelector<HTMLElement>(".fill")!.style.width = `${s.progress.progress}%`;
    dom.progress.querySelector(".file")!.textContent = `${s.progress.file} ${s.progress.progress.toFixed(0)}%`;
  }
}

function renderControls(s: State) {
  dom.generate.disabled = !canGenerate(s);
  dom.generate.textContent = s.model.status === "loading" ? "Loading model…" : s.running ? "Generating…" : "Generate";
  dom.stop.disabled = !s.running;
}

function renderBoard(s: State, enter: boolean) {
  // step 3 is one intro line until there is a run to explore
  const hasRun = s.running || s.steps.length > 0;
  dom.explore.hidden = !hasRun;
  dom.exploreIntro.hidden = hasRun;

  // mid-run: only a progress card. everything else waits until the run is over
  if (s.running) {
    dom.scroller.innerHTML = generating(s.steps.length, MAX_STEPS, s.promptIds.length);
    dom.inspector.innerHTML = `<p class="muted">Available once the run is over.</p>`;
    dom.output.innerHTML = `<p class="muted">Generating… the text appears when the run is over.</p>`;
    return;
  }
  const step = currentStep(s);
  const ids = step ? step.input_ids : s.promptIds;
  const n = ids.length;
  const sel = selectedColumn(s);

  dom.scroller.innerHTML = !n
    ? `<p class="empty-state">Press <b>Generate</b>. Every forward pass is kept, so you can step through them afterwards.</p>`
    : before(ids, s.tokens, sel, !step) + (step ? logits(step, s.tokens, sel, enter) + after(step, s.tokens, sel, enter) : "");
  dom.inspector.innerHTML = inspector(step, s.tokens, sel);
  dom.output.innerHTML = generated(step, s.promptIds.length, s.tokens, enter);
}

function renderTimeline(s: State) {
  dom.overview.innerHTML = overview(s.steps, s.view, s.running);
  dom.overviewTip.hidden = true;
  const atStart = s.running || s.view <= 0, atEnd = s.running || s.view >= s.steps.length - 1;
  dom.first.disabled = dom.prev.disabled = atStart;
  dom.next.disabled = dom.last.disabled = atEnd;
  // classList.toggle only re-adds the class when it was off, so the pulse plays once each time Next lights up
  dom.next.classList.toggle("lit", !atEnd);
  dom.scrub.disabled = s.running || s.steps.length < 2;
  dom.scrub.max = String(Math.max(s.steps.length - 1, 0));
  dom.scrub.value = String(Math.max(s.view, 0));
  dom.scrubLabel.textContent = s.running ? "generating…" : s.steps.length ? `forward pass ${s.view + 1} of ${s.steps.length}` : "no passes yet";
}

/** keep the newest column and the newest generated token in view, without moving the page */
function followLatest() {
  // instant, not smooth: holding → shouldn't queue up scroll animations
  dom.scroller.scrollLeft = dom.scroller.scrollWidth;
  const text = dom.output.querySelector(".gen-text");
  if (text) text.scrollTop = text.scrollHeight;
}

/** purely visual, so it skips State */
export function setShowProb(on: boolean) {
  document.body.classList.toggle("show-prob", on);
}

/** the system message only exists inside the chat template */
export function setTemplateEnabled(on: boolean) {
  dom.system.disabled = !on;
}

/**
 * show the hover card above one overview column, or hide it
 *
 * @param s - state, for the step and its tokens
 * @param index - the hovered pass, null when the pointer leaves
 */
export function showOverviewTip(s: State, index: number | null) {
  const tip = dom.overviewTip;
  const column = index === null ? null : dom.overview.querySelector<HTMLElement>(`[data-step="${index}"]`);
  if (index === null || !column) { tip.hidden = true; return; }
  tip.innerHTML = overviewTip(s.steps[index], s.tokens);
  tip.hidden = false;
  // centered over the column, kept inside the section
  const box = tip.parentElement!.getBoundingClientRect();
  const col = column.getBoundingClientRect();
  const left = col.left + col.width / 2 - box.left - tip.offsetWidth / 2;
  tip.style.left = `${Math.max(8, Math.min(box.width - tip.offsetWidth - 8, left))}px`;
  tip.style.top = `${col.top - box.top - tip.offsetHeight - 6}px`;
}
