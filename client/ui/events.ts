import type { PromptOptions } from "../../src/hood/prompt";
import { dom } from "./dom";

// DOM events -> intents. reads what the user typed or clicked and hands it to a handler;
// what actually happens is main.ts's call

export type Handlers = {
  generate: (modelId: string, prompt: string, options: PromptOptions) => void
  stop: () => void
  select: (col: number) => void
  scrub: (index: number) => void
  // move through finished passes; false = nothing to move through (then keys keep their normal job)
  shift: (delta: number) => boolean
  jump: (to: "first" | "last") => boolean
  modelChanged: () => void
  copy: () => void
  // overview column under the pointer
  hover: (index: number | null) => void
  showProb: (on: boolean) => void
  template: (on: boolean) => void
};

/** the number in data-<key> of the clicked element or its ancestor, null if none */
function dataNumber(e: Event, key: "col" | "step"): number | null {
  const el = (e.target as HTMLElement).closest<HTMLElement>(`[data-${key}]`);
  return el ? Number(el.dataset[key]) : null;
}

export function bindEvents(h: Handlers) {
  dom.generate.onclick = () => h.generate(dom.model.value, dom.prompt.value, {
    template: dom.template.checked,
    system: dom.system.value.trim(),
  });
  dom.stop.onclick = () => h.stop();
  dom.first.onclick = () => h.jump("first");
  dom.prev.onclick = () => h.shift(-1);
  dom.next.onclick = () => h.shift(1);
  dom.last.onclick = () => h.jump("last");
  dom.scrub.oninput = () => h.scrub(Number(dom.scrub.value));
  dom.showProb.onchange = () => h.showProb(dom.showProb.checked);
  dom.template.onchange = () => h.template(dom.template.checked);
  dom.model.onchange = () => h.modelChanged();
  dom.copy.onclick = () => h.copy();

  // board cell -> inspect its column; generated chip or overview column -> jump to that pass
  const onNumber = (key: "col" | "step", act: (n: number) => void) => (e: Event) => {
    const n = dataNumber(e, key);
    if (n !== null) act(n);
  };
  dom.scroller.onclick = onNumber("col", h.select);
  dom.output.onclick = onNumber("step", h.scrub);
  dom.overview.onclick = onNumber("step", h.scrub);
  dom.overview.onpointerover = (e) => h.hover(dataNumber(e, "step"));
  dom.overview.onpointerleave = () => h.hover(null);

  // ← / → step through passes, Home / End jump to either end.
  // only when there are passes: otherwise the keys keep scrolling the page as usual
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    const handled =
      e.key === "ArrowLeft" ? h.shift(-1)
      : e.key === "ArrowRight" ? h.shift(1)
      : e.key === "Home" ? h.jump("first")
      : e.key === "End" ? h.jump("last")
      : false;
    if (handled) e.preventDefault();
  });
}
