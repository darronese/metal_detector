import { EngineClient } from "./engine/client";
import {
  beginGenerate, beginLoad, beginPass, createState, needsLoad, receive, select, stopGenerating, viewStep,
  type Run,
} from "./state";
import { bindEvents } from "./ui/events";
import { initView, render, setShowProb, setTemplateEnabled, showOverviewTip } from "./ui/view";

// entry point: connects the three parts and nothing else
//   ui/events  --intent-->  state  --render-->  ui/view
//   engine     --message--> state  --render-->  ui/view

const state = createState();
const engine = new EngineClient();

function startRun(run: Run) {
  beginGenerate(state);
  engine.start(run.prompt, run.options);
  render(state);
}

// generation = (load if needed) -> encode -> one pass after another until state says stop
engine.onMessage((msg) => {
  const effect = receive(state, msg);
  render(state, effect);
  if (effect.start) startRun(effect.start);
  if (effect.next) {
    beginPass(state);
    engine.step();
    render(state);
  }
});

/** go to a pass; false when there is nothing to step through */
function move(index: number) {
  const moved = viewStep(state, index);
  if (moved) render(state, { follow: true });
  return moved;
}

bindEvents({
  generate: (modelId, prompt, options) => {
    // the model picked in step 1 isn't loaded yet: load it, the run starts when it's ready
    if (needsLoad(state, modelId)) {
      beginLoad(state, modelId, { prompt, options });
      engine.load(modelId);
      render(state);
    } else startRun({ prompt, options });
  },
  stop: () => { stopGenerating(state); render(state); },
  select: (col) => { select(state, col); render(state); },
  // moving between passes changes the board's width: follow so the last column and new token stay in view
  scrub: (index) => { move(index); },
  shift: (delta) => move(state.view + delta),
  jump: (to) => move(to === "first" ? 0 : state.steps.length - 1),
  modelChanged: () => render(state),
  hover: (index) => showOverviewTip(state, index),
  showProb: setShowProb,
  template: setTemplateEnabled,
});

initView();
render(state);
