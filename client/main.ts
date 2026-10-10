import { EngineClient } from './engine/client'
import { beginGenerate, beginLoad, beginPass, createState, receive, select, stopGenerating, viewStep } from './state'
import { bindEvents } from './ui/events'
import { initView, render, setShowProb, setTemplateEnabled, showOverviewTip } from './ui/view'

// entry point: connects the three parts and nothing else
//   ui/events  --intent-->  state  --render-->  ui/view
//   engine     --message--> state  --render-->  ui/view

const state = createState()
const engine = new EngineClient()

// generation = encode, then one pass after another until state says stop
engine.onMessage((msg) => {
  const effect = receive(state, msg)
  render(state, effect)
  if (effect.next) {
    beginPass(state)
    engine.step()
    render(state)
  }
})

bindEvents({
  load: (model_name) => { beginLoad(state); engine.load(model_name); render(state) },
  generate: (prompt, options) => { beginGenerate(state); engine.start(prompt, options); render(state) },
  stop: () => { stopGenerating(state); render(state) },
  select: (col) => { select(state, col); render(state) },
  // moving between passes changes the board's width: follow so the last column and new token stay in view
  scrub: (index) => { viewStep(state, index); render(state, { follow: true }) },
  shift: (delta) => { viewStep(state, state.view + delta); render(state, { follow: true }) },
  jump: (to) => { viewStep(state, to === 'first' ? 0 : state.steps.length - 1); render(state, { follow: true }) },
  hover: (index) => showOverviewTip(state, index),
  showProb: setShowProb,
  template: setTemplateEnabled,
})

initView()
render(state)
