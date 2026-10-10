import { DEFAULT_SYSTEM } from '../../src/hood/prompt'
import { canGenerate, currentStep, MAX_STEPS, selectedColumn, type State } from '../state'
import { dom } from './dom'
import { after, before, generated, generating, inspector, logits, overview, overview_tip } from './templates'

// state -> screen. reads State, writes the DOM, never changes State

export type RenderOptions = { enter?: boolean, follow?: boolean }

/** one-time setup of controls that start with a value */
export function initView() {
  dom.system.value = DEFAULT_SYSTEM
}

export function render(s: State, { enter = false, follow = false }: RenderOptions = {}) {
  renderHeader(s)
  renderControls(s)
  renderBoard(s, enter)
  renderTimeline(s)
  dom.status.textContent = s.status
  if (follow) followLatest()
}

function renderHeader(s: State) {
  const { status, device, dtype } = s.model
  dom.load.disabled = status === 'loading'
  dom.badge.className = `badge ${device ?? ''}`
  dom.badge.textContent = status === 'ready' ? `${device} · ${dtype}` : status === 'loading' ? 'loading…' : 'not loaded'
  dom.badge.title = device === 'webgpu' ? 'running on the GPU' : device === 'wasm' ? 'no usable WebGPU, running on the CPU (wasm)' : ''

  dom.progress.hidden = !s.progress
  if (s.progress) {
    dom.progress.querySelector<HTMLElement>('.fill')!.style.width = `${s.progress.progress}%`
    dom.progress.querySelector('.file')!.textContent = `${s.progress.file} ${s.progress.progress.toFixed(0)}%`
  }
}

function renderControls(s: State) {
  dom.generate.disabled = !canGenerate(s)
  dom.generate.textContent = s.running ? 'Generating…' : 'Generate'
  dom.stop.disabled = !s.running
}

function renderBoard(s: State, enter: boolean) {
  // mid-run: only a progress card. everything else waits until the run is over
  if (s.running) {
    dom.scroller.innerHTML = generating(s.steps.length, MAX_STEPS, s.prompt_ids.length)
    dom.inspector.innerHTML = `<p class="muted">Available once the run is over.</p>`
    dom.output.innerHTML = `<p class="muted">Generating… the text appears when the run is over.</p>`
    return
  }
  const step = currentStep(s)
  const ids = step ? step.input_ids : s.prompt_ids
  const n = ids.length
  const sel = selectedColumn(s)

  dom.scroller.innerHTML = !n
    ? `<p class="empty-state">Load a model, then <b>Generate</b>. Every forward pass is kept, so you can step through them afterwards.</p>`
    : before(ids, s.tokens, sel, !step) + (step ? logits(step, s.tokens, sel, enter) + after(step, s.tokens, sel, enter) : '')
  dom.inspector.innerHTML = inspector(step, s.tokens, sel)
  dom.output.innerHTML = generated(step, s.prompt_ids.length, s.tokens, enter)
}

function renderTimeline(s: State) {
  dom.overview.innerHTML = overview(s.steps, s.view, s.running)
  dom.overview_tip.hidden = true
  const at_start = s.running || s.view <= 0, at_end = s.running || s.view >= s.steps.length - 1
  dom.first.disabled = dom.prev.disabled = at_start
  dom.next.disabled = dom.last.disabled = at_end
  // classList.toggle only re-adds the class when it was off, so the pulse plays once each time Next lights up
  dom.next.classList.toggle('lit', !at_end)
  dom.scrub.disabled = s.running || s.steps.length < 2
  dom.scrub.max = String(Math.max(s.steps.length - 1, 0))
  dom.scrub.value = String(Math.max(s.view, 0))
  dom.scrub_label.textContent = s.running ? 'generating…' : s.steps.length ? `forward pass ${s.view + 1} of ${s.steps.length}` : 'no passes yet'
}

/** keep the newest column and the newest generated token in view, without moving the page */
function followLatest() {
  // instant, not smooth: holding → shouldn't queue up scroll animations
  dom.scroller.scrollLeft = dom.scroller.scrollWidth
  const text = dom.output.querySelector('.gen-text')
  if (text) text.scrollTop = text.scrollHeight
}

/** purely visual, so it skips State */
export function setShowProb(on: boolean) {
  document.body.classList.toggle('show-prob', on)
}

/** the system message only exists inside the chat template */
export function setTemplateEnabled(on: boolean) {
  dom.system.disabled = !on
}

/**
 * show the hover card above one overview column, or hide it
 *
 * @param s - state, for the step and its tokens
 * @param index - the hovered pass, null when the pointer leaves
 */
export function showOverviewTip(s: State, index: number | null) {
  const tip = dom.overview_tip
  const column = index === null ? null : dom.overview.querySelector<HTMLElement>(`[data-step="${index}"]`)
  if (index === null || !column) { tip.hidden = true; return }
  tip.innerHTML = overview_tip(s.steps[index], s.tokens)
  tip.hidden = false
  // centered over the column, kept inside the section
  const box = tip.parentElement!.getBoundingClientRect()
  const col = column.getBoundingClientRect()
  const left = col.left + col.width / 2 - box.left - tip.offsetWidth / 2
  tip.style.left = `${Math.max(8, Math.min(box.width - tip.offsetWidth - 8, left))}px`
  tip.style.top = `${col.top - box.top - tip.offsetHeight - 6}px`
}
