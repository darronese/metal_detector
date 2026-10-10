import type { PromptOptions } from '../../src/hood/prompt'
import { dom } from './dom'

// DOM events -> intents. reads what the user typed or clicked and hands it to a handler;
// what actually happens is main.ts's call

export type Handlers = {
  load: (model_name: string) => void
  generate: (prompt: string, options: PromptOptions) => void
  stop: () => void
  select: (col: number) => void
  scrub: (index: number) => void
  shift: (delta: number) => void        // move through finished passes by this many
  jump: (to: 'first' | 'last') => void
  hover: (index: number | null) => void   // overview column under the pointer
  showProb: (on: boolean) => void
  template: (on: boolean) => void
}

export function bindEvents(h: Handlers) {
  dom.load.onclick = () => h.load(dom.model.value.trim())
  dom.generate.onclick = () => h.generate(dom.prompt.value, {
    template: dom.template.checked,
    system: dom.system.value.trim(),
  })
  dom.stop.onclick = () => h.stop()
  dom.first.onclick = () => h.jump('first')
  dom.prev.onclick = () => h.shift(-1)
  dom.next.onclick = () => h.shift(1)
  dom.last.onclick = () => h.jump('last')
  dom.scrub.oninput = () => h.scrub(Number(dom.scrub.value))
  dom.show_prob.onchange = () => h.showProb(dom.show_prob.checked)
  dom.template.onchange = () => h.template(dom.template.checked)

  // click any cell to inspect its column
  dom.scroller.onclick = (e) => {
    const cell = (e.target as HTMLElement).closest<HTMLElement>('[data-col]')
    if (cell) h.select(Number(cell.dataset.col))
  }

  // click a generated token to jump to the forward pass that added it
  dom.output.onclick = (e) => {
    const chip = (e.target as HTMLElement).closest<HTMLElement>('[data-step]')
    if (chip) h.scrub(Number(chip.dataset.step))
  }

  // overview: click a column to jump to that pass, hover for its numbers
  dom.overview.onclick = (e) => {
    const col = (e.target as HTMLElement).closest<HTMLElement>('[data-step]')
    if (col) h.scrub(Number(col.dataset.step))
  }
  dom.overview.onpointerover = (e) => {
    const col = (e.target as HTMLElement).closest<HTMLElement>('[data-step]')
    h.hover(col ? Number(col.dataset.step) : null)
  }
  dom.overview.onpointerleave = () => h.hover(null)

  // ← / → step through passes, Home / End jump to either end
  document.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return
    if (e.key === 'ArrowLeft') h.shift(-1)
    else if (e.key === 'ArrowRight') h.shift(1)
    else if (e.key === 'Home') h.jump('first')
    else if (e.key === 'End') h.jump('last')
    else return
    e.preventDefault()
  })
}
