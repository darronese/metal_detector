// every element the ui touches, looked up once. ids match index.html

const $ = <T extends HTMLElement>(id: string) => {
  const el = document.getElementById(id)
  if (!el) throw new Error(`index.html is missing #${id}`)
  return el as T
}

export const dom = {
  model: $<HTMLInputElement>('model'),
  load: $<HTMLButtonElement>('load'),
  badge: $('badge'),
  progress: $('progress'),
  prompt: $<HTMLTextAreaElement>('prompt'),
  template: $<HTMLInputElement>('template'),
  system: $<HTMLInputElement>('system'),
  generate: $<HTMLButtonElement>('generate'),
  stop: $<HTMLButtonElement>('stop'),
  first: $<HTMLButtonElement>('first'),
  prev: $<HTMLButtonElement>('prev'),
  next: $<HTMLButtonElement>('next'),
  last: $<HTMLButtonElement>('last'),
  overview: $('overview'),
  overview_tip: $('overview-tip'),
  scroller: $('scroller'),
  inspector: $('inspector'),
  scrub: $<HTMLInputElement>('scrub'),
  scrub_label: $('scrub-label'),
  show_prob: $<HTMLInputElement>('show-prob'),
  output: $('output'),
  status: $('status'),
}
