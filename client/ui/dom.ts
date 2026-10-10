// every element the ui touches, looked up once. ids match index.html

const $ = <T extends HTMLElement>(id: string) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`index.html is missing #${id}`);
  return el as T;
};

export const dom = {
  model: $<HTMLSelectElement>("model"),
  badge: $("badge"),
  modelHint: $("model-hint"),
  progress: $("progress"),
  prompt: $<HTMLTextAreaElement>("prompt"),
  template: $<HTMLInputElement>("template"),
  system: $<HTMLInputElement>("system"),
  generate: $<HTMLButtonElement>("generate"),
  stop: $<HTMLButtonElement>("stop"),
  first: $<HTMLButtonElement>("first"),
  prev: $<HTMLButtonElement>("prev"),
  next: $<HTMLButtonElement>("next"),
  last: $<HTMLButtonElement>("last"),
  overview: $("overview"),
  overviewTip: $("overview-tip"),
  scroller: $("scroller"),
  inspector: $("inspector"),
  scrub: $<HTMLInputElement>("scrub"),
  scrubLabel: $("scrub-label"),
  showProb: $<HTMLInputElement>("show-prob"),
  output: $("output"),
  final: $("final"),
  copy: $<HTMLButtonElement>("copy"),
  status: $("status"),
  explore: $("explore"),
  exploreIntro: $("explore-intro"),
};
