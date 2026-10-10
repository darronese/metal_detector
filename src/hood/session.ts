import type { PreTrainedModel, Tensor } from '@huggingface/transformers'
import { forward_pass, stop_tokens, type Inputs } from './generate'
import { logsumexp, probability, summarize_row, type Column } from './logits'
import type { Tokenizer } from './tokenizer'

// vocab rows kept per pass (top-k of the last position)
const ROWS = 8
// candidates listed per position
const TOP_K = 5

// everything needed to look at one forward pass, small enough to send anywhere (no tensors)
export type Step = {
  // which forward pass, 0-based
  step: number,
  // input_ids[0] BEFORE the pass, length = seq
  input_ids: number[],
  // logits.dims[2]
  vocab: number,
  // vocab ids kept, ascending (top-k of the last position)
  rows: number[],
  // [seq][rows.length] logits[0, pos, rows[r]]
  logits: number[][],
  // [seq][rows.length] softmax over the full vocab, at those ids
  probs: number[][],
  // [seq] per-position summary
  columns: Column[],
  // argmax of logits[0, seq - 1, :], appended by cat
  winner: number,
  // winner is one of the model's stop tokens
  eos: boolean,
  // softmax probability of ANY stop token at the last position: how close it came to ending
  p_stop: number,
  // forward pass time
  ms: number,
}

/**
 * The only place that knows the logits layout: [1, seq, vocab] flattened, batch 0 only.
 *
 * @param logits - the raw output of one forward pass
 * @returns one zero-copy view per position, each [vocab]
 */
function position_rows(logits: Tensor): Float32Array[] {
  const [, seq, vocab] = logits.dims
  // fp16 models hand back float16 logits, the math in logits.ts wants float32
  const data = (logits.type === 'float32' ? logits : logits.to('float32')).data as Float32Array
  // subarray is a view into the same memory, not a copy
  return Array.from({ length: seq }, (_, pos) => data.subarray(pos * vocab, (pos + 1) * vocab))
}

/**
 * One prompt being generated one forward pass at a time, keeping a summary of each pass's logits.
 * Uses the same forward_pass as the terminal
 */
export class Session {
  private inputs: Inputs | null = null
  private step = 0
  // causal attention: position j only sees tokens <= j, so re-running the same prefix gives the same
  // row (up to float noise). cache each position's summary instead of re-scanning 150k ids every pass
  private lse_cache: number[] = []
  private column_cache: Column[] = []
  // every id that means "done", from the tokenizer AND generation_config
  private stop: Set<number>

  /**
   * @param model - loaded model, runs every forward pass
   * @param tok - loaded tokenizer, encodes the prompt
   */
  constructor(private model: PreTrainedModel, private tok: Tokenizer) {
    this.stop = stop_tokens(model, tok.tokenizer.eos_token_id)
  }

  /**
   * Encode the prompt and reset everything. No forward pass yet.
   *
   * @param text - the already built prompt (see build_prompt)
   * @returns input_ids[0] of the prompt
   */
  start(text: string): number[] {
    this.inputs = this.tok.encode(text)
    this.step = 0
    this.lse_cache = []
    this.column_cache = []
    return this.current_ids()
  }

  /**
   * Exactly one forward pass, summarized.
   *
   * @returns what went in, what came out, and what nearly won
   */
  async next(): Promise<Step> {
    if (!this.inputs) throw new Error('Session.next() called before start()')

    // forward_pass appends the winner to inputs, so read the ids first
    const input_ids = this.current_ids()
    const t0 = performance.now()
    const { winner, logits: tensor } = await forward_pass(this.model, this.inputs)
    const ms = performance.now() - t0

    const rows = position_rows(tensor)
    const last = rows.length - 1
    this.summarize_new_positions(rows)

    // vocab rows to draw: the last position's top candidates, in vocab order like the real tensor
    const kept = this.column_cache[last].top.slice(0, ROWS).map(c => c.id).sort((a, b) => a - b)
    const w = Number(winner)
    return {
      step: this.step++,
      input_ids,
      vocab: tensor.dims[2],
      rows: kept,
      logits: rows.map(row => kept.map(id => row[id])),
      probs: rows.map((row, pos) => kept.map(id => probability(row, this.lse_cache[pos], [id]))),
      columns: this.column_cache.map(c => ({ ...c, top: c.top.slice(0, TOP_K) })),
      winner: w,
      eos: this.stop.has(w),
      p_stop: probability(rows[last], this.lse_cache[last], this.stop),
      ms,
    }
  }

  /**
   * @returns input_ids[0] as plain numbers (the tensor holds int64 -> bigint)
   */
  private current_ids(): number[] {
    return (this.inputs!.input_ids.tolist()[0] as bigint[]).map(Number)
  }

  /**
   * Scan only positions not seen before; earlier ones come from the cache.
   *
   * @param rows - this pass's logits, one view per position
   */
  private summarize_new_positions(rows: Float32Array[]) {
    for (let pos = this.lse_cache.length; pos < rows.length; pos++) {
      this.lse_cache[pos] = logsumexp(rows[pos])
      this.column_cache[pos] = summarize_row(rows[pos], this.lse_cache[pos], Math.max(ROWS, TOP_K))
    }
  }
}
