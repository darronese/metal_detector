/**
 * math on ONE row of logits: the scores one position gives every vocab id, shape [vocab]
 * knows nothing about tensors, positions, or batches. the caller hands in the row
 */

// one vocab id and how much the model wanted it
export type Candidate = { id: number, logit: number, prob: number };

// what one position's row of logits predicts for the NEXT position
export type Column = { top: Candidate[], entropy: number };

/**
 * log(sum(exp(row))), the log of softmax's denominator, so any prob is exp(logit - lse)
 * max-subtracted first so exp never overflows (same trick as torch.logsumexp)
 *
 * @param row - one position's logits [vocab]
 * @returns the log of the softmax denominator
 */
export function logsumexp(row: Float32Array): number {
  let max = -Infinity;
  for (const x of row) if (x > max) max = x;
  let sum = 0;
  for (const x of row) sum += Math.exp(x - max);
  return max + Math.log(sum);
}

/**
 * softmax probability of a set of ids, added together
 *
 * @param row - one position's logits [vocab]
 * @param lse - logsumexp(row), passed in so it's computed once per row
 * @param ids - vocab ids to add up (one id -> that token's probability)
 * @returns total probability of those ids
 */
export function probability(row: Float32Array, lse: number, ids: Iterable<number>): number {
  let p = 0;
  for (const id of ids) p += Math.exp(row[id] - lse);
  return p;
}

/**
 * top-k candidates and the entropy of softmax(row), in one pass over the vocab
 *
 * @param row - one position's logits [vocab]
 * @param lse - logsumexp(row), MUST be from this same row or every prob is wrong (obviously)
 * @param k - how many candidates to keep
 * @returns the top k (best first, ties go to the lower id like argmax) and the entropy in nats
 */
export function summarizeRow(row: Float32Array, lse: number, k: number): Column {
  const top: Candidate[] = [];
  let entropy = 0;
  for (let id = 0; id < row.length; id++) {
    const logit = row[id];
    const logp = logit - lse;
    const prob = Math.exp(logp);
    // prob underflows to 0 for very unlikely ids, and 0 * log(0) would be NaN
    if (prob > 0) entropy -= prob * logp;

    // keep top sorted descending; only touch it when this logit beats the current k-th
    if (top.length < k || logit > top[top.length - 1].logit) {
      let i = top.length;
      while (i > 0 && top[i - 1].logit < logit) i--;
      top.splice(i, 0, { id, logit, prob });
      if (top.length > k) top.pop();
    }
  }
  return { top, entropy };
}
