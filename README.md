# metal_detector

## Looking at logits, one forward pass at a time

Interactive tool I built based on my understanding from how causal language models predict and generate tokens

Instead of seeing the final generated text, you can inspect logits produced under each forward pass, which tokens the model considered, and understand how the next token was selected

Goal was to build something concrete to make sure I really understood autoregressive generation by visualizing and architecting the process here, and maybe help other people who don't want to suffer as much

## Boring stuff I learned that no-one cares about (copied from `shovel`)

### Logits and Model Outputs

The general process of a autoregressive model looks like this:

`Text → Tokenizer → Token IDs → Transformer → Hidden States → LM Head → Logits → Next Token`

### Probabilities and Token Selection

- With autoregressive language models, the method used to select the next token from the probability distribution is important.
- **Greedy decoding** selects the highest-scoring token at each step. While simple, it can lead to repetitive or poor results over longer sequences.
- **Sampling** randomly selects a token based on the probability distribution instead of always choosing the highest-scoring token. This can produce more diverse outputs.
- **Beam search** keeps track of multiple possible sequences instead of committing to one token sequence at each step.
- Different decoding methods can produce different outputs even when using the same model and input prompt.

### Autoregressive Generation

- Language models generate text by repeatedly predicting and selecting the next token.
- The selected token becomes part of the input sequence for the next prediction.
- This process continues until the model reaches a stopping condition.
- Hugging Face's `model.generate()` handles this process automatically, but custom generation loops can be implemented to better understand how token selection works.

### How This Relates to Metal Detector

Metal Detector focuses on understanding what happens between passing tokens into a language model and selecting the next token.

Rather than only looking at the final generated response, the project focuses on examining the logits and probability distributions produced during generation. This helps illustrate how different tokens are scored and how the decoding strategy determines which token gets selected next.
