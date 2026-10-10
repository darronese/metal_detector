import { cat, ones_like, PreTrainedModel, Tensor } from "@huggingface/transformers";
import { loadModel } from "./load/model";
import { loadTokenizer } from "./load/tokenizer";

type Inputs = {
  input_ids: Tensor,
  attention_mask: Tensor,
}

/**
 * My own forward pass function that
 *  1: grabs outputs from models
 *  2: reshapes it
 *  3: attach the winner and new attention mask of the same shape back into our original inputs
 *
 * @param model - our model we use
 * @param inputs - the inputs we will be appending to and pulling from
 * @returns winner token id, to see if we hit end of sequence
 */
async function forward_pass(model: PreTrainedModel, inputs: Inputs): Promise<bigint> {
  // (forward pass) no past_key_values given, so transformers.js feeds empty caches (same as use_cache=False)
  // input_ids [1, seq] -> logits [1, seq, vocab]
  const outputs: { logits: Tensor } = await model(inputs)

  // get into the last row, no negative indexing on Tensor so use seq - 1
  // argmax gives back a scalar int64 tensor holding the TOKEN id
  const seq = outputs.logits.dims[1]
  const winner = outputs.logits.slice(0, seq - 1).argmax()

  // reshape winner [] => [1, 1], so it can join the input_ids [1, seq] -> [1, seq + 1]
  const reshaped_winner = winner.view(1, 1)
  const new_attention_mask_input = ones_like(reshaped_winner)

  // cat allocates a brand new tensor each step, same expense as torch.cat
  // dim=1 to avoid adding another prompt dimension
  inputs.input_ids = cat([inputs.input_ids, reshaped_winner], 1)
  inputs.attention_mask = cat([inputs.attention_mask, new_attention_mask_input], 1)
  return winner.item() as bigint
}

async function main() {
  // load model specified by user: npm start -- <model_name> <prompt>
  const [model_name, user_prompt] = process.argv.slice(2);
  const model = await loadModel(model_name)
  const { tokenizer, apply_template, encode, decode } = await loadTokenizer(model_name)

  // apply template from prompt
  const prompt = apply_template(
    [
      {
        "role": "system",
        "content": "You are a sarcastic, playful assistant"
      },
      {
        "role": "user",
        "content": `${user_prompt}`
      }
    ]);

  const inputs: Inputs = encode(prompt)
  const prompt_length = inputs.input_ids.dims[1]
  for (let i = 0; i < 200; i++) {
    const output = await forward_pass(model, inputs)
    // token ids are int64 -> bigint, eos_token_id is a number
    if (output === BigInt(tokenizer.eos_token_id)) {
      break
    }
  }

  // only decode the newly generated tokens
  const ids = inputs.input_ids.tolist()[0] as bigint[]
  console.log(decode(ids.slice(prompt_length)))
}

main()
