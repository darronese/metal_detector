import { loadModel } from "./hood/model";
import { loadTokenizer } from "./hood/tokenizer";
import { forward_pass, stop_tokens, type Inputs } from "./hood/generate";
import { build_prompt } from "./hood/prompt";

async function main() {
  // load model specified by user: npm start -- <model_name> <prompt>
  const [model_name, user_prompt] = process.argv.slice(2);
  const model = await loadModel(model_name)
  const tok = await loadTokenizer(model_name)
  const { tokenizer, encode, decode } = tok

  // template: false -> raw prompt, no roles or system message (often it will not close off on its own, requires a special end token from prompt instructions)
  const inputs: Inputs = encode(build_prompt(tok, user_prompt, { template: false }))
  const prompt_length = inputs.input_ids.dims[1]
  const stop = stop_tokens(model, tokenizer.eos_token_id)
  for (let i = 0; i < 200; i++) {
    const { winner } = await forward_pass(model, inputs)
    // token ids are int64 -> bigint, stop ids are numbers
    if (stop.has(Number(winner))) {
      break
    }
  }

  // only decode the newly generated tokens
  const ids = inputs.input_ids.tolist()[0] as bigint[]
  console.log(decode(ids.slice(prompt_length)))
}

main()
