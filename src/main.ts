import { loadModel } from "./hood/model";
import { loadTokenizer } from "./hood/tokenizer";
import { forward_pass, type Inputs } from "./hood/generate";

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

  const inputs: Inputs = encode(user_prompt)
  const prompt_length = inputs.input_ids.dims[1]
  for (let i = 0; i < 200; i++) {
    const { winner } = await forward_pass(model, inputs)
    // token ids are int64 -> bigint, eos_token_id is a number
    if (winner === BigInt(tokenizer.eos_token_id)) {
      break
    }
  }

  // only decode the newly generated tokens
  const ids = inputs.input_ids.tolist()[0] as bigint[]
  console.log(decode(ids.slice(prompt_length)))
}

main()
