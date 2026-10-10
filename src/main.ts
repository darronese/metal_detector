import { loadModel } from "./hood/model";
import { loadTokenizer } from "./hood/tokenizer";
import { forwardPass, stopTokens, type Inputs } from "./hood/generate";
import { buildPrompt } from "./hood/prompt";

async function main() {
  // load model specified by user: npm start -- <modelName> <prompt>
  const [modelName, userPrompt] = process.argv.slice(2);
  const model = await loadModel(modelName);
  const tok = await loadTokenizer(modelName);
  const { tokenizer, encode, decode } = tok;

  // template: false -> raw prompt, no roles or system message (often it will not close off on its own, requires a special end token from prompt instructions)
  const inputs: Inputs = encode(buildPrompt(tok, userPrompt, { template: false }));
  const promptLength = inputs.input_ids.dims[1];
  const stop = stopTokens(model, tokenizer.eos_token_id);
  for (let i = 0; i < 200; i++) {
    const { winner } = await forwardPass(model, inputs);
    // token ids are int64 -> bigint, stop ids are numbers
    if (stop.has(Number(winner))) {
      break;
    }
  }

  // only decode the newly generated tokens
  const ids = inputs.input_ids.tolist()[0] as bigint[];
  console.log(decode(ids.slice(promptLength)));
}

main();
