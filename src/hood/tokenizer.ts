import { AutoTokenizer } from "@huggingface/transformers";

type Message = {
  role: string,
  content: string,
};

// what a vocab id "matches to": raw is the vocab entry (Ġthere), text is what it decodes to ( there)
export type TokenInfo = { raw: string, text: string, special: boolean };

export type Tokenizer = Awaited<ReturnType<typeof loadTokenizer>>;

/**
 * Function that loads tokenizer and gives an encode and decode function
 *
 * @param modelName - The name of model the user wants to use (must be a list from Transformers.js)
 * @returns an object with tokenizer, applyTemplate, encode, decode, and tokenInfo functions
 */
export async function loadTokenizer(modelName: string) {
  const tokenizer = await AutoTokenizer.from_pretrained(modelName);

  // the tokenizer only maps token -> id, flip it so any id (any row of the logits) can be shown
  const idToRaw: string[] = [];
  for (const [raw, id] of tokenizer.get_vocab()) idToRaw[id] = raw;
  const special = new Set(tokenizer.all_special_ids);

  // closure remembers tokenizer
  const applyTemplate = (messages: Message[]) => tokenizer.apply_chat_template(messages, {tokenize: false, add_generation_prompt: true}) as string;
  const encode = (text: string) => tokenizer(text, {add_special_tokens: false, return_tensor: true});
  const decode = (ids: number[] | bigint[]) => tokenizer.decode(ids, {clean_up_tokenization_spaces: false});
  // ids past the vocab are padding rows of the output layer: they get a logit but no token
  const tokenInfo = (id: number): TokenInfo => idToRaw[id] === undefined
    ? { raw: "", text: "", special: true }
    : { raw: idToRaw[id], text: decode([id]), special: special.has(id) };
  return { tokenizer, applyTemplate, encode, decode, tokenInfo, vocabEntries: idToRaw.length };
}
