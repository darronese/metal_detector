import { AutoTokenizer } from '@huggingface/transformers'

type Message = {
  role: string,
  content: string,
}

/**
 * Function that loads tokenizer and gives an encode and decode function
 *
 * @param model_name - The name of model the user wants to use (must be a list from Transformers.js)
 * @param prompt - The prompt the user inputs
 * @returns an object with tokenizer, apply_template, encode, and decode functions 
 */
export async function loadTokenizer(model_name: string) {
  const tokenizer = await AutoTokenizer.from_pretrained(model_name);

  // closure remembers tokenizer
  const apply_template = (messages: Message[]) => tokenizer.apply_chat_template(messages, {tokenize: false, add_generation_prompt: true}) as string
  const encode = (text: string) => tokenizer(text, {add_special_tokens: false, return_tensor: true})
  const decode = (ids: number[] | bigint[]) => tokenizer.decode(ids, {clean_up_tokenization_spaces: false})
  return { tokenizer, apply_template, encode, decode }
}
