import type { PromptOptions } from "../../src/hood/prompt";
import type { Request, Response } from "./messages";

/**
 * The page's handle on the engine: starts the worker and gives typed methods instead of raw postMessage.
 * Nothing outside engine/ knows a Worker is involved.
 */
export class EngineClient {
  private worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });

  private send(msg: Request) {
    this.worker.postMessage(msg);
  }

  onMessage(handler: (msg: Response) => void) {
    this.worker.onmessage = (e: MessageEvent<Response>) => handler(e.data);
  }

  load(modelName: string) { this.send({ type: "load", modelName }); }
  start(prompt: string, options: PromptOptions) { this.send({ type: "start", prompt, options }); }
  step() { this.send({ type: "step" }); }
}
