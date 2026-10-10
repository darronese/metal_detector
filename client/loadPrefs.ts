import type { DataType } from "@huggingface/transformers";
import type { Device, LoadPrefs } from "./engine/messages";

// which device/dtype to ask the worker for, decided on the page because only the page can see
// the URL and the touch screen (the worker can't tell an iPad from a Mac)

const DEVICES: Device[] = ["webgpu", "wasm"];
const DTYPES: DataType[] = ["fp32", "fp16", "q8", "q4", "q4f16"];

/**
 * phones and tablets get the CPU by default: loading on their GPU can use more memory than
 * mobile browsers allow a tab, and iOS kills (reloads) the page when that happens
 */
function isMobile(): boolean {
  if ((navigator as any).userAgentData?.mobile) return true;
  const ua = navigator.userAgent;
  // iPadOS pretends to be a Mac; a touch screen gives it away
  return /iPhone|iPad|iPod|Android/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
}

/**
 * ?device=wasm|webgpu and ?dtype=q4|q4f16|q8|fp16|fp32 override everything (test links);
 * unknown values are ignored
 *
 * @returns what to send with the load message, and a sentence saying why (shown under step 1)
 */
export function loadPrefs(): { prefs: LoadPrefs, why: string } {
  const q = new URLSearchParams(location.search);
  const device = DEVICES.find(d => d === q.get("device"));
  const dtype = DTYPES.find(d => d === q.get("dtype"));

  if (device || dtype) {
    const what = [device && `device=${device}`, dtype && `dtype=${dtype}`].filter(Boolean).join(", ");
    return { prefs: { device, dtype }, why: `Using ${what} from the link.` };
  }
  if (isMobile()) {
    return { prefs: { device: "wasm" }, why: "On phones this uses the CPU: slower, but it needs less memory than the GPU path." };
  }
  return { prefs: {}, why: "" };
}
