/**
 * Eager model initialisation — call once at app startup before navigating
 * to the main screen. Awaiting the returned promise ensures all three models
 * (NLU, embed, agent) are ready before any user interaction begins, eliminating
 * the C4 startup race where startTurn() fires before initEmbed() completes.
 */

import { initNlu, setNluPath }     from './nlu';
import { initEmbed, setEmbedPath } from './embed';
import { initAgent, setAgentPath } from './agent';

export interface ModelPaths {
  nlu:   string;  // Llama 3.2-1B-Instruct Q4_K_M .gguf path
  embed: string;  // nomic-embed-text-v1.5 .gguf path
  agent: string;  // Gemma 4-2B-IT Q4_K_M .gguf path
}

/**
 * Register model paths without loading — each model loads lazily on first use
 * via ensureNlu / ensureEmbed / ensureAgent. Call once after download completes
 * or on startup when cached paths exist.
 */
// expo-file-system returns file:// URIs; llama.rn native needs bare filesystem paths.
function toNativePath(uri: string): string {
  return uri.startsWith('file://') ? uri.slice(7) : uri;
}

export function setModelPaths(paths: ModelPaths): void {
  setNluPath(toNativePath(paths.nlu));
  setEmbedPath(toNativePath(paths.embed));
  setAgentPath(toNativePath(paths.agent));
}

/**
 * Eagerly initialise all three on-device models in sequence.
 * Only use this when you want them all pre-loaded (e.g. benchmarking).
 * Sequential (not parallel) to avoid OOM on constrained devices.
 */
export async function initModels(paths: ModelPaths): Promise<void> {
  await initNlu(paths.nlu);
  await initEmbed(paths.embed);
  await initAgent(paths.agent);
}
