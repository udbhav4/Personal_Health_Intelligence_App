# Gliimr: Fully On-Device Mental Health Inference with a DBN + Gemma 4 Hybrid

**How a three-model pipeline and Loopy Belief Propagation deliver probabilistic health state tracking on a mid-range Android phone with zero network calls**

---

## What Gliimr Does | Why It Exists

Gliimr is a React Native (Expo) mental and physical health companion that runs entirely on-device. The app maintains a calibrated probabilistic model of user health state by fusing two evidence streams: passive sensor readings collected every 15 minutes (pedometer, screen usage, communication logs, sleep proxy) and self-reported data extracted from natural-language conversation. It generates a Doctor Report synthesising 180 days of evidence as a clinical handoff document.

No health data leaves the device. This is a hard architectural constraint enforced at the data layer: there is no remote API call in the sensor collection pipeline, no telemetry backend, no cloud inference endpoint. Every architectural decision in the stack follows from the requirement that inference must execute entirely within the memory budget and compute ceiling of a mid-range Android device. Gemma 4 2B-IT (Q4_K_M GGUF) is one of three models in the on-device pipeline that makes this possible.

---

## Three-Model Architecture and Memory Budget

Three GGUF models run on-device:

1. **Llama 3.2-1B-Instruct Q4_K_M (~800 MB)** — NLU layer: intent classification, undo detection, named entity recognition. Runs first on every turn and is released before Gemma loads.
2. **nomic-embed-text-v1.5 Q4_K_M** — Memory compression: vectorises old conversation turns into embeddings stored in SQLite via sqlite-vec. Recency-weighted cosine similarity retrieval uses log-linear hybrid decay. One of the few embedding models published as GGUF (sentence-transformers and ONNX require runtimes absent from Hermes); ~270 MB, 768-dim vectors; managed by the same singleton lifecycle as the other two models.
3. **Gemma 4-E2B-IT Q4_K_M (~1.7 GB)** — Reasoning agent: executes the ReAct loop, structured synthesis, and the 7-call Doctor Report pipeline.

Total footprint (~3 GB) exceeds mid-range Android RAM (4–8 GB) once OS overhead is included. Sequential singleton loading applied: Llama releases before Gemma allocates, a 300 ms OS-reclaim pause follows, and Gemma is held as a singleton across turns to avoid its 3–5 second cold-start.

The three-model split is not an aesthetic choice. Approximately 60% of messages are social, undo operations, or acknowledgements. Routing these through Gemma costs a 1.7 GB allocation and a 3–5 second cold-start per "thanks." Llama 3.2-1B handles all NLU tasks instantly. Gemma is never loaded unless synthesis is required.

Gemma 4 E2B-IT is the only instruction-tuned model in the 1.5–2.5 GB GGUF range that reliably follows structured output formats — JSON tool calls and label-delimited synthesis tokens — across a 6144-token context window. The full context block (recent conversation pairs, memory summaries, DBN belief window, changed nodes, hypothesis, user message) reaches 5,000+ tokens in Reflect mode. Smaller models truncate evidence; larger models do not fit on device alongside Llama.


## The DBN: Why Not Just use Gemma

The health state model is a Dynamic Bayesian Network with three node categories: objective sensor input nodes (pedometer, screen proxy, communication, etc.); self-reported EMA input nodes (stress, pain, loneliness, etc.); latent inferred nodes (mental stress, physical stress). Conditional Probability Distribution (CPD) tables are bundled as `cpd-tables.json`.

Offline training used Phased Structural EM (LBP as E-step, three phases 40%->70%->100%) on NHANES (population priors), StudentLife (primary CPT training), and LifeSnaps (wearable sensor validation). The runtime inference engine is a TypeScript LBP port of the Python training server's `_loopy_bp_beliefs()`. `applyInterSlice()` applies inter-slice transition matrices so health state persists and decays across time without requiring user re-entry.

LBP was chosen over Variable Elimination because LBP scales linearly with edges while VE scales exponentially with treewidth — infeasible for a 37-node graph in a mobile TypeScript runtime.

The DBN exists because an LLM cannot maintain a calibrated probability distribution over health states across days. The CPDs are trained on NHANES, StudentLife, and LifeSnaps — thousands of subjects with clinical measurements. Gemma's implicit health priors derive from general text corpora, not clinical survey populations. Gemma handles structural understanding and natural-language generation from the DBN posterior and the vectorDB. These responsibilities are kept strictly separate.


## Gemma 4 in the Conversational Pipeline

Every user turn passes through a deterministic routing pipeline before Gemma is loaded. The acknowledgement-only fast path uses a static word-set check — if the turn matches, it resolves with a deterministic string and no model runs.

For non-trivial turns: Llama performs undo detection, intent classification, segmentation and NER. If the intent requires an undo, an intelligently handled specific data replacement/deletion is done. Intent classification categorizes user text into one or more from - reporting, social interaction or querying. Extracted entities are written to the `user_data_sensorless` SQLite table, to be fed into the DBN inference engine.

The MCP tool `run_dbn_inference` calls `inferenceEngine.ts` -> `runLBP()` -> a `BeliefResult` posterior. `selectBeliefWindow()` filters this posterior to the context budget: 15 nodes at confidence gate 0.35 for queries, 8 nodes at gate 0.55 for reports. Llama is then released, the OS reclaim pause executes, and Gemma loads with `n_ctx 6144`.

**Glance mode** is a single Gemma completion capped at 400 tokens. **Reflect mode** runs in two phases. Phase A: `runReactHypothesisLoop()` executes a max-3-step ReAct loop — `get_changed_nodes`, `get_user_memory`, then `get_belief_trend` batched across changed nodes, then a `HYPOTHESIS:` output. Phase B: `generator()` runs structured synthesis, producing ACK / CAUSE / EFFECT / LINK / SOLUTION / QUESTION labelled output.

Every tool call in the ReAct loop routes through an intelligent `dispatchTool()`, the single authoritative dispatch function. If Gemma generates a tool name not in the registered MCP set, `dispatchTool()` returns an error observation string — it does not execute. This is a hard TypeScript boundary, not a prompt-level instruction. The official MCP TypeScript SDK was not used: its stdio transport calls `child_process.spawn`, a Node.js built-in absent from Hermes and blocked by Android's process sandbox.

`runReactHypothesisLoop()` is sequential rather than batched because Gemma selects which nodes to probe with `get_belief_trend` after reading `get_user_memory` output. Batching requires the model to commit to relevance judgments without evidence. At 2B parameters, node selection accuracy is materially better after the memory read.

Gemma receives the DBN posterior as a belief window, past posterior trends, semantic memory summaries, and recent conversation. It never processes raw health sensor values. It reasons over the posterior to form a hypothesis, then synthesises that hypothesis into structured natural language. Glance completes in approximately 10 seconds. Reflect streams its first token in approximately 1 second and completes in approximately 30 seconds.


## Doctor Report Pipeline

Firstly, `detectHiddenPatterns()` classifies 180 days of sensor and DBN evidence into Tier 1 (symptom-linked patterns) and Tier 2 (secondary patterns). The Doctor Report is a 7-call sequential Gemma 4 pipeline, after algorithmic inference of importance. Call 2 runs `runReportHypothesisLoop()`: a max-6-step sequential ReAct loop reading changed DBN nodes through `get_changed_nodes`, `get_user_memory` over a 180-day window, followed by 4–5 `get_belief_trend` calls (one node per call), terminating at the `OBSERVATION:` stop token, producing a 3–5 sentence HYPOTHESIS. Calls 4–8 are section executors — one Gemma call per section at 200 tokens each.

`applyValidators()` post-processes every executor output, enforcing data citations, "forgotten"/"silent" framing language, and "limitations" language. The pipeline is sequential because Gemma cannot commit to a node list before reading the 180-day memory store. The `OBSERVATION:` stop-token loop is a deterministic boundary enforced in TypeScript.

---

## Passive Sensing and the Sleep Proxy

Four collectors run on a 15-minute background task: `clockCollector` (always-on, no permission required), `activityCollector` (expo-pedometer → active_ratio), `screenUsageCollector` (Android UsageStats API), and `communicationCollector` (CallLog + SMS). Sleep duration is estimated as the longest continuous screen-off window between 8 PM and 11 AM. This value populates the `sleep_quality` DBN node, weighted with lower confidence than direct self-report.

Daily sleep questions are reliably abandoned; the screen proxy requires no user action and is never surfaced as clinical data — it is one weighted signal in the DBN posterior.

---

## Challenges Overcome

The ~3 GB combined model footprint on a 4–8 GB device required sequential singleton loading, explicit Llama release, and the 300 ms OS reclaim pause. Android OS eviction under memory pressure is handled by `ensureAgent()` and `ensureNlu()` — if either context is evicted, the singleton reinitialises transparently from its stored file path on the next call.

The pgmpy Python DBN library does not run in React Native. Hermes has no Python runtime; Android's sandbox blocks fork/exec, ruling out a Python sidecar; bundling CPython + pgmpy + numpy would add 150–300 MB to the APK. The entire LBP inference engine was ported to TypeScript: `runLBP()`, `applyInterSlice()`, `contractLast()`, and `contractAxis()` were each validated against the Python implementation for numerical equivalence.

Training and runtime discretisation thresholds must match exactly — a mismatch silently corrupts evidence entering the DBN. This was solved by exporting `population_norm_stats.json` from the preprocessing pipeline and baking it into the app bundle; thresholds are never fetched at runtime.

Gemma 4 2B occasionally deviates from label-delimited format in long sequences. This is mitigated by the `dispatchTool()` hard boundary (hallucinated tool names return error observations), strict label parsing in `generator()`, and stop-token termination of the ReAct loop at `OBSERVATION:`.

Since DBN and chat data will be less in the initial few days, the model has been explicitly programed to not hallucinate advice.