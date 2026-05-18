# Medical Health App — Complete System Design

**Version 8.0 — DBN Retraining Completion + Full Runtime Pipeline Hardening**

**Environment:** Google Colab (DBN training) + VS Code (all other Python) + React Native Expo (mobile frontend)

---

## Table of Contents

1. [What is new in v6.0](#what-is-new-in-v60)
2. [What is new in v7.0](#what-is-new-in-v70)
3. [What is new in v7.1](#what-is-new-in-v71)
4. [What is new in v7.2](#what-is-new-in-v72)
5. [What is new in v7.3](#what-is-new-in-v73)
6. [What is new in v7.4](#what-is-new-in-v74)
6a. [What is new in v7.5](#what-is-new-in-v75)
6b. [What is new in v7.6](#what-is-new-in-v76)
6c. [What is new in v7.7](#what-is-new-in-v77)
6d. [What is new in v7.8](#what-is-new-in-v78)
6e. [What is new in v7.9](#what-is-new-in-v79)
6f. [What is new in v8.0](#what-is-new-in-v80)
7. [System Philosophy](#1-system-philosophy)
8. [Phase 0 — Development Environment Setup](#2-phase-0--development-environment-setup)
9. [Datasets — Selection, Roles, and Limitations](#3-datasets--selection-roles-and-limitations)
10. [Dataset Roles — Precise Separation](#35-dataset-roles--precise-separation)
11. [Data Cleaning — Per Dataset](#36-data-cleaning--per-dataset)
12. [Column Pruning and Validated Scale Merges](#365-column-pruning-and-validated-scale-merges)
13. [Statistical Preprocessing — Per Dataset](#37-statistical-preprocessing--per-dataset)
14. [Data Harmonisation — Multi-Dataset Training Architecture](#38-data-harmonisation--multi-dataset-training-architecture)
15. [DBN Node Taxonomy — Three Layers](#4-dbn-node-taxonomy--three-layers)
16. [Node Training Source Categories](#45-node-training-source-categories)
17. [Manual Edge Injection for Self-Report Nodes](#46-manual-edge-injection-for-self-report-nodes)
18. [Training vs Runtime — The Full Separation](#5-training-vs-runtime--the-full-separation)
19. [Sub-Dimension User Profile Attribute Layer](#55-sub-dimension-user-profile-attribute-layer)
20. [Rolling Inference Window and Dual Trigger Architecture](#56-rolling-inference-window-and-dual-trigger-architecture)
21. [Window-Aware Training Pipeline](#57-window-aware-training-pipeline--new--v72)
22. [K-Means Discretisation Pipeline](#58-k-means-discretisation-pipeline--new--v73)
23. [DBN Training — Phased Structural EM with Expert Knowledge](#59-dbn-training--phased-structural-em-with-expert-knowledge--new--v74)
24. [High-Weight Node Derivation from CPT](#6-high-weight-node-derivation-from-cpt)
21. [Data Source Flag and Confidence System](#7-data-source-flag-and-confidence-system)
22. [Onboarding — Radical Simplification](#8-onboarding--radical-simplification)
23. [Extended Temporal Flag System](#9-extended-temporal-flag-system)
24. [Evidence Fusion — Self-Report vs Passive Priority](#10-evidence-fusion--self-report-vs-passive-priority)
25. [Contextual Awareness — Chat vs Dashboard](#11-contextual-awareness--chat-vs-dashboard)
26. [Delta Inference — Contextual Insight Engine](#12-delta-inference--contextual-insight-engine)
27. [Two-Direction Insight Generation](#13-two-direction-insight-generation)
28. [Pre-Insight Proactive Question System](#14-pre-insight-proactive-question-system)
28b. [Question Cascade — Post-NER Guided Interview](#14b-question-cascade--post-ner-guided-interview)
29. [Insight Button Logic](#15-insight-button-logic)
30. [Daily Notification Design](#16-daily-notification-design)
31. [Turn Routing — Stage Pipeline](#17-turn-routing--stage-pipeline)
32. [Hybrid Node Structure — When to Separate vs Merge](#18-hybrid-node-structure--when-to-separate-vs-merge)
33. [Confidence Gates — Three Gates + Confidence Check](#19-confidence-gates--three-gates--confidence-check)
34. [Hybrid Two-Layer Insight Generation](#20-hybrid-two-layer-insight-generation)
34b. [Agentic Architecture — Four Patterns](#20b-agentic-architecture--four-patterns)
35. [SQLite Schema](#21-sqlite-schema)
36. [JSON Session State Schema](#22-json-session-state-schema)
37. [Prompt Templates](#23-prompt-templates)
38. [Alert System](#24-alert-system)
39. [Dashboard Design](#25-dashboard-design)
40. [Doctor PDF Report](#26-doctor-pdf-report)
41. [Evaluation Strategy](#27-evaluation-strategy)
42. [Critical Build Order](#28-critical-build-order)
43. [Architecture Decision Log](#29-architecture-decision-log)
44. [Future Scope](#30-future-scope)

---

## What is new in v6.0

- *Passive-first philosophy formalised: sensors before questions always. Among the confirmed initial Layer 2 nodes, pain_level is the clearest example of a node with no passive proxy. As more self-report nodes are added (one question = one node), any node with no passive proxy qualifies for a proactive question when it is a high-weight contributor. The proactive question rule is proxy-driven, not pain-specific.*
- *Data source flag system: every node carries `data_source` and `confidence`, computed from `source_weight x recency x proxy_strength`. Insight language adapts to confidence level.*
- *Two-direction insight: every report generates both effects (downstream delta) and causes (upstream parents). Not just stress percentage.*
- *Confidence-gated insight: nodes with confidence below threshold are excluded from insight generation or flagged with lower-certainty language.*
- *Persistence follow-up: when user volunteers a statement, one follow-up question determines `temporal_flag` — today/week/daily — solving the expiry ambiguity.*
- *Daily notification redesign: one question per day at user-set time, optional follow-up expansion, skip without penalty, 7-day suppression per node.*
- *Hybrid node structure: separate nodes when questions measure distinct psychological dimensions; merge only when one plain-English question covers both constructs.*
- *Notification count added as passive `social_activity` proxy — counts app notifications to capture WhatsApp/Signal calls missed by call_log.*
- *Onboarding simplified to age + sex + notification time preference only. Everything else inferred passively or collected organically through chat.*
- *Pre-insight proactive question extended to all intent classes, not just `stress_query`. High-weight unfilled parent nodes trigger one question before any insight.*
- *Training vs runtime separation fully documented: survey columns (PHQ-9, PSS, BigFive etc.) shape CPTs during Structural EM training and disappear. Runtime nodes consist of all Layer 1 passive nodes, all Layer 2 self-report nodes (count TBD — finalised through hybrid node structure analysis), and the 2 latent Layer 3 nodes. Survey items are never asked of the user.*
- *Structural EM with Loopy Belief Propagation explained: E-step fills latent node values, M-step finds optimal structure via HillClimbSearch. Survey columns encoded into CPT then discarded.*

---

## What is new in v7.0

- *Two-layer personalisation architecture: Coarse DBN (14-16 nodes, safe parameter budget) + Sub-Dimension User Profile Attributes (unlimited, stored in SQLite). Survey scale items become runtime attribute nodes, not just training columns.*
- *Sub-dimension attributes solve the personalisation vs overfitting tradeoff: PHQ-9 items, PSS items, loneliness sub-dimensions, PANAS items each become queryable user profile attributes updated by pointed questions over time. They aggregate into coarse DBN evidence via evidence fusion — DBN stays small, personalisation grows progressively.*
- *Progressive personalisation pipeline: Week 1-2 pure passive observation. Week 3+ contextual pointed questions triggered by user statements. Daily notification rotates through unanswered sub-dimension attributes. Model personalises over weeks without overwhelming the user.*
- *Expiry per attribute type: stable traits (neuroticism) = 90-180 days. Chronic states (loneliness) = 14-30 days. Weekly patterns (PSS) = 7 days. Daily states (PANAS) = 1-3 days. Acute symptoms = 1 day. All configurable in `feature_node_config.json`.*
- *`app_usage numRunning` added as cognitive load proxy: daily mean and evening peak of running app tasks captures mental busyness signal distinct from screen_time.*
- *All survey scale items restored to training data: PHQ-9 all 9 items, PSS all 10 items, PANAS 14 items (removed only Proud/Determined/Guilty/Hostile — no DBN node mapping), all 20 loneliness items, all 11 VR-12 items. Removing items would introduce bias. EM decides redundancy for CPT training; sub-dimension layer decides what gets asked at runtime.*

---

## What is new in v7.1

- *Rolling inference window replaces daily batch: DBN inference fires every N minutes (configurable via `inference_interval_minutes` in `feature_node_config.json`) rather than once per day. Passive sensor evidence reflects the current window; self-report evidence carries forward from SQLite until expiry.*
- *Dual inference trigger: (1) Timer trigger — fires every N minutes, updates passive nodes from latest sensor window, runs inference silently, writes snapshot to SQLite. (2) Chat trigger — fires immediately when user reports something, merges new self-report into current passive snapshot, runs inference and returns insight. Both triggers call the same `build_dbn_evidence()` and inference pipeline.*
- *DBN temporal edges span both passive and self-report nodes across slices: `mood(t) → mood(t+1)`, `sleep_duration(t) → mental_stress(t+1)`. At each timer tick, the previous slice posterior becomes the prior for the next slice via temporal edges. Mid-window chat updates produce a new snapshot that becomes the settled state of the current slice, which the next timer tick conditions on.*
- *`inference_trigger` field added to session state and SQLite snapshots: values are `"timer"`, `"self_report"`, or `"manual"`. Allows audit trail of what caused each inference pass.*
- *SQLite `daily_snapshots` redesigned: `date PRIMARY KEY` replaced by `(date, snapshot_time)` composite key. `trigger_type` column added. Multiple snapshots per day are now the expected behaviour, not an error.*
- *Recency decay now sub-daily: confidence recency component decays at the granularity of `inference_interval_minutes`, not days.*
- *Dashboard intra-day timeline added: shows how stress and node beliefs evolved within the day as a time series, not just a single daily aggregate.*

---

## What is new in v7.2

- *Window-aware training pipeline: `data_harmon.py` refactored to produce one row per `(uid, date, window_start)` instead of one row per `(uid, date)`. A single `WINDOW_MINUTES` constant controls granularity across the entire pipeline — 15, 60, 360, or 1440 minutes. All sensing loaders, daily aggregates, and the final merge key are updated consistently.*
- *Sensor-alive architecture replaces quality filters: daily heuristics (`dark_entries >= 3`, `unlock_span_h >= 6`) are removed. Whether a window is alive or dead is determined at window granularity using always-on validator sensors (activity accelerometer, wifi, bluetooth, gps, audio). Behavioural sensors with zero readings in an alive window store `0` (genuine inactivity), not NaN. Dead windows store NaN and are excluded from DBN training.*
- *Window-level screen time: `screen_time_window_minutes` derived per window slot from dark sensing (daytime) and phonelock sensing (all hours). Night slots (21:00-07:00) use phonelock only since dark sensing is unreliable in dark rooms. Daytime slots cross-validate both sensors.*
- *EMA fall-through carry-forward: EMA responses are state declarations, not point events. Each response fills its window slot and carries forward to all subsequent slots within the same day until a newer response arrives. Day boundary is hard — carry-forward never crosses midnight. EMA columns are never subject to the alive/dead rule.*
- *Nighttime carry-forward: `nighttime_active_minutes` and `nighttime_unlock_minutes` (20:00-02:00 window) are computed as daily aggregates and joined to the following day's window rows as `prev_night_active_minutes` and `prev_night_unlock_minutes`.*
- *`raw_date_text` field for backdated self-reports: when a user says "I drank a lot last Wednesday" or "I was in pain on 1st April", the NER SLM extracts the exact date phrase as spoken (`raw_date_text`). TypeScript resolves the phrase to an ISO date string using `chrono-node` on-device — so the SLM never performs calendar arithmetic and any natural language date expression is handled correctly. Backdated reports are written to SQLite with the correct date, and a retroactive snapshot recomputation pass replays inference forward from that date to today through temporal edges.*
- *`raw_text` forwarded from NER to insight generator: the user's exact symptom phrase (e.g. "pain in eyes and redness") is forwarded to the insight generator for linguistic specificity. The DBN still does all formal reasoning; `raw_text` influences language only. This preserves the two-brain architecture invariant.*

---

## What is new in v7.3

- *Dataset roles fully clarified and contradictions removed: NHANES contributes prior tables only — never enters the training CSV. StudentLife is the primary CPT training source. LifeSnaps is promoted from validation-only to a secondary training source for wearable-derived nodes (heart rate, Fitbit sleep, Fitbit activity) that StudentLife does not have. LifeSnaps also remains the CPT validation source for sleep and energy nodes.*
- *Node training source categories formalised: four categories established — StudentLife-only nodes, LifeSnaps-only nodes, harmonised nodes (StudentLife + LifeSnaps concatenated), and NHANES-prior-informed nodes. Each category has distinct preprocessing and training pipeline rules.*
- *Data harmonisation pipeline specified: harmonised nodes require distribution overlap check before concatenation. If distributions align → min-max normalise within each dataset then concatenate. If distributions diverge → z-score within each dataset then concatenate. Discretisation thresholds always fit on StudentLife training uids only, then applied identically to LifeSnaps rows and at runtime.*
- *Manual edge injection formalised for self-report nodes: `smoking`, `alcohol_use`, and `pain_level` exist in StudentLife as survey columns and their CPT edges are learned by Structural EM directly. If any self-report node lacks corresponding columns in both training datasets, domain-knowledge edges are injected manually into pgmpy before EM runs. An isolated node with no edges is inert in a Bayesian network — it cannot affect inference regardless of its state.*
- *Data cleaning steps specified per dataset from actual data inspection: NHANES sentinel codes, BMI corruption, StudentLife sleep_hours broadcast bug, LifeSnaps age join failure, and cross-dataset physiological bound violations all documented with exact fix instructions.*
- *Statistical preprocessing pipeline specified: cleaning → train/val split → per-uid winsorsation → within-uid z-score (continuous sensors) → min-max normalisation (survey scales) → survey forward-fill within uid → distribution overlap check → harmonised column creation → discretisation threshold fitting on train split only.*
- *NHANES prior table construction specified: weighted frequency computation per node state per age/sex stratum, saved as JSON lookup. Applied at runtime to initialise node priors for new users before any self-report evidence exists.*

---

## What is new in v7.4

- *Code module reorganisation: all preprocessing modules moved to `Codebase/training/statistical_data_preprocessing/`, all training modules in `Codebase/training/model_training/`. Each folder has a `main.py` orchestrator that runs steps in order.*
- *Column pruning and validated scale merges formalised: PHQ-9 items → `phq_total` (sum 9 items, 0–27), PSS-10 → `pss_total` (reverse-score items 4, 5, 7, 8 before summing, 0–40), PANAS → `panas_pa` (9 positive affect items) + `panas_na` (9 negative affect items), UCLA-20 → `lonely_total`, BigFive → `extraversion` + `neuroticism`. Statistical drops per dataset documented in Section 3.65.*
- *Two-track discretisation: clinical/validated thresholds used where established cutoffs exist (PHQ, PSS, BMI, sleep_hours, smoking); K-Means data-driven for all behavioural and sensor columns. For shared nodes across datasets, K-Means cutoffs derived from the larger dataset are applied to the other.*
- *True Loopy BP (max-product) confirmed as E-step inference method: messages passed on the original DAG without junction tree construction, forward and backward, until convergence. Chosen over VariableElimination because LBP scales linearly with edges while VE scales exponentially with treewidth for the 37-node graph.*
- *Likelihood tables for soft evidence: `data_likelihood_tables.py` computes P(observed_bin | node_state) for each node — injected as soft evidence during LBP E-step instead of hard-assigning discrete states to training rows.*
- *Phased Structural EM: Phase 1 (40% data, HC 200 steps, perturbation after), Phase 2 (70% data, HC 300 steps, perturbation after), Phase 3 (100% data, HC 500 steps, no perturbation). Structure warmed up on data subset before full-data refinement.*
- *Expert knowledge enforcement: forced edges (domain knowledge, cannot be removed or reversed by HillClimbSearch), forbidden edges (all dynamic→static edges except forced ones), per-node max-parents cap. Startup validation checks forced edges form no cycle and forced ∩ forbidden is empty.*
- *`_sanitize_structure()` guard: called before every HillClimbSearch invocation — strips any reverse-of-forced edge that HC may have added, then re-enforces all forced edges. Prevents pgmpy HC from exploiting BIC score gains by reversing required causal directions mid-search.*
- *`_hard_kick()` plateau escape: when structure is unchanged for consecutive iterations in phases 1 or 2, 8 random non-forced edges are stripped before the next HC call. Breaks out of local optima that perturbation alone cannot escape.*
- *`visualize_bn.py` updated: sfdp graphviz layout for organic brain-like cluster, dark background (`#1a1a2e`), white forced edges, dashed light-grey learned edges, orange `FancyArrowPatch` arcs for temporal self-loops. Accepts `--struct` path argument; output PNG auto-named from input JSON path.*

---

## What is new in v7.5

- *Four agentic patterns introduced: (A) ReAct Agent for insight generation — multi-step tool-calling loop replacing fixed prompt templates, runs on-device via Gemma 4-2B-IT; (B) Reflection Loop — self-critique pass before every insight reaches the user, catches overclaiming on low-confidence nodes; (C) Plan-and-Execute Agent for doctor brief — model writes a structured plan first, then executes each step independently; (D) Memory-Augmented Agent — `user_profile_attributes` table extended with an on-device semantic vector layer enabling relevance-based memory retrieval in addition to structured node-name lookup.*
- *`UserMemory` class introduced: wraps SQLite structured store and an on-device vector store behind a unified interface. Two retrieval modes: `retrieve_structured(node)` for exact node lookups with expiry check; `retrieve_relevant(context_string)` for semantic similarity search across all stored observations. All data stays on-device.*
- *On-device embedding model introduced: a lightweight sentence embedding model (ONNX format, CPU-only, bundled in app assets) converts user observation text to dense vectors at storage and retrieval time. No text leaves the device.*
- *Self-report semantic deduplication added: before inserting a new `self_report` row, the embedding of `raw_text` is compared against existing embeddings in the on-device vector store. High cosine similarity triggers a confidence update on the existing record rather than a duplicate insert. Prevents Layer 2 context from being polluted with near-identical phrasings of the same symptom.*
- *Layer 2 retrieval upgraded from time-window to semantic search: the 7-day cutoff on `summary_text` retrieval is replaced by vector similarity search over all stored summaries. Relevance, not recency, determines what context is surfaced.*
- *Trend computation tool added to the ReAct agent tool registry: `get_trend(node, days)` fits a linear slope over the last N days of DBN posteriors, returns direction (rising/stable/falling) and magnitude. Enables temporal framing in insights — "mental stress has been rising for 5 days" rather than treating each snapshot in isolation.*
- *Two-model on-device setup formalised: a smaller, faster model handles NER extraction, intent routing, and storage-time summarisation. Gemma 4-2B-IT handles ReAct agent reasoning, reflection loop, and doctor brief planning.*
- *Model selection finalised: NER/intent model — Llama 3.2-1B-Instruct (Q4\_K\_M GGUF); agent/reasoning model — Gemma 4-2B-IT (Q4\_K\_M GGUF). Both run on-device via `react-native-llama.cpp`.*
- *FastAPI server role narrowed: hosts only the DBN inference endpoint. pgmpy is Python-only and cannot run in React Native — this is the sole reason FastAPI exists at runtime. No user health data passes through the server for any agentic operation.*
- *Agentic build order added to Section 28 as Phase 4.*

---

## What is new in v7.6

- *Two-pass NER segmentation: Llama 3.2-1B first splits a multi-claim user message into individual health claim segments (e.g. "I was really stressed today and drank coffee to feel good" → ["I was really stressed today", "drank coffee to feel good"]). A separate NER call then runs per segment, guaranteeing one entity per call. Eliminates the unreliable multi-entity extraction behaviour of a single NER call on compound messages.*
- *`raw_date_text` replaces `report_date` in NER output: the SLM now extracts the exact date phrase as spoken (e.g. "last Wednesday", "1st April") as `raw_date_text`. TypeScript resolves the phrase to an ISO date string using `chrono-node` on-device. The SLM never performs calendar arithmetic — any natural language date expression the user can type is handled correctly.*
- *`chrono-node` added to mobile package: resolves arbitrary natural language temporal expressions to ISO dates entirely on-device. No backend call needed for date resolution.*
- *KV cache strategy updated: the segmentation prompt is pre-warmed (not the NER prompt). Segmentation always runs first on every message and busts the NER cache anyway — warming the seg prompt eliminates the one always-cold call.*
- *`deduplicateEntities`: across-segment deduplication using key = `original_col ?? source_col ?? node_name`. When the same node is extracted from multiple segments (e.g. a compound sentence repeated the symptom), the higher-confidence entity wins. Result: exactly one row per node per message.*
- *`NluResult.unmatched` field: health-relevant text that has no column or node match (e.g. "I drank a lot of coffee today") is captured as a `string | null` field on the NluResult. This is the connector between the 1B NER pass and the 2B inference call — the unmatched content triggers a background Gemma call to reason about indirect downstream effects.*
- *Two distinct Gemma 4-2B-IT calls: (1) Background inference call — fired immediately after NER completes whenever `unmatched` is non-null; Gemma reasons about indirect DBN node effects and may write them via `write_indirect_effect()` with confidence capped at 0.5. (2) Insight call — Glance or Reflect (then called Rapid/Deep-think), user-triggered or condition-triggered, produces the insight string shown to the user.*
- *Dual-mode Gemma agent — Glance and Reflect (introduced in v7.6 as "Rapid" and "Deep-think"; renamed in v7.7): user selects mode before sending. Glance = single structured Gemma call, no ReAct loop, context pre-injected (~5 s). Reflect = full ReAct loop with tool calls (~30 s).*
- *Pre-injection of context: DBN state + recent observations + sensor snapshot are injected into the system prompt before both Glance and Reflect calls. The agent always has current belief state without needing to call `query_dbn` as a first tool step in Glance mode.*
- *Streaming output: tokens streamed to the UI as generated for both modes. A 30-second Reflect response starts appearing within ~1 second — making latency imperceptible to the user.*
- *Thinking block capture: Gemma 4-2B-IT emits `<|channel>thought`...`<channel|>` blocks when `<|think|>` is in the system prompt. These blocks are captured and stored (not stripped), enabling future audit of the model's reasoning chain. Requires llama.cpp post-April 2026 for correct Gemma 4 tag parsing.*
- *Two-tier insight: Tier 1 = Glance structured call (~5 s, always available); Tier 2 = Reflect ReAct (~30 s, user-selected via mode toggle).*
- *Pipeline latency decoupling: fast path (1B seg → 1B NER → cascade) completes in ~4–8 s and returns to the user. Background Gemma inference call and insight generation are decoupled — the user receives cascade questions while deeper reasoning runs in parallel.*

---

## What is new in v7.7

- *Three-feature UI architecture introduced: Talk (interactive conversation, Glance/Reflect mode), Journal (private entry, no model response), Report (PDF export, future). Each feature maps to a distinct agent flow. The Report feature is not yet implemented; Talk and Journal are fully wired.*
- *Mode terminology aligned with implementation: "Rapid" renamed to **Glance**; "Deep-think" renamed to **Reflect**. Glance = single structured Gemma call, no ReAct loop, ~5 s. Reflect = full ReAct loop, ~30 s streamed. All code, prompts, and UI labels use these names.*
- *Two-phase Talk turn split: `startTurn` (Phase 1) and `completeTurn` (Phase 2) replace the single `runTurn` function. Phase 1 runs undo-check → NER → initial inference → builds question list. If no questions → completes inline and returns `{ done: true }`. If questions exist → stores state in `_pendingTurns` map, returns `{ done: false, turnId, questions }`. Phase 2 writes answers → marks evidence → re-runs inference (Glance) or defers to ReAct (Reflect) → Gemma → store.*
- *`_pendingTurns` map: an in-process `Map<string, PendingTurn>` holds the phase 1 state (`sessionId`, `userMessage`, `isUndoTurn`, `beliefs`, `memory`) between the two phases, keyed by `turnId`. The entry is deleted when `completeTurn` consumes it.*
- *Inline completion shortcut: if Phase 1 builds zero follow-up and cascade questions, it completes the full turn inline and returns `{ done: true }` — no round-trip through the question UI and no `completeTurn` call required.*
- *`DisplayQuestion` unified type exported from agent.ts: covers both `FollowUpQuestion` and `CascadeQuestion` for uniform UI rendering. Fields: `original_col`, `node_name`, `question`, `opts?`, `range?`. UI never distinguishes question origin — rendering logic is identical for both types.*
- *`CascadeQuestion` opts and range fields added: `CascadeQuestion` interface now carries `opts?: { v: number; l: string }[]` and `range?: { min: number; max: number; unit: string }`, matching `FollowUpQuestion` structure. `tryAdd()` in questionCascade.ts populates these from the column entry. Cascade questions are now structurally identical to follow-up questions at the UI boundary.*
- *`markEvidenceWritten()` export added to mcp.ts: sets `_session.lastEvidenceWriteAt = Date.now()`. Called in `completeTurn` after proactive answers are written, allowing the second `run_dbn_inference` call to pass the inference throttle. Without this, the throttle would block the Phase 2 inference because `writeProactiveAnswer` bypasses the normal evidence write path that updates `lastEvidenceWriteAt`.*
- *Inference quota management per mode: Glance — Phase 1 fires inference count 1; Phase 2 fires count 1→2 after writing answers. Reflect — Phase 1 fires count 1; Phase 2 skips direct inference so the ReAct loop can use the 1→2 shot itself after storing indirect evidence.*
- *`writeProactiveAnswer` private helper: writes one proactive Q&A answer to `user_data_sensorless` with `data_source='proactive'`, `was_proactive=1`, `merge_mode='scale'`, `temporal_flag='decaying'`. Called per answer in `completeTurn` before the inference re-run.*
- *Journal feature (`runJournalTurn`): NER with reflect-mode confidence gates → `run_dbn_inference` → `storeChatMessage` (user only). No Gemma call. Gemma is skipped entirely; beliefs update silently. The journal entry is preserved in `chat_messages` for memory retrieval but produces no model response.*
- *ReAct prompt format updated to match implementation: Gemma is prompted using plain-text THOUGHT/TOOL_CALL/OBSERVATION/RESPONSE tokens, not `<tool_call>` XML tags. Format: `TOOL_CALL: {"name": "...", "arguments": {...}}` — system appends `OBSERVATION: {result}` — model continues. RESPONSE: terminates the loop. `<|think|>` is not used; reasoning is expressed via THOUGHT: lines.*
- *Gemma instruct prompt format documented: `<bos><start_of_turn>system\n{system}\n<end_of_turn>\n<start_of_turn>user\n{context}\n<end_of_turn>\n<start_of_turn>model\n`. Observation injections use the user turn slot: `<start_of_turn>user\nOBSERVATION: ...<end_of_turn>\n<start_of_turn>model\n`.*
- *Reflect mode ReAct tool registry narrowed to four tools: `run_dbn_inference`, `get_changed_nodes`, `store_indirect_evidence`, `get_user_memory`. `store_indirect_evidence` is capped at one use per turn to prevent evidence flooding. `run_dbn_inference` and `get_changed_nodes` are allowed only after new evidence has been stored — the system prompt instructs the agent not to re-call these if context already contains current beliefs.*
- *Tool dispatch is direct TypeScript function dispatch in the current implementation (`dispatchTool()` in mcp.ts), not JSON-RPC over stdio MCP servers. The MCP stdio-server architecture described in v7.5 is the target design; the current implementation uses a direct in-process switch to the same underlying functions. The interface contract (tool name + arguments object) is identical.*
- *State-based router in App.tsx: `screen: 'home' | 'talk' | 'journal'` state drives which screen renders. No react-navigation dependency. Navigation library can be adopted later without changing screen components.*
- *Gemma persona and response guide codified in agent.ts: health companion persona with six hard rules (no diagnoses, no medications, no clinical language, no probability numbers, lead with empathy, redirect off-topic queries gracefully). Response guide specifies six structural elements: acknowledge → causes → effects → soft suggestion → retrospective question (conditional) → follow-up questions. Retrospective question is only included when a specific pattern was surfaced this turn — never as a template.*

---

## What is new in v7.8

### Passive Sensing Architecture — Phase 1

Passive sensing is now implemented as a real on-device background process (expo-background-fetch + expo-task-manager). The background task fires every `SENSING_INTERVAL_MINUTES = 15` minutes; iOS may throttle the actual interval longer, but the task is idempotent and safe to call multiple times within the same window.

**sensor_windows table (new):** All passive sensor readings are written as columnar rows. One row per `(date, snapshot_time, node_name, source_column)` per firing window. `INSERT OR REPLACE` semantics — a second firing in the same window silently overwrites the previous row for that node/source_column pair.

```
sensor_windows
  date              TEXT  — YYYY-MM-DD (local device timezone)
  snapshot_time     TEXT  — YYYY-MM-DDTHH:MM:SS at firing time (local, updated v8.0)
  window_start      TEXT  — HH:MM:SS = snapshot_time − SENSING_INTERVAL_MINUTES
  node_name         TEXT  — DBN node name (e.g. 'time_of_day', 'activity')
  source_column     TEXT  — column key used in evidence layer
  data_source       TEXT  — 'clock' | 'pedometer' | 'prev_day_*' | ...
  raw_value         REAL  — nullable; raw numeric reading
  raw_unit          TEXT  — nullable; unit string (e.g. 'steps', 'hour')
  discretized_value TEXT  — validated state label (e.g. 'morning', 'high', 'low')
  confidence        REAL  — [0, 1]
```

**Phase 1 collectors (implemented):**

| Collector | Source | Output node | source_column | Discretization |
|-----------|--------|-------------|---------------|----------------|
| `clockCollector` | System clock (`Date`) | `time_of_day` | `time_of_day` | hour → morning/afternoon/evening/night |
| `activityCollector` | `expo-sensors` Pedometer | `activity` | `active_ratio` | steps/window → active_ratio → low/high via two-regime logic |

**Two-regime activity discretization (updated in v8.0):**
- **Cold-start** (< 7 distinct calendar days in `sensor_windows`): absolute clinical threshold. `active_ratio = steps / 1500`. `<= 0.25 → low`, `> 0.25 → high`. Threshold updated from 0.33 → 0.25 after retraining (new K-Means bin edge).
- **Calibrated** (≥ 7 days history): personal history majority vote. Bin 30-day historical ratios using same clinical threshold; if majority (> 50%) fall in same bin as current → confidence 0.90, else 0.75. The z-score calibrated regime described in v7.8 has been replaced — z-score is a training-time normalization tool; runtime uses clinical threshold + personal history for consistency.

The activity collector queries the last 30 days of `sensor_windows` to determine which regime applies. `CLINICAL_THRESHOLD = 0.25` in `activityCollector.ts` must match the K-Means trained bin edge in `feature_node_config.json`.

**Phase 2+ stubs (not yet implemented):** `heart_rate`, `sleep_physio`, `screen_usage`, `communication` — all return null from their collector stubs. Their native expo modules (HealthKit / Health Connect, screen-time APIs) are not yet installed.

**Prev-day aggregation:** On the first sensing window of each new calendar day, yesterday's `sensor_windows` rows are aggregated into summary rows for today's inference context. Called once per day — guarded by `_lastPrevDayDate` module-level variable.

```
Aggregation strategy per node:
  activity  (continuous ratio):  mean of non-null raw_value → re-discretize at cold threshold
  time_of_day (label-only):      majority vote on discretized_value
  default fallback:              majority vote on discretized_value (or mean if numeric)
```

Prev-day rows use the **actual node_name** (e.g. `time_of_day`) with a `prev_day_` prefix on `source_column` (e.g. `prev_day_time_of_day`). Confidence is fixed at `0.60` — below the cold-start threshold of `0.65` — so current-window evidence always supersedes prev-day evidence in the belief update.

**`initPassiveSensing(db)`** — public entry point. Call once after `initDb()` at app startup. Registers the background fetch task and immediately runs one collection pass so the first inference window has sensor evidence before the background task fires.

---

### Social Events Nodes Fixed

`social_events_positive` and `social_events_negative` had `self_report: false` in `feature_node_config.json` (and its mobile asset mirror `feature-node-config.json`). This silently broke the entire evidence pipeline for these two nodes:

```
self_report: false
  → _build_source_col_to_node() skips node          (export_column_question_map.py)
  → column_question_map.json omits both rows
  → NODE_SCHEMA excludes nodes                       (nlu.ts module load)
  → SOURCE_COL_TO_NODE reverse map excludes nodes
  → NLU never matches user statements to these nodes
  → Questions never asked via cascade
  → DBN receives zero evidence for both nodes
  → Learned edges have no effect on inference
```

**Fix applied:** Both nodes set to `self_report: true`, `ask_interval_days: 7` in both `configs/feature_node_config.json` and `Codebase/mobile/src/assets/feature-node-config.json`. The column question map was re-exported (97 entries total), and `column-question-map.json` was synced to the mobile assets directory.

Both nodes' `source_column_bins` are keyed by `prev_day_events_positive` and `prev_day_events_negative` — these are the actual EMA columns in the training data. The reverse map now correctly resolves `prev_day_events_positive → social_events_positive` and `prev_day_events_negative → social_events_negative` in both the export script and the NLU module.

---

### Slider UI for Social-Events Intensity Questions

`prev_day_events_positive` and `prev_day_events_negative` use a 1–7 Likert intensity scale (not a discrete option set). The CSV `options_json` field for these rows is blank — the existing option-parsing path falls through to no `opts` and no `range`, producing an unusable question widget.

**Fix:** `_RANGE_OVERRIDES` dict added to `export_column_question_map.py` and `_RANGES` dict updated in `build_column_question_map.py`. The export script checks `_RANGE_OVERRIDES` first before parsing CSV options, injecting a `range` entry:

```python
_RANGE_OVERRIDES: dict[str, dict] = {
    'prev_day_events_positive': {'min': 1, 'max': 7, 'unit': 'intensity'},
    'prev_day_events_negative': {'min': 1, 'max': 7, 'unit': 'intensity'},
}
```

The exported JSON entries now carry `"range": {"min": 1, "max": 7, "unit": "intensity"}` instead of missing `opts`. The `DisplayQuestion` type's `range?` field causes the UI to render a slider rather than chip buttons for these two questions.

---

### Topic Threading

Every user message now carries a `topic` string — a compressed semantic label derived at NER time and stored persistently in `chat_messages`. Topics serve two purposes: (1) they enrich the semantic memory query before `get_user_memory` runs, improving retrieval relevance; (2) they annotate the raw context window so Gemma can see what each previous message was about at a glance.

**Topic derivation in `runNer`:**

For each segment:
- **Matched** (entities extracted): `topic = entity.summary_text` joined per segment. `summary_text` is the NER-produced compressed factual note (e.g. `"stressed about exams"`, `"slept 5 hours last night"`).
- **Unmatched** (no L1/L2/L3 match, including exceptions): walk-back to the most recent `chat_messages` row with a non-empty `topic` for this `session_id`. Fetched lazily and cached within the NER call. This ensures unmatched segments (e.g. `"I had a coffee"`) inherit the session topic rather than leaving it blank.

Topics from all segments are deduplicated while preserving order, then joined into a single `topic` string returned as `NluResult.topics`.

**Topic usage in `startTurn`:**

```
1. getRecentPairs(db, sessionId)
     → SELECT topic, content FROM chat_messages WHERE role='user' AND evicted=0
       ORDER BY created_at DESC LIMIT MEMORY_BUFFER_SIZE
     → returns RecentPair[] = { topic: string; content: string }[]

2. setRecentTopics(recentPairs.map(p => p.topic).filter(Boolean).join(' '))
     → stored in _session.recentTopics
     → get_user_memory appends this to the query string:
       query = currentUserMessage + ' ' + recentTopics
     → semantic search is now topic-enriched before NER even runs

3. detectUndoIntent (Stage 1) → isUndoTurn
   classifyIntent  (Stage 2) → intentResult
   runNer          (Stage 3) → only if hasReport=true OR isUndoTurn=true
     → nerResult.topics is always derived from NER regardless of undo state
     (runUndoWork in Stage 3b handles undo-specific work; it does NOT run NER)

4. topic = nerResult.topics.join(', ')
     → stored in _pendingTurns as pending.topic

5. After Gemma response:
     storeChatMessage(db, sessionId, turnId, 'user', userMessage, topic)
     storeChatMessage(db, sessionId, turnId, 'model', response)
     → user message is stored WITH topic; model message has no topic
```

**Topic formatting in `buildContextBlock`:**

```typescript
const formatted = recentPairs
  .map(p => p.topic ? `[${p.topic}] ${p.content}` : p.content)
  .join('\n');
// Output: "[stressed about exams, slept 5 hours] I was really stressed..."
```

These are PREVIOUS turns only — the current turn's topic is not yet written to DB when Gemma runs, and is not included in recentPairs.

**Important ordering constraint:** `setRecentTopics` must be called BEFORE `dispatchTool('get_user_memory', ...)`. The memory query is enriched using the stored recent topics at call time. If the order is reversed, the memory search runs without topic enrichment.

---

### Undo Handling — Stage 3 + Stage 3b

`detectUndoIntent` runs as Stage 1 and sets `isUndoTurn`. NER (Stage 3) runs when `hasReport=true` OR `isUndoTurn=true` — in both cases NER produces the same `nerResult`. The old `runUndoFlow` (which ran NER internally as a separate fork) is replaced by `runUndoWork`, which receives the Stage 3 NER result as input.

```
Stage 1: detectUndoIntent → isUndoTurn
Stage 2: classifyIntent   → IntentResult (hasReport, hasQuery, ...)
Stage 3: runNer — runs if hasReport=true OR isUndoTurn=true
          → nerResult: { entities, unmatched, topics }
Stage 3b: runUndoWork(db, sessionId, turnId, nerResult) — runs if isUndoTurn=true
           → looks up prevTurnId and prevTurnNodes
           → resolves targets = nerNodes ∩ prevNodes
           → calls undo_last_entry { turn_id: prevTurnId, node_names: targets }
           → captures preUndoBeliefs from inference_snapshots[1]
           → correctedNodes = Set(targets)
           → newNodes = entities.map(e.node_name).filter(n ∉ correctedNodes)
           isPureUndo = preUndoBeliefs !== null
                        && unmatched.length === 0
                        && newNodes.length === 0
```

`nerResult.topics` is always populated from Stage 3, regardless of whether undo work ran.

**Pure undo short-circuit:** If `isPureUndo=true`, the turn short-circuits immediately after Stage 3b: an ack is returned, `storeChatMessage` is called with the derived topic, and Gemma is never called. This is the only case where Gemma is skipped in a Talk turn.

**`isPureUndo` bug fix:** The old condition was `preUndoBeliefs !== null && unmatched.length === 0`. This had a bug: a message like "stress was wrong — also chest pain" would match `unmatched.length === 0` (chest pain was NER-extracted into `entities`, not `unmatched`) but would still silently drop the chest pain entity. The new condition adds `newNodes.length === 0`, where `newNodes` are entities NOT in `correctedNodes`. If any entity is genuinely new (not a correction), the turn continues to Gemma normally.

---

### `_pendingTurns` Map Updated

`PendingTurn` stores the full context needed for Phase 2, including the intent classification result and trend summary produced in Phase 1:

```typescript
interface PendingTurn {
  sessionId:          string;
  userMessage:        string;
  isUndoTurn:         boolean;
  beliefs:            BeliefResult | null;       // Phase 1 inference result
  memorySummaries:    string[];                  // get_user_memory results
  recentPairs:        RecentPair[];              // raw { topic, content } pairs for Gemma
  topic:              string;                    // NER-derived topic for storeChatMessage
  isEarlyInteraction: boolean;                   // getReportedTurnCount < 3
  intentResult:       IntentResult;             // Stage 2 classifyIntent output
  trendSummary:       string;                    // get_belief_trend output (when hasTrendQuery)
}
```

`recentPairs` is passed directly to `buildContextBlock` in Phase 2 — the same pairs fetched in Phase 1. `topic` is passed to `storeChatMessage` after Gemma responds. `intentResult` and `trendSummary` drive `selectBeliefWindow` and `buildContextBlock` in Phase 2 identically to how they did in Phase 1's inline-completion path.

---

## What is new in v7.9

- *Intent routing architecture formalised as an 8-stage pipeline (Stage 0–7). Each stage has a precise trigger condition; only the stages whose conditions are met run on any given turn. Stage 0 (isAckOnly) short-circuits without any model call; Stage 2 (classifyIntent) runs on every non-ack turn.*
- *`classifyIntent` function added to nlu.ts: Llama, `n_predict=55`, `temperature=0.0`. Returns `IntentResult`: `{hasReport, hasQuery, hasTrendQuery, isSocial, isThirdPartyQuery, queryNodes[]}`. Runs as Stage 2, always, before NER.*
- *`socialFastPath = isSocial && !hasQuery && !hasReport` — skips NER, skips DBN inference, forces Gemma glance mode at `n_predict=80`.*
- *Stage 3 (NER) now runs conditionally: only when `hasReport=true` OR `isUndoTurn=true`. Replaces the old fork where undo internally called NER and regular turns called NER separately.*
- *`runUndoFlow` refactored to `runUndoWork(db, sessionId, turnId, nerResult)`: no longer runs NER internally. Takes NER result produced by Stage 3 as input; handles only soft-delete + target resolution + `isPureUndo` check.*
- *`isPureUndo` bug fixed: added `newNodes.length === 0` check. Old condition allowed `"stress wrong — also chest pain"` to silently drop chest pain. New condition: `preUndoBeliefs !== null && unmatched.length === 0 && newNodes.length === 0`.*
- *`formatBeliefSummary` replaced by `selectBeliefWindow`: dynamic gate/budget based on intent. `isEarlyInteraction` → gate=0.0, all nodes, `[prior — no user data yet]` annotation. `hasQuery` → gate=0.35, budget=15, queryNodes forced in. Report-only → gate=0.55, budget=8. socialFastPath → empty string.*
- *`isEarlyInteraction` replaces `isFirstInteraction`: fires when `getReportedTurnCount(db, sessionId) < 3`. Counts distinct turns with `user_data_sensorless` entries — ignores social/query/greeting turns. Activates prior-mode belief display for the first 3 real DBN-updating turns.*
- *`buildContextBlock` injections expanded to 7 stackable types: CORRECTION TURN, FIRST INTERACTION, QUERY TURN, MIXED SOCIAL+QUERY, SOCIAL TURN, THIRD-PARTY QUERY, NOTE (unresolved queryNodes). All conditions are independent — any combination may fire on the same turn.*
- *`get_belief_trend` MCP tool added: `{node_names, window_days=14, session_id}` → `{trends: Record<string, string>}`. Per-node summary format: `"stress: avg moderate, started high, currently low"`. Skips node if < 3 entries in window. Available to Gemma in Reflect mode via `GEMMA_TOOLS`. In Glance mode, fetched directly by `startTurn` when `hasTrendQuery=true` and injected as a TREND HISTORY block.*
- *PERSONA guardrail added: Gemma never claims the user is "improving," "making progress," "trending," or "doing better/worse over time." Current state only; honest acknowledgement when trends are asked about.*
- *Stage 0 `isAckOnly` exact wordlist: `["ok","okay","k","got it","thanks","thank you","cool","alright","sure","noted","👍","👌"]`. Exact match only — no partial/substring. Canned warm response, no model calls.*
- *`PendingTurn` interface updated: adds `isEarlyInteraction: boolean`, `intentResult: IntentResult`, `trendSummary: string` to the fields stored between Phase 1 and Phase 2.*

---

## What is new in v8.0

### DBN Retraining — Population Z-Score + Clinical Threshold Alignment

The retraining pipeline (Steps 1–5) was completed. The core change: sensor columns that are bounded or ordinal (active_ratio, etc.) are **not** z-scored during training. Population z-score is applied only to unbounded continuous sensor columns. This change ensures runtime discretization matches training-time discretization exactly.

**Population norm stats export:**
- `configs/population_norm_stats.json` generated by the training pipeline: `{col: {mean, std}}` for every sensor column that uses z-score discretization.
- `Codebase/mobile/src/core/sensorCollectors/populationNormStats.ts` auto-generated from this file by `sync_configs.py`. Metro bundler cannot reach `configs/` at runtime — the stats are baked into the TS file at sync time. The `discretizeByEdges` helper is also exported from this file.

**Activity threshold update — 0.33 → 0.25:**
The K-Means trained bin edge for `active_ratio` changed from 0.33 to 0.25 after retraining. All runtime references updated:
- `activityCollector.ts`: `CLINICAL_THRESHOLD = 0.25`, `clinicalBin(ratio): ratio <= 0.25 → 'low'`
- `passiveSensing.ts` prev-day aggregation: changed `< 0.33` → `<= 0.25`

**Two-regime activity discretization (updated):**
- **Cold-start** (< 7 distinct calendar days in `sensor_windows`): absolute clinical threshold. `active_ratio = steps / 1500`. `<= 0.25 → low`, `> 0.25 → high`.
- **Calibrated** (≥ 7 days history): personal history majority vote. Bin historical ratios by same clinical threshold; if majority of 30-day history falls in same bin as current reading → confidence 0.90, else 0.75.
- The v7.8 z-score calibrated regime is replaced by majority vote. Population z-score is a training-time tool; runtime uses clinical threshold + personal history consistently.

**feature_node_config.json updates:**
- `state_map` fields added to 35 `source_column_bins` entries where source state labels did not match node state labels (k-modes collapse during training created this mismatch).
- `is_composite: true` added **only** to true summed instruments: `phq_total` (PHQ-9), `gad_total` (GAD-7). All other multi-item columns (`pain_level`, `diabetes_status`, `chronic_condition`) are independent binary/ordinal questions — NOT composite.
- This distinction is critical: the composite guard in `resolveNodeValue` now checks `binConf.is_composite === true` rather than `COMPOSITE_ITEM_COUNT[col] > 1`, which was incorrectly blocking independent questions.

---

### sync_configs.py — Post-Training Sync Script (new)

`Codebase/backend/sync_configs.py` is the mandatory last step after every training run.

**Actions:**
1. Copies 5 JSON files from `configs/` → `Codebase/mobile/src/assets/` (with underscore→dash rename):
   - `feature_node_config.json` → `feature-node-config.json`
   - `dbn_structure.json` → `dbn-structure.json`
   - `cpd_tables.json` → `cpd-tables.json`
   - `source_column_descriptions.json` → `source-column-descriptions.json`
   - `column_question_map.json` → `column-question-map.json`
2. Regenerates `populationNormStats.ts` from `configs/population_norm_stats.json`.

**Post-training export order (mandatory):**
```
1. python -m Codebase.backend.export_cpds          → configs/cpd_tables.json
2. python -m Codebase.backend.export_column_question_map   → configs/column_question_map.json  (only if CSV changed)
3. python Codebase/backend/sync_configs.py         → mobile assets + populationNormStats.ts
```

---

### export_cpds.py — inter_trans Shape Fix (C1 + C1-B)

**Problem:** Static nodes (no t=1 parent — e.g. `mental_health`, `loneliness`, `extraversion`, `neuroticism`) produced a 1D inter_trans array during export. The first fix wrapped this as `[result]` giving `[1×k]`. `applyInterSlice` expects `[k×k]`: `trans.map(row => row.reduce(...))` on a `[1×k]` matrix produces `[scalar]` → `normalize([scalar]) = [1.0]` → `safeVec([1.0], k)` = uniform priors. 6 nodes silently lost all distributional information.

**Fix:** `export_cpds.py` now constructs a proper `[k×k]` identity-like matrix for all static nodes. All 20 affected nodes in `cpd_tables.json` were regenerated.

**Runtime guard added (E2):** `applyInterSlice` in `inferenceEngine.ts` now validates `trans.length === k && trans[0].length === k` before applying temporal priors. Malformed matrices are skipped with a `console.warn` rather than silently corrupting beliefs.

---

### Model Initialization — Eager Init + Eviction Recovery (C1-REG + E1)

**initModels.ts (new):** Eager init orchestrator. Called from `App.tsx` splash screen. Initializes all three models in sequence:
```
initNlu(nluPath) → initEmbed(embedPath) → initAgent(agentPath)
```
App.tsx awaits this promise before rendering any screen. If any path is empty, a user-visible error is surfaced — silent skip no longer possible.

**_modelPath + ensureX() pattern (all three model modules):**
Each model module (`nlu.ts`, `embed.ts`, `agent.ts`) stores `_modelPath: string | null` at init time. `ensureNlu()` / `ensureEmbed()` / `ensureAgent()` check if `_ctx === null` (OS eviction signal) and reinit from stored path before proceeding.

**embedText eviction recovery (E1):** `embedText()` now calls `ensureEmbed()` before using `_ctx`. Previously it threw `'embed: not initialized'` — this meant OS eviction during `storeMemory` → `embedText` would cause memory writes to silently fail. Now transparently recovered.

**OS eviction retry pattern (agent.ts):** Wrapped around all native model calls:
```typescript
try {
  result = await _ctx.completion(...)
} catch (e) {
  if (isNativeContextError(e)) {
    _ctx = null;
    await ensureAgent();
    result = await _ctx!.completion(...)   // single retry
  } else throw;
}
```
`isNativeContextError` matches: `context.*null | llamarpc | native.*invalid | llama.*not.*init`.

---

### Evidence Layer — Three-Path Completeness + Universal Fallback

`buildDbnEvidence` now collects evidence through three ordered passes, followed by a universal fallback:

```
Pass 1: Sensorless nodes (user_data_sensorless, STALENESS_DAYS keyed)
Pass 2: Sensor nodes — current window (fetchSensorSnapshot → latest sensor_windows row)
Pass 3: Prev-day nodes (fetchPrevDaySnapshot → source_column LIKE 'prev_day_%')
Pass 4: Universal fallback — any DBN node still absent from both evidence and
        prior_factors gets MARGINAL_PRIORS[node] injected
```

**Design principle:** Every DBN node must always contribute to LBP — either through collected data or through its trained marginal prior. The universal fallback guarantees cold-start inference (first app open, no data) uses trained priors rather than LBP's implicit uniform assumption.

**fetchPrevDaySnapshot fix (H1):** Queries `WHERE source_column LIKE 'prev_day_%'`, not `WHERE node_name LIKE 'prev_day_%'`. `computePrevDayAggregates` writes `node_name = original_node_name` (e.g. `'activity'`) and only prefixes `source_column` with `prev_day_`. The original query returned zero rows, making the entire M15 prev-day path dead.

**Confidence gate (unchanged):** `confidence >= 0.85` → hard evidence; below → soft `prior_factors` vector via `toPriorVector`.

---

### resolveNodeValue — Complete Column Resolution Pipeline

`resolveNodeValue` in `nlu.ts` now handles all column types correctly. Resolution order:

```
1. passthrough branch: if method='passthrough', raw string IS the node state — write directly
2. value_map: if binConf.value_map exists, apply first:
              numeric raw_value string → intermediate label (e.g. '1' → 'married')
3. state_map: apply to map source label → node label (e.g. 'poor_fair' → 'low')
4. state validation: verify result exists in node's state_labels — null if not
5. is_composite guard: if binConf.is_composite === true AND answers < 80% threshold → null
```

Previously: passthrough columns returned null (H6), value_map was ignored (M3), is_composite blocked independent binary questions (M4).

---

### Composite Scoring — Partial Score Resolution (L2)

`maybeResolveComposite` in `nlu.ts` handles partial PHQ-9 / GAD-7 scoring:

- **< 80% items answered** → `node_value = null` (too unreliable to score)
- **≥ 80% items answered** → compute partial sum → discretize → write `node_value` with confidence: `0.65 + 0.25 × (answered / total)`
  - e.g. 4/5 items answered → confidence 0.85; 5/5 → 0.90

`getCompositeSourceCol(itemCol)` helper exported from `nlu.ts`. Used by `completeTurn` in `agent.ts` to look up the composite source column (e.g. `phq_sleep → phq_total`) before calling `maybeResolveComposite`. Previously `completeTurn` passed the item column directly — `COMPOSITE_ITEM_COUNT['phq_sleep']` was undefined → function returned immediately → PHQ-9 scoring never fired from cascade answers (H2).

---

### Session Management — Bleeding Prevention + Stale Turn Cleanup

**resetSession() on session change (M5):** `agent.ts` stores `_lastSessionId` in module scope. At the start of `startTurn`, if `sessionId !== _lastSessionId`, `resetSession()` is called before any processing. Prevents stale `latestBeliefs` from a previous session becoming the `turnStartBeliefs` baseline for a new session's first `get_changed_nodes` call.

**evictStalePendingTurns() in completeTurn (E3):** `evictStalePendingTurns()` is now called at the start of both `startTurn` and `completeTurn`. Previously only called at `startTurn`. If a user abandoned a question prompt and never called `completeTurn`, the stale pending turn would survive until the next `startTurn`. TTL = 30 minutes.

---

### store_indirect_evidence — Full Validation (L1 + M12)

`handleStoreIndirectEvidence` in `mcp.ts` now validates both `node_name` and `node_value`:

```
node_name validation: must exist in MODEL_PARENTS (the DBN graph)
node_value validation: must exist in MODEL_STATES[nodeName]
On failure: { stored: false, reason: 'invalid_node_value', valid_values: [...] }
```

Previously: `node_name` was checked but `node_value` accepted any string. Gemma could store `stress_ema = 'very_high'` → `slist.indexOf(evidence[node])` returns -1 → node pinned to state[0] silently.

---

### Snapshot Time Standardization (M1)

All writers — `infer.ts` and `mcp.ts` — now use identical format:

```typescript
new Date().toLocaleString('sv').replace(' ', 'T').slice(0, 19)
// → 'YYYY-MM-DDTHH:MM:SS' (device local time)
```

Previously `infer.ts` used `toLocaleTimeString('sv')` → `HH:MM:SS`. Since `snapshot_time` is part of the composite PRIMARY KEY and ordering uses `ORDER BY date DESC, snapshot_time DESC`, mixing ISO-datetime strings with plain time strings produced incorrect temporal ordering when rows from both paths existed in the same table.

`db.ts` schema comment updated: `-- local YYYY-MM-DDTHH:MM:SS (window end)`.

---

### get_changed_nodes — co-influenced Label (L18)

Nodes that share a parent with an evidence node (co-influenced siblings) are now labeled `'co-influenced'` instead of `'effect'`. They are not causal descendants of the evidence node — they are siblings that happen to share a common cause. Mislabeling them as `'effect'` caused Gemma to reason about incorrect causal chains.

---

### Few-Shot Example Corrections (H7 + M2)

Invalid node names removed from all NLU few-shot examples:
- `anxiety_level` → `stress_ema` in `CLASSIFY_INTENT_SHOTS`
- `mood_valence` → `mood` in `CLASSIFY_INTENT_SHOTS` and `RESPONSE_GUIDE`
- `sleep_duration_hrs` → `sleep_quality`

---

### questionCascade.ts — Suppression Query Fix (M11)

`isSuppressed()` in `questionCascade.ts` now uses `ORDER BY created_at DESC LIMIT 1`. Previously missing ORDER BY caused SQLite to potentially return the oldest row. If oldest had expired suppression and newest had valid suppression → question was incorrectly shown to the user.

---

### sqlite-vec Probe at Startup (L21)

`db.ts` `initDb()` now runs `probeSqliteVec()` immediately after opening the database. Executes a probe query using `vec_distance_cosine`. If the sqlite-vec extension was not included in the native build, this throws a descriptive error at startup rather than an opaque SQL error on the first `searchMemory()` call.

---

### Frontend — v0.dev App-Build Analysis

`Codebase/mobile/src/app-build/` contains a **Next.js / web app** generated by v0.dev. It requires full porting to React Native before it can run on-device. Key findings:

**Runtime incompatibilities:** `localStorage`, `document`, `window.matchMedia`, `framer-motion`, `recharts`, all `@radix-ui/*` components, HTML elements (`div`, `textarea`, `input`), CSS class system (`className`). None of these work in React Native.

**Dependencies to replace:**
| Web package | React Native replacement |
|---|---|
| `framer-motion` | `react-native-reanimated` |
| `recharts` | `victory-native` or `react-native-gifted-charts` |
| All `@radix-ui/*` | RN primitives |
| `next` / `react-dom` / `next-themes` | Remove |
| `localStorage` (use-store.ts) | `@react-native-async-storage/async-storage` |

**Mock data locations:**
- `chat-screen.tsx`: `SIMULATED` constant (lines 33–48) — 9 hardcoded responses; replace with `startTurn` / `completeTurn` / `runJournalTurn`
- `dashboard-screen.tsx`: `screenData`, `activityData` arrays + hardcoded ring values 68/78; replace with `getLatestBeliefs()` + `sensor_windows` queries
- `onboarding-screen.tsx`: `onComplete` writes to localStorage only; must wire to `onboardingHelpers` + `user_data_sensorless` DB writes

**Backend gaps:**
- No question card component for `DisplayQuestion[]` (two-phase Talk flow)
- No belief → 0–100 ring score conversion function (dashboard rings)
- No permission request flow (pedometer, call log, SMS, screen usage)
- No model download / GGUF loading screen
- `Report` / `Ultra` modes have no backend implementation (`AgentMode` is `'glance' | 'reflect'` only)
- `db: DB` and `sessionId: string` not threaded to any screen (must use a React context)

**Birthdate format bug:** `DateWheelPicker` produces `DD-Mon-YYYY`; `onboardingHelpers.computeAgeYears` expects ISO `YYYY-MM-DD`. Parsing fails silently → NaN age.

**Porting priority order:**
1. React context provider for `db`, `sessionId`, `modelsReady`
2. `use-store.ts` → AsyncStorage
3. Wire `onboarding-screen` to `onboardingHelpers` + DB writes; fix birthdate format
4. Wire `chat-screen.send()` to `startTurn`/`completeTurn`/`runJournalTurn`; build question card component
5. Wire `dashboard-screen` to `getLatestBeliefs()` + sensor DB queries; write belief → score mapper
6. Add permissions request screen (post-onboarding)
7. Port all components from web primitives to RN primitives
8. Replace all `framer-motion` with `react-native-reanimated`

---

## 1. System Philosophy

This app does three things simultaneously: it passively watches you through phone sensors, it listens to what you tell it, and it maintains a probabilistic model of your health state over time. Every architectural decision flows from one core constraint — it must run as a student project with zero budget, zero clinical ground truth, and a single developer.

**The two-brain architecture** is the central design: the DBN is the memory and the reasoner, the SLM is the communicator. Neither can do the other's job. The DBN cannot explain itself in English. The SLM cannot maintain probabilistic belief over time. Together they cover the full interaction cycle.

### Passive-First Philosophy

The most successful health apps — Oura, Whoop, Apple Health — ask almost nothing from the user. They observe. They infer. They occasionally surface one insight. The user feels understood without feeling interrogated. This app follows the same principle: passive sensing is the primary data source; user input is a supplement that corrects or enriches passive inference, not a replacement for it.

### Chat vs Dashboard Distinction

The dashboard shows the overall state: physical stress %, mental stress %, 7-day trends. The chat is an interactive causal window. When a user reports something, the chat insight explains both what that specific thing causes downstream and what the DBN sees as causing it upstream.

### Training vs Runtime Separation — Two-Layer Architecture

The system uses two distinct layers at runtime:

- **Layer A — Coarse DBN:** 6 passive nodes, 6+ self-report nodes, 2 latent nodes — trained on all survey data, safe parameter budget (~14-16 nodes total).
- **Layer B — Sub-Dimension Attribute Store:** Every individual item from PHQ-9, PSS, PANAS, loneliness scale, VR-12, BigFive stored in SQLite as `user_profile_attributes`. These attributes are NOT DBN nodes — they accumulate through pointed questions over weeks and aggregate into coarse DBN evidence via evidence fusion.

### Rolling Inference — Not Daily Batch

The DBN runs every N minutes (configurable) AND immediately when the user sends a message. These are two independent triggers calling the same inference pipeline.

---

## 2. Phase 0 — Development Environment Setup

| Tool | Role | Why |
|------|------|-----|
| Google Colab | DBN training | Free GPU for HillClimbSearch + Structural EM |
| Google Drive | Version control substitute | Save after every session |
| VS Code | All non-GPU Python logic | Evidence fusion, gates, SQLite, FastAPI |
| pgmpy | DBN construction | HillClimbSearch, BicScore, VariableElimination, BeliefPropagation, ExpectationMaximization |
| HuggingFace | Zero-shot fallback | `facebook/bart-large-mnli` for intent classification |
| React Native Expo | Mobile frontend | expo-health, expo-sqlite, expo-notifications, react-native-llama.cpp |
| SQLite | On-device storage | Atomic writes, zero installation cost |

---

## 3. Datasets — Selection, Roles, and Limitations

| Dataset | Role | Type | Used for |
|---------|------|------|----------|
| StudentLife | Primary CPT training | Longitudinal | Structural EM — phone sensors + EMA + survey scales |
| LifeSnaps | Secondary CPT training + validation | Longitudinal | CPT training for wearable-derived nodes; CPT validation for sleep and energy nodes |
| NHANES | Population priors only | Cross-sectional | Prior table construction — never enters training CSV |

> **NHANES** is never merged into the training CSV under any circumstances. It has no date column and no temporal structure — it cannot teach the DBN how variables evolve over time. Its sole contribution is a stratified prior lookup table: `P(node_state | age_group, sex)` for self-report nodes. This prior is applied at runtime to initialise node beliefs for new users before any self-report evidence exists.

> **StudentLife** is the primary training source. 49 participants, longitudinal hourly observations, phone sensors, EMA, and validated survey scales (PHQ-9, PSS, PANAS, loneliness, VR-12, BigFive). All CPT edges between nodes are learned here via Structural EM. Survey columns shape CPTs during training and are discarded at runtime.

> **LifeSnaps** has two roles. First, it is a secondary training source for wearable-derived signals that StudentLife lacks — `hourly_bpm`, `prev_night_resting_hr`, `prev_night_minutesAsleep`. These signals train CPT edges for nodes like `heart_rate` and `sleep_quality` (wearable version). Second, it remains a validation source — after Structural EM converges on StudentLife, LifeSnaps log-likelihood checks whether CPTs generalise to a different population (European adults, 2021, Fitbit). 71 participants.

> **The old statement that "StudentLife, LifeSnaps, and GLOBEM are merged into one unified daily CSV" is incorrect and removed.** LifeSnaps enters training only via harmonised columns (Section 3.8), not as a row-concatenated merge. GLOBEM is not used in the current implementation.

---

## 3.5. Dataset Roles — Precise Separation

Understanding what each dataset contributes to the DBN prevents the most common implementation errors.

### What NHANES contributes

NHANES contributes exactly one thing: a prior probability table. For each self-report node (`alcohol_use`, `smoking`, `pain_level`), NHANES gives `P(node_state | age_group, sex)` — the population-level distribution before any individual user data exists.

```python
# NHANES → prior table only
# Example output structure saved to feature_node_config.json:
{
  "priors": {
    "alcohol_use": {
      "18-25_M": {"none": 0.44, "mild": 0.38, "high": 0.18},
      "18-25_F": {"none": 0.51, "mild": 0.35, "high": 0.14},
      "26-35_M": {"none": 0.39, "mild": 0.41, "high": 0.20},
      ...
    },
    "smoking": { ... },
    "pain_level": { ... }
  }
}
```

At runtime: new user onboards with age=22, sex=male → DBN initialises `alcohol_use` with `[0.44, 0.38, 0.18]` instead of uniform `[0.33, 0.33, 0.33]`. NHANES never touches any CPT edge. It only sets where a node starts before evidence arrives.

### What StudentLife contributes

StudentLife is the only dataset where the DBN actually **learns**. Structural EM runs here. HillClimbSearch finds the graph structure here. All CPT numbers — the conditional probability tables that define `P(mental_stress | mood, sleep, screen_time)` — come from StudentLife training rows.

StudentLife also contains survey columns for `smoking`, `alcohol_use`, and `pain_level` (via PHQ, VR-12, and lifestyle questionnaires). This means CPT edges between these self-report nodes and stress nodes are **learned automatically by Structural EM** — not injected manually.

### What LifeSnaps contributes

LifeSnaps contributes CPT signal for wearable-derived nodes that StudentLife cannot provide. No student in StudentLife wore a Fitbit. So `hourly_bpm → mental_stress`, `prev_night_resting_hr → sleep_quality` cannot be learned from StudentLife alone.

LifeSnaps rows enter the training pipeline only through harmonised columns (Section 3.8). For signals unique to LifeSnaps, the relevant training rows come exclusively from LifeSnaps participants. For signals present in both datasets, rows from both are combined in a single harmonised column.

LifeSnaps also validates: after training on StudentLife (and harmonised LifeSnaps), the learned CPTs are checked against LifeSnaps participants held out from training. Poor validation log-likelihood means the CPTs overfit to the StudentLife population.

---

## 3.6. Data Cleaning — Per Dataset

All cleaning steps are applied before any statistical preprocessing. The order within each dataset is: sentinel removal → range-based nullification → population filter → audit.

### NHANES Cleaning

From actual data inspection (9,254 rows × 52 cols, all float64):

1. Replace sentinel codes with NaN — these are NHANES "refused" / "don't know" encodings, not real values:
   - `PAD680` → replace 9999, 7777 with NaN
   - `ALQ130` → replace 999, 777 with NaN
   - `ALQ142` → replace 999, 777 with NaN
   - `SMQ040` → replace 7, 9 with NaN
   - `ALQ121` → replace 99 with NaN
   - `DPQ010` through `DPQ090` → replace 7, 9 with NaN
2. Nullify impossible BMI: `BMI <= 0` or `BMI > 80` → NaN (70 rows confirmed corrupt from inspection)
3. Nullify impossible age: `RIDAGEYR < 0` → NaN
4. Drop rows where `RIDAGEYR < 18` — population filter, not data error. App targets adults only. (~3,398 rows removed, ~5,856 remain)
5. Audit all 52 cols for remaining negative values where negative is physiologically impossible → NaN
6. Audit NaN rates post-cleaning, document per column — columns with >60% NaN after cleaning are structurally sparse due to NHANES submodule sampling. Do not impute these. Use available-case analysis only.

**Do not impute any NHANES values.** NHANES describes population distributions. Imputing values means inventing population statistics. Available-case analysis is correct: each row contributes to the prior computation for the columns it has valid data for.

### StudentLife Cleaning

From actual data inspection (91,536 rows × 131 cols, 49 uids, hourly granularity):

1. Fix `sleep_hours` broadcast corruption — **critical, do first before any other step.** Sleep hours is a daily survey value that was broadcast across all 24 hourly rows. The max of 214 and p95 of 103 confirm summation occurred somewhere downstream. Fix: take `first()` per `uid+date` group, then nullify values outside physiological range `[2, 14]` → NaN
2. Nullify impossible `screen_time_window_minutes > 60` → NaN (window is 60 minutes maximum)
3. Nullify `sedentary_ratio` or `active_ratio` outside `[0, 1]` → NaN
4. Nullify rows where `sedentary_ratio + active_ratio > 1.05` → set both to NaN (physically impossible)
5. Nullify `call_duration_total < 0` → NaN
6. Nullify `avg_running_tasks_window < 0` → NaN
7. Nullify `prev_evening_peak_running_tasks < 0` → NaN
8. Check all EMA cols (`mood_happy`, `mood_sad`, `mood_how`, `stress_ema_level`, `prev_day_sleep_ema_*`, `exercise_*`) for values outside their documented scale range → NaN
9. Check all PHQ cols against range `[0, 3]` — values outside → NaN
10. Check all PSS cols against range `[0, 4]` — values outside → NaN
11. Check all PANAS cols against range `[1, 5]` — values outside → NaN
12. Check all loneliness cols against range `[1, 3]` — values outside → NaN
13. Check all VR-12 cols against range `[1, 5]` — values outside → NaN
14. Check all BigFive cols (`e_talkative`, `n_depressed`, etc.) against range `[1, 5]` — values outside → NaN
15. Check `prev_day_sleep_ema_hours` outside `[0, 14]` → NaN
16. Confirm `hour` col contains only integers `0–23` — values outside → NaN
17. Confirm `date` col parses cleanly as datetime — flag malformed rows
18. Audit NaN rates post-cleaning, **separately** for sensor cols, survey cols, and EMA cols — these have structurally different missingness patterns and must not be treated identically

### LifeSnaps Cleaning

From actual data inspection (159,508 rows × 33 cols, 71 ids, hourly granularity):

1. Fix `age` column — currently all NaN at row level due to join failure. Extract per-id static value and map back:
   ```python
   age_lookup = df3.groupby('id')['age'].apply(
       lambda x: pd.to_numeric(x, errors='coerce').dropna().iloc[0]
       if pd.to_numeric(x, errors='coerce').notna().any() else np.nan
   )
   df3['age'] = df3['id'].map(age_lookup)
   ```
2. Cast `age` to numeric — mixed dtype warning flagged on load. Non-numeric → NaN
3. Cast `bmi` to numeric — mixed dtype warning flagged on load. Non-numeric → NaN
4. Nullify impossible `hourly_bpm < 30` or `hourly_bpm > 200` → NaN
5. Nullify impossible `prev_night_minutesAsleep > 720` (12 hours) or `< 0` → NaN
6. Nullify `hourly_steps < 0` → NaN
7. Nullify `prev_day_steps < 0` or `> 100000` → NaN
8. Nullify `daily_active_ratio` or `daily_sedentary_ratio` outside `[0, 1]` → NaN
9. Nullify rows where `daily_active_ratio + daily_sedentary_ratio > 1.05` → set both to NaN
10. Nullify `prev_night_resting_hr < 30` or `> 120` → NaN
11. Nullify `prev_night_temp_variation` > ±5°C → NaN (implausible variation)
12. Check all EMA cols (`today_*`, `prev_day_*` mood cols) for values outside documented scale → NaN
13. Encode `gender`: `'MALE'` → 1, `'FEMALE'` → 2, anything else → NaN. Keep original col
14. Drop structurally absent cols — not node decisions, purely data quality: `prev_night_spo2` (81% NaN — device not available to most participants), `prev_day_scl_avg` (97.9% NaN — near-absent), `prev_day_stress_score` (72% NaN — derived score, not raw signal)
15. Confirm `hour` contains only `0–23`
16. Confirm `date` parses as datetime
17. Audit NaN rates post-cleaning per column

---

## 3.65. Column Pruning and Validated Scale Merges

Applied after data cleaning (Section 3.6) and before statistical preprocessing (Section 3.7). These decisions are based on correlation/VIF analysis and validated scale design conventions. Reducing redundant columns before normalization and discretisation prevents noise and collinearity from inflating CPT complexity.

### StudentLife — Validated Scale Merges

Validated psychological scales must be scored as their authors intended. Individual items are merged into composite scores before any statistical preprocessing.

| Survey | Items used | Merge rule | Result column | Score range |
|--------|-----------|-----------|--------------|-------------|
| PHQ-9 | 9 items (`phq_interest` → `phq_death`) | Sum | `phq_total` | 0–27 |
| PSS-10 | 10 items (`pss_1` → `pss_10`) | Reverse-score items 4, 5, 7, 8 (as `4 − item`), then sum | `pss_total` | 0–40 |
| PANAS | 18 items (14 used, 4 removed) | Positive affect sum (9 items) | `panas_pa` | 9–45 |
| PANAS | same | Negative affect sum (9 items) | `panas_na` | 9–45 |
| UCLA Loneliness-20 | 20 items (`lonely_1` → `lonely_20`) | Sum | `lonely_total` | 20–60 |
| BigFive | `e_*` cols (extraversion items) | Sum | `extraversion` | varies |
| BigFive | `n_*` cols (neuroticism items) | Sum | `neuroticism` | varies |
| VR-12 | 11 items | Keep `vr_general_health` + `vr_pain_interference` only; drop remaining 9 | 2 cols | 1–5 each |

> **PSS reverse scoring:** items 4, 5, 7, 8 are phrased positively — score as `4 − item_value` before summing. Standard PSS-10 scoring protocol.

> **PANAS split:** PANAS Positive Affect items: interested, strong, enthusiastic, active, alert, inspired, attentive (+ determined, proud — removed from the original 20 per v7.0). PANAS Negative Affect items: distressed, upset, scared, irritable, nervous, jittery, afraid (+ guilty, hostile — removed). Do not merge into a single score — positive and negative affect are orthogonal dimensions, not opposites.

> **VR-12 decision:** only `vr_general_health` maps to the `general_health` DBN node; `vr_pain_interference` maps to `pain_level`. The other 9 VR-12 items do not have corresponding runtime nodes. Keeping them adds columns without improving CPT learning for any node in the graph.

### StudentLife — Statistical Drops (Correlation/VIF)

| Column dropped | Reason |
|----------------|--------|
| `sedentary_ratio` | r = 1.0 with `active_ratio` — perfect multicollinearity by construction (`sedentary + active = 1.0`). Redundant |
| `prev_day_sedentary_ratio` | r = 1.0 with `prev_day_active_ratio` — same reason |
| `exercise_have` | r = 0.833 with `exercise_type` — near-perfect collinearity; `exercise_type` is more informative |

### LifeSnaps — Statistical Drops (Correlation/VIF)

| Column dropped | Reason |
|----------------|--------|
| `daily_sedentary_ratio` | r = 1.0 with `daily_active_ratio` — redundant by construction |

> Any additional LifeSnaps pairs with r > 0.85 after the full correlation pass should be reviewed. Keep the column with lower NaN rate or stronger physiological meaning for its mapped DBN node.

---

## 3.7. Statistical Preprocessing — Per Dataset

Applied after cleaning, in the order listed. The train/val split happens before any threshold-based computation to prevent leakage.

### Order of Operations

```
NHANES:
  clean  →  available-case prior computation  →  save prior JSON  →  done

STUDENTLIFE:
  clean  →  uid-level train/val split  →  per-uid winsorise  →
  per-uid z-score (continuous sensors only)  →
  min-max normalise (survey scale cols only)  →
  survey forward-fill within uid  →
  flag training-eligible rows  →  done

LIFESNAPS:
  clean  →  per-id winsorise  →  physiological bounds  →
  per-id z-score (continuous sensor cols)  →
  EMA cols: keep NaN as-is (structural absence, not error)  →  done
```

### Train/Val Split — StudentLife, Uid Level

**This must happen before any threshold computation.** Fitting thresholds on the full dataset leaks val uid distributions into the cleaning parameters — a silent form of data leakage.

```python
import numpy as np
np.random.seed(42)
uids = df['uid'].unique()  # 49 uids
np.random.shuffle(uids)
split_idx = int(0.8 * len(uids))
train_uids = uids[:split_idx]   # 39 uids
val_uids   = uids[split_idx:]   # 10 uids

train_df = df[df['uid'].isin(train_uids)]
val_df   = df[df['uid'].isin(val_uids)]
```

All subsequent threshold computations — winsorsation bounds, z-score means/stds, discretisation thresholds — are computed on `train_df` only and then applied to `val_df`.

### Per-Uid Winsorsation — StudentLife

Global winsorsation is incorrect for this dataset. One student averages 45 min screen time/hour; another averages 5 min. Global 99th percentile flattens real individual variation that the DBN should learn.

```python
sensor_cols = [
    'sedentary_ratio', 'active_ratio',
    'screen_time_window_minutes', 'dark_window_minutes',
    'unlocked_window_minutes', 'avg_running_tasks_window',
    'call_count', 'call_duration_total', 'sms_count'
]

def uid_winsorise(df, col, lower=0.01, upper=0.99):
    return df.groupby('uid')[col].transform(
        lambda x: x.clip(x.quantile(lower), x.quantile(upper))
    )

for col in sensor_cols:
    df[col] = uid_winsorise(df, col)
```

### Within-Uid Z-Score — StudentLife Continuous Sensors

Raw sensor values are not comparable across uids — one student's "high screen time" is another's minimum. Z-scoring within uid puts all participants on the same relative scale before discretisation.

```python
cols_to_normalise = [
    'screen_time_window_minutes', 'dark_window_minutes',
    'unlocked_window_minutes', 'avg_running_tasks_window',
    'call_count', 'call_duration_total', 'sms_count',
    'sleep_hours'
]

for col in cols_to_normalise:
    df[col + '_z'] = df.groupby('uid')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
```

**Do NOT z-score:**
- `sedentary_ratio`, `active_ratio` — already 0-1, ratio is meaningful as-is
- EMA cols — fixed ordinal scale, absolute value matters for CPT learning
- PHQ/PSS/PANAS/loneliness/VR-12/BigFive — validated instrument scores, normalising destroys their meaning

### Min-Max Normalisation — Survey Scale Cols

PHQ-9, PSS, PANAS, loneliness, VR-12, and BigFive items are on different scales (0-3, 0-4, 1-5, 1-3). Normalise all to 0-1 for comparability before Structural EM. Do not z-score these — their absolute position within the validated scale carries clinical meaning.

```python
scale_ranges = {
    # PHQ-9 items: 0-3
    'phq_interest': (0, 3), 'phq_depressed': (0, 3), 'phq_sleep': (0, 3),
    'phq_tired': (0, 3), 'phq_appetite': (0, 3), 'phq_failure': (0, 3),
    'phq_concentrate': (0, 3), 'phq_psychomotor': (0, 3), 'phq_death': (0, 3),
    # PSS items: 0-4
    'pss_1': (0, 4), 'pss_2': (0, 4), 'pss_3': (0, 4), 'pss_4': (0, 4),
    'pss_5': (0, 4), 'pss_6': (0, 4), 'pss_7': (0, 4), 'pss_8': (0, 4),
    'pss_9': (0, 4), 'pss_10': (0, 4),
    # PANAS items: 1-5
    'panas_interested': (1, 5), 'panas_distressed': (1, 5), 'panas_upset': (1, 5),
    'panas_strong': (1, 5), 'panas_scared': (1, 5), 'panas_enthusiastic': (1, 5),
    'panas_active': (1, 5), 'panas_irritable': (1, 5), 'panas_alert': (1, 5),
    'panas_inspired': (1, 5), 'panas_nervous': (1, 5), 'panas_attentive': (1, 5),
    'panas_jittery': (1, 5), 'panas_afraid': (1, 5),
    # Loneliness items: 1-3
    **{f'lonely_{i}': (1, 3) for i in range(1, 21)},
    # VR-12 items: 1-5
    'vr_general_health': (1, 5), 'vr_moderate_activity': (1, 5),
    'vr_climb_stairs': (1, 5), 'vr_physical_limit_work': (1, 5),
    'vr_physical_limit_kind': (1, 5), 'vr_emotional_limit_work': (1, 5),
    'vr_emotional_limit_care': (1, 5), 'vr_pain_interference': (1, 5),
    'vr_energy': (1, 5), 'vr_downhearted': (1, 5), 'vr_social_interference': (1, 5),
    # BigFive items: 1-5
    'e_talkative': (1, 5), 'n_depressed': (1, 5), 'e_reserved_r': (1, 5),
    'n_tense': (1, 5), 'n_worries': (1, 5), 'e_quiet_r': (1, 5),
    'n_stable_r': (1, 5), 'n_moody': (1, 5), 'e_sociable': (1, 5),
    'n_nervous': (1, 5),
}

for col, (lo, hi) in scale_ranges.items():
    if col in df.columns:
        df[col + '_norm'] = (df[col] - lo) / (hi - lo)
```

### Survey Forward-Fill Within Uid — StudentLife

Surveys are answered once or a few times per participant, not per window. The same value is valid across all window rows for that uid throughout the study period. Forward-fill (then backward-fill to catch uids who answered mid-study) within uid:

```python
survey_cols = [c for c in df.columns if any(
    c.startswith(p) for p in ['phq_','pss_','panas_','lonely_','vr_','e_','n_']
)]

df[survey_cols] = df.groupby('uid')[survey_cols].transform(
    lambda x: x.ffill().bfill()
)
# 3 uids have zero survey responses — NaN remains for these uids
# Do not impute — cannot put another person's survey data onto an individual
```

### Training-Eligible Row Flagging — StudentLife

From inspection: sensor NaN and survey NaN have partially independent causes. Sensor NaN = dead window. Survey NaN = absent response. Define training-eligible rows explicitly:

```python
df['training_eligible'] = (
    df['sedentary_ratio'].notna() &   # sensor alive (activity as validator)
    df['phq_interest'].notna()         # survey present
)
# ~59,987 eligible rows from 46 uids (3 uids have no survey data at all)
```

### Per-Id Winsorsation — LifeSnaps

Same logic as StudentLife. Per participant, not global:

```python
lifesnaps_sensor_cols = [
    'hourly_steps', 'hourly_bpm',
    'prev_night_minutesAsleep', 'prev_day_steps', 'prev_night_resting_hr'
]

for col in lifesnaps_sensor_cols:
    df3[col] = pd.to_numeric(df3[col], errors='coerce')
    df3[col] = df3.groupby('id')[col].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
    )
```

### Per-Id Z-Score — LifeSnaps Continuous Sensors

```python
ls_normalise_cols = [
    'hourly_steps', 'hourly_bpm',
    'prev_night_minutesAsleep', 'prev_day_steps', 'prev_night_resting_hr'
]

for col in ls_normalise_cols:
    df3[col + '_z'] = df3.groupby('id')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
```

### NHANES Prior Table Construction

NHANES does not go through the standard preprocessing pipeline. Its output is a prior lookup table, not a training CSV.

```python
df_nhanes = pd.read_csv('nhanes_merged_cleaned.csv')
# [apply cleaning steps from Section 3.6 first]

# Bin age
df_nhanes['age_group'] = pd.cut(
    df_nhanes['RIDAGEYR'],
    bins=[18, 25, 35, 50, 80],
    labels=['18-25', '26-35', '36-50', '50+']
)

# Compute weighted marginal per node per stratum
# (no survey weights in current file — use count-based)
def compute_prior(df, node_col, state_labels):
    return (
        df.groupby(['age_group', 'RIAGENDR', node_col])
          .size()
          .groupby(level=[0, 1])
          .transform(lambda x: x / x.sum())
          .rename('probability')
          .reset_index()
    )

# Save to feature_node_config.json under 'priors' key
# Applied at runtime in evidence fusion when data_source = 'prior_only'
```

---

## 3.8. Data Harmonisation — Multi-Dataset Training Architecture

### The Core Problem

StudentLife and LifeSnaps measure overlapping and non-overlapping constructs with different instruments. Simply concatenating all rows into one CSV creates two problems: heavy cross-dataset NaN (every LifeSnaps row has NaN for StudentLife-only columns and vice versa), and measurement non-equivalence (same construct, different instruments, potentially different distributions).

The solution is a four-category node architecture that determines how each node's training data is assembled.

### Four Node Training Source Categories

| Category | Description | Example nodes | NaN behaviour in training CSV |
|----------|-------------|--------------|-------------------------------|
| **StudentLife-only** | Signal exists only in StudentLife | `screen_time`, `social_activity`, `app_usage` | LifeSnaps rows have NaN for these cols — expected |
| **LifeSnaps-only** | Signal exists only in LifeSnaps | `heart_rate`, `sleep_quality_wearable` | StudentLife rows have NaN for these cols — expected |
| **Harmonised** | Same construct in both datasets — concatenated into one column | `activity_level`, `sedentary_time` | No NaN from dataset source — both contribute real values |
| **NHANES-prior-informed** | Self-report nodes — exist in DBN, prior from NHANES, CPT edges from StudentLife survey cols | `alcohol_use`, `smoking`, `pain_level` | Standard StudentLife rows — no LifeSnaps contribution |

### Why Cross-Dataset NaN Is Acceptable for Non-Harmonised Cols

Structural EM handles column-level NaN correctly — it marginalises over missing variables rather than conditioning on them. The CPT for `heart_rate → mental_stress` is learned purely from LifeSnaps rows where `heart_rate` is present. The CPT for `screen_time → mental_stress` is learned purely from StudentLife rows. They do not interfere with each other.

The one weakness: joint relationships between one StudentLife-only column and one LifeSnaps-only column cannot be learned from any row, because no single row has both. Cross-dataset CPT edges will be estimated from marginals rather than joint observations. This is an acceptable tradeoff — the alternative (dropping all LifeSnaps-only or all StudentLife-only columns) loses more signal than the NaN cost.

After Structural EM converges, flag any learned edges that cross the dataset boundary (one parent from a StudentLife-only column, one from a LifeSnaps-only column). Treat those specific CPT entries with appropriately lower confidence. Do not drop them.

### Harmonised Column Construction

For nodes where both datasets have a valid proxy for the same construct:

**Step 1 — Distribution overlap check.**

Before concatenating, verify the two columns' distributions actually overlap. If they don't, combining them raw means the same discretisation threshold cuts the two populations differently.

```python
import matplotlib.pyplot as plt

# Check for activity_level:
print(train_df['active_ratio'].describe())          # StudentLife
print(df3['daily_active_ratio'].describe())          # LifeSnaps

# Visual check
fig, axes = plt.subplots(1, 2)
train_df['active_ratio'].hist(ax=axes[0], bins=30)
df3['daily_active_ratio'].hist(ax=axes[1], bins=30)
plt.show()
```

**Step 2 — Scale based on overlap result.**

- Distributions overlap well → min-max normalise within each dataset, then concatenate
- Distributions diverge significantly → z-score within each dataset, then concatenate

```python
# Case A: distributions overlap — min-max normalise
sl_min, sl_max = train_df['active_ratio'].min(), train_df['active_ratio'].max()
studentlife['activity_unified'] = (studentlife['active_ratio'] - sl_min) / (sl_max - sl_min)

ls_min, ls_max = df3['daily_active_ratio'].min(), df3['daily_active_ratio'].max()
lifesnaps['activity_unified'] = (df3['daily_active_ratio'] - ls_min) / (ls_max - ls_min)

# Case B: distributions diverge — z-score within dataset
studentlife['activity_unified'] = (
    (studentlife['active_ratio'] - train_df['active_ratio'].mean())
    / train_df['active_ratio'].std()
)
lifesnaps['activity_unified'] = (
    (df3['daily_active_ratio'] - df3['daily_active_ratio'].mean())
    / df3['daily_active_ratio'].std()
)
```

**Step 3 — Add dataset source flag.**

```python
studentlife['dataset_source'] = 0   # 0 = StudentLife
lifesnaps['dataset_source']   = 1   # 1 = LifeSnaps
```

**Step 4 — Concatenate.**

```python
combined = pd.concat([studentlife, lifesnaps], axis=0, ignore_index=True)
# activity_unified: real value for every row (no cross-dataset NaN)
# screen_time_window_minutes: real for StudentLife rows, NaN for LifeSnaps rows
# hourly_bpm: NaN for StudentLife rows, real for LifeSnaps rows
```

### Confirmed Harmonised Columns

Based on actual column inspection of both datasets:

| Unified column | StudentLife source | LifeSnaps source | Scale check needed |
|---------------|-------------------|-----------------|-------------------|
| `activity_unified` | `active_ratio` | `daily_active_ratio` | Yes — different instruments |
| `sedentary_unified` | `sedentary_ratio` | `daily_sedentary_ratio` | Yes |
| `sleep_hours_unified` | `sleep_hours` (EMA) | `prev_night_minutesAsleep` / 60 | Yes — EMA vs wearable |

### Discretisation — Thresholds Fit on Training Split Only

After harmonised columns are built, discretise into the 3-state node categories the DBN uses. **Thresholds must be computed on StudentLife training uids only**, then applied to StudentLife val uids, LifeSnaps rows, and at runtime — identically.

```python
# Fit on training split only
thresholds = {}
for col, node_name in [
    ('activity_unified', 'activity_level'),
    ('sedentary_unified', 'sedentary_time'),
    ('screen_time_window_minutes', 'screen_time'),
]:
    q33 = train_df[col].quantile(0.33)
    q67 = train_df[col].quantile(0.67)
    thresholds[node_name] = {'q33': q33, 'q67': q67}
    # Apply to full dataset
    combined[node_name] = pd.cut(
        combined[col],
        bins=[-np.inf, q33, q67, np.inf],
        labels=['low', 'medium', 'high']
    )

# Save thresholds — used at runtime for identical discretisation
import json
with open('feature_node_config.json', 'w') as f:
    json.dump({'discretisation_thresholds': thresholds}, f)
```

### Final Training CSV Structure

```
One row per (uid/id, date, hour)

Identity cols:        uid, date, hour, dataset_source
Harmonised cols:      activity_unified, sedentary_unified, sleep_hours_unified
                      → these have real values for all rows
SL-only cols:         screen_time_window_minutes, avg_running_tasks_window,
                      call_count, sms_count, dark_window_minutes, unlocked_window_minutes
                      → NaN for LifeSnaps rows
LS-only cols:         hourly_bpm, prev_night_resting_hr, prev_night_minutesAsleep
                      → NaN for StudentLife rows
Survey cols:          phq_*, pss_*, panas_*, lonely_*, vr_*, bigfive_*
                      → StudentLife rows only (after forward-fill within uid)
Discretised nodes:    activity_level, sedentary_time, screen_time, ...
                      → used as DBN node columns in Structural EM
Target latent nodes:  mental_stress, physical_stress
                      → filled by E-step during Structural EM
```

---

## 4. DBN Node Taxonomy — Three Layers

### Layer 1 — Passive Observable Nodes

Directly computed from phone sensors. No SLM involvement. These nodes are never asked about because the phone fills them automatically.

| Node | Source signal | States | Merge mode | Passive proxy strength | Training source |
|------|--------------|--------|------------|----------------------|-----------------|
| `activity_level` | Accelerometer steps | low / medium / high | replace | 0.75 | Harmonised (SL + LS) |
| `sleep_duration` | HealthKit or screen-off proxy | poor / adequate / excessive | replace | 0.85 | StudentLife primary |
| `sleep_quality` | HRV via HealthKit, fallback: night screen | poor / adequate / good | replace | 0.70 | Harmonised (SL + LS wearable) |
| `screen_time` | Screen on/off events + self-report additive | low / medium / high | **additive** | 0.80 | StudentLife-only |
| `sedentary_time` | Accelerometer still periods | low / medium / high | replace | 0.75 | Harmonised (SL + LS) |
| `social_activity` | Call log + SMS count + notification count | low / medium / high | replace | 0.50 | StudentLife-only |
| `heart_rate` | Wearable BPM | low / normal / elevated | replace | 0.90 | LifeSnaps-only |

> `social_activity` proxy strength is 0.50 because call_log and SMS miss WhatsApp, Signal, and FaceTime. `notification_count` is added as a supplementary proxy. **This does not read notification content.**

> `screen_time` uses **additive** merge mode. The passive dark sensor captures phone screen time only. Self-reported additional screen time (laptop, TV) is added to the passive baseline.

> `heart_rate` is a LifeSnaps-only node. It has no StudentLife equivalent. Its CPT edges (`heart_rate → physical_stress`, `heart_rate → mental_stress`) are learned exclusively from LifeSnaps rows. At runtime, this node is filled by a connected wearable if available, or remains at prior if no wearable is present.

### Layer 2 — Self-Reported Nodes

Populated only when the user says something. Override passive proxies when present. Stored with expiry dates.

> **IMPORTANT:** The nodes shown below are the confirmed minimum initial set. The rule "one question = one node" applies. The proactive question rule applies to ANY node with zero passive proxy, high CPT-spread weight, and confidence below threshold.

| Node | States | Passive proxy | Merge mode | Proactive question? | Training source | NHANES prior? |
|------|--------|--------------|------------|---------------------|-----------------|---------------|
| `pain_level` | none / mild / high | None | replace | **Yes** | StudentLife (VR-12 col) | Yes |
| `mood` | low / neutral / good | Proxied by sleep + social + activity | replace | No | StudentLife (PANAS, EMA) | No |
| `energy_level` | low / medium / high | Proxied by sleep + activity | replace | No | StudentLife (PSQI, PANAS) | No |
| `physical_exercise` | none / low / high | Weak — misses gym, swimming, cycling | replace | No | StudentLife (EMA exercise) | No |
| `alcohol_use` | none / mild / high | None | replace | **Yes** | StudentLife (lifestyle survey) | Yes |
| `smoking` | none / mild / high | None | replace | **Yes** | StudentLife (lifestyle survey) | Yes |

### Layer 3 — Latent Inferred Nodes

Never directly observed. Computed purely from DBN inference. `allows_self_report = false` for both.

> This is a fundamental design constraint. If these nodes allowed self-report, a user saying "I am not stressed" would override DBN inference even when passive data shows 3 hours of sleep and zero steps. That collapses the app into a mood diary.

#### `physical_stress` (low / moderate / high)
- High weight contributors: `pain_level`, `sedentary_time`, `activity_level`
- Medium: `sleep_quality`, `physical_exercise`, `heart_rate`
- Low: `energy_level`
- Weights derived from CPT spread post-training — not hardcoded.

#### `mental_stress` (low / moderate / high)
- High weight contributors: `mood`, `sleep_duration`, `screen_time`
- Medium: `social_activity`, `energy_level`, `sleep_quality`, `alcohol_use`, `heart_rate`
- Low: `sedentary_time`, `smoking`
- Weights derived from CPT spread post-training — not hardcoded.

---

## 4.5. Node Training Source Categories

Each node belongs to exactly one training source category. This determines how its training data is assembled.

### Category 1 — StudentLife-Only Nodes

Signal exists only in StudentLife. LifeSnaps rows contribute NaN for these columns — expected and acceptable.

- `screen_time` ← `screen_time_window_minutes`
- `social_activity` ← `call_count`, `sms_count`, notification data
- `app_usage` ← `avg_running_tasks_window`

CPT edges for these nodes are learned exclusively from StudentLife training rows.

### Category 2 — LifeSnaps-Only Nodes

Signal exists only in LifeSnaps. StudentLife rows contribute NaN for these columns.

- `heart_rate` ← `hourly_bpm`, `prev_night_resting_hr`

CPT edges for these nodes are learned exclusively from LifeSnaps rows. At runtime, these nodes are filled only if a wearable is connected. If no wearable exists, the node sits at its NHANES-equivalent population prior (or uniform prior if no NHANES data applies).

### Category 3 — Harmonised Nodes (StudentLife + LifeSnaps)

Same construct, compatible instruments, distribution-checked, normalised, concatenated into one unified column. No cross-dataset NaN for these columns — every row has a real value.

- `activity_level` ← `activity_unified` (SL: `active_ratio`, LS: `daily_active_ratio`)
- `sedentary_time` ← `sedentary_unified` (SL: `sedentary_ratio`, LS: `daily_sedentary_ratio`)
- `sleep_hours_unified` ← (SL: `sleep_hours` EMA, LS: `prev_night_minutesAsleep` / 60)

CPT edges for these nodes are learned from all rows — both StudentLife and LifeSnaps contribute.

### Category 4 — NHANES-Prior-Informed Nodes

Self-report nodes. They exist in the DBN with edges learned from StudentLife survey columns. NHANES provides their starting prior `P(node_state | age, sex)`. They are never learned from LifeSnaps (LifeSnaps has no equivalent survey data).

- `alcohol_use` — StudentLife lifestyle survey cols; NHANES ALQ cols → prior
- `smoking` — StudentLife lifestyle survey cols; NHANES SMQ cols → prior
- `pain_level` — StudentLife VR-12 pain col; NHANES CDQ/MCQ cols → prior

---

## 4.6. Manual Edge Injection for Self-Report Nodes

### Why This Section Exists

A node with no edges in a Bayesian network is completely inert. It updates its own state when evidence arrives, but that update propagates nowhere — no other node sees it, no downstream effect on `mental_stress` or `physical_stress`. An isolated node does not affect inference regardless of what value it holds.

For `smoking`, `alcohol_use`, and `pain_level`: these nodes have corresponding survey columns in StudentLife (`lifestyle survey`, `VR-12 pain_interference`). Structural EM will discover and learn their edges automatically from temporal co-occurrence in training rows. No manual injection needed for the current dataset.

### When Manual Edge Injection Is Required

If a future self-report node is added that has no corresponding column in either StudentLife or LifeSnaps, Structural EM cannot discover its edges — it has no co-occurrence data. In this case:

1. Assert the edge exists based on peer-reviewed medical literature
2. Inject it into pgmpy before EM runs
3. EM estimates the CPT strength from whatever partial data exists

```python
from pgmpy.models import DynamicBayesianNetwork

dbn = DynamicBayesianNetwork()

# Manually inject edges for nodes with no training data coverage
# Direction: established by medical literature, not data
dbn.add_edge(('smoking', 0), ('physical_stress', 1))       # smoking(t) → physical_stress(t+1)
dbn.add_edge(('smoking', 0), ('sleep_quality', 1))         # smoking(t) → sleep_quality(t+1)
dbn.add_edge(('alcohol_use', 0), ('sleep_quality', 1))     # alcohol(t) → sleep_quality(t+1)
dbn.add_edge(('alcohol_use', 0), ('mental_stress', 1))     # alcohol(t) → mental_stress(t+1)
dbn.add_edge(('pain_level', 0), ('physical_stress', 1))    # pain(t) → physical_stress(t+1)
dbn.add_edge(('pain_level', 0), ('mental_stress', 1))      # pain(t) → mental_stress(t+1)

# Then run Structural EM — it will estimate CPT values for all edges,
# including the manually injected ones, from whatever data is available.
# CPTs for manually injected edges will be weaker but directionally correct.
```

> Manual edge injection is defensible for these nodes because the causal directions are established in medical literature, not assumed. `smoking → physical_stress` and `alcohol_use → sleep_quality → mental_stress` are not hypotheses. They are known causal pathways. The CPT value (the *strength* of the relationship) is estimated by EM from data; only the *existence* of the edge is asserted manually.

---

## 5. Training vs Runtime — The Full Separation

### What Structural EM Does

Structural EM alternates between two steps until convergence:

- **E-step:** Uses Loopy Belief Propagation to estimate latent node values given the current structure and CPTs — passes messages between all connected nodes until beliefs stabilise, producing soft probability labels for `mental_stress` and `physical_stress` on each training row.
- **M-step:** Uses HillClimbSearch with BIC score to find the optimal graph structure and CPT values given the now-labelled data.

This is why survey columns disappear after training: the E-step compresses PHQ-9, PSS, PANAS, BigFive etc. into CPT numbers. After training, those CPT numbers capture everything the survey columns taught.

```
# Structural EM — simplified flow

Training data row (StudentLife):
  phq_depressed=2.1, panas_nervous=3.4, sleep_hours=5.1, screen_time=420

E-step (Loopy Belief Propagation):
  P(mental_stress=high | all observed) = 0.76
  → soft label assigned to latent node

M-step (HillClimbSearch):
  Finds that mood + sleep + screen_time are the strongest predictors
  → CPT written: P(mental_stress=high | mood=low, sleep=poor, screen=high) = 0.78

After training:
  phq_depressed → encoded in CPT → column gone
  panas_nervous → encoded in CPT → column gone
  mood_happy    → this becomes the runtime 'mood' node
```

### Structural EM — Implementation Optimisations

Unoptimised Structural EM on ~60,000 training rows is bottlenecked by the E-step (VE inference per row), not the M-step HillClimbSearch. Optimisations below are ordered by impact.

#### Priority Order

| Priority | Optimisation | Targets | Impact |
|----------|-------------|---------|--------|
| 1 | **Evidence deduplication** | E-step | Huge — discrete rows repeat; pre-compute MAP per unique pattern, map back. |
| 2 | **Parallelisation** | E-step | High — rows are fully independent; distribute across CPU cores. |
| 3 | **Data subsampling** | E-step | High — use 20% stratified sample in early iterations, 100% in final 2–3. |
| 4 | **Strict search space constraints** | E-step + M-step | Medium — blacklist impossible edges (temporal violations, medically illogical); whitelist forced edges; cap max parents. Fewer parents → smaller CPT tables → faster VE. |
| 5 | **Warm start (initial skeleton)** | Iterations | Medium — run HillClimbSearch independently per dataset first; use result as starting DAG. Reduces total iterations needed. |
| 6 | **Incremental structural search** | M-step | Low — limit HillClimb to 5–10 steps in early (noisy) iterations; increase depth only as parameters stabilise. |
| 7 | **Structural caching** | M-step | Low — cache BIC scores per parent-child config; reuse if node's parents unchanged from previous iteration. Complex to implement; skip unless M-step is still a bottleneck after the above. |

#### Inference Method

**Implemented: True Loopy BP (max-product)** — messages passed directly on the original DAG (forward pass: roots→leaves, backward pass: leaves→roots) until convergence. No junction tree construction. Chosen over VariableElimination because LBP scales linearly with edges while VE scales exponentially with treewidth; for the 37-node graph with dense connections, LBP is significantly faster at E-step scale.

**Soft evidence via likelihood tables:** `data_likelihood_tables.py` precomputes `P(observed_bin | node_state)` for each node from training data. During the E-step each observed column contributes a likelihood vector rather than a hard discrete assignment — preserving more information from continuous signals.

> VariableElimination remains valid as a fallback if LBP convergence fails on specific graph configurations. In practice the 37-node graph converges reliably within a few message-passing iterations.

### What Exists at Runtime

| Runtime node | Filled by | Training source | Survey cols that shaped CPT |
|-------------|-----------|-----------------|----------------------------|
| `activity_level` | Accelerometer (daily) | Harmonised | PAD615, PAQ605, exercise EMA, LS active_ratio |
| `sleep_duration` | HealthKit screen-off (daily) | StudentLife | PSQI sleep_hours, PHQ-9 phq_sleep, DPQ030 |
| `sleep_quality` | HRV proxy (daily) | Harmonised | PSQI sleep_quality_rating, PSS items, LS resting_hr |
| `screen_time` | Dark sensor + self-report (daily) | StudentLife-only | StudentLife dark sensing, EMA study spaces |
| `sedentary_time` | Accelerometer (daily) | Harmonised | PAD680, EMA activity, LS sedentary_ratio |
| `social_activity` | Call log + SMS + notifications (daily) | StudentLife-only | Loneliness scale, BigFive extraversion |
| `heart_rate` | Wearable BPM if connected (daily) | LifeSnaps-only | LS hourly_bpm, prev_night_resting_hr |
| `pain_level` | User self-report only | StudentLife (manual edges) | VR-12 pain interference, MCQ160A/N, CDQ001 |
| `mood` | User self-report or passive proxy | StudentLife | PANAS, EMA Mood, n_depressed, vr_downhearted |
| `energy_level` | Passive proxy + self-report | StudentLife | PSQI low_enthusiasm, PANAS active/alert, phq_tired |
| `physical_exercise` | User self-report + weak accelerometer | StudentLife | EMA Exercise have/type, PAQ605, PAD615 |
| `alcohol_use` | User self-report (onboarding + volunteered) | StudentLife (manual edges) | ALQ111, ALQ121, ALQ130, ALQ142 |
| `smoking` | User self-report (onboarding + volunteered) | StudentLife (manual edges) | SMQ020, SMQ040, SMD641, SMD650 |
| `mental_stress` | DBN inference only | StudentLife + Harmonised | PHQ-9, PSS, PANAS, BigFive neuroticism |
| `physical_stress` | DBN inference only | StudentLife + Harmonised | VR-12 pain/health, cardiovascular, diabetes |

---

## 5.5. Sub-Dimension User Profile Attribute Layer

This layer is the solution to the personalisation vs overfitting tradeoff. The DBN cannot have 40+ nodes with only ~60,000 training rows — it would overfit. But the user deserves pointed, personalised questions from validated psychological scales. The sub-dimension layer decouples these two concerns.

### What Sub-Dimension Attributes Are

Every item from every survey scale — PHQ-9 items, PSS items, PANAS items, loneliness items, VR-12 items, BigFive items — becomes a sub-dimension attribute stored in SQLite in `user_profile_attributes`. **They are NOT DBN nodes.** They do not appear in the DBN graph. They exist only to enrich the evidence fed into the DBN's coarse nodes.

### How Sub-Dimension Attributes Are Filled

| Method | When | Example |
|--------|------|---------|
| Contextual follow-up | After user volunteers a statement | User: "I feel lonely" → system asks: "Is it more that you have no one to confide in, or that you feel left out of groups?" |
| Daily notification | Once per day at user-set time | "To improve your insight: How often did you feel hopeless this week?" |
| Organic chat extraction | User mentions something specific | "I can't concentrate on anything lately" → SLM maps to `phq_concentrate` |

### Expiry per Attribute Type

| Attribute type | Example | Expiry | Reason |
|---------------|---------|--------|--------|
| Stable personality trait | `neuroticism_score`, `extraversion_score` | 90-180 days | Personality is stable for months |
| Chronic state | `lonely_intimate`, `lonely_relational` | 14-30 days | Slowly shifting social context |
| Weekly pattern | PSS items (perceived stress) | 7 days | PSS framed as "past month" but weekly refresh appropriate |
| Daily state | PANAS items, PHQ mood items | 1-3 days | Momentary affect changes daily |
| Acute symptom | `pain_during_sleep`, `phq_sleep` | 1 day | Changes nightly |

### SQLite Table — `user_profile_attributes`

```sql
CREATE TABLE IF NOT EXISTS user_profile_attributes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    node_name      TEXT NOT NULL,
    node_value     REAL NOT NULL,   -- normalised 0.0 - 1.0
    raw_value      REAL,            -- original scale value
    scale_name     TEXT,            -- 'PHQ-9', 'PSS', 'UCLA', etc.
    confidence     REAL,
    data_source    TEXT,            -- 'self_report' / 'inferred'
    expires_date   TEXT NOT NULL,
    last_updated   TEXT,
    created_at     TEXT DEFAULT (datetime('now'))
);
```

### Pattern D — Memory-Augmented Agent

**Goal:** Give the insight agent two distinct ways to retrieve what it knows about the user — structured lookup for exact node queries, and semantic search for context that is thematically relevant but may not match on node name. This directly formalises what `user_profile_attributes` already does conceptually. All memory — both structured and semantic — stays on-device.

**Why this matters architecturally:** The current SQLite table supports only structured queries by node name. This misses cases where a user reported "I've been waking up at 3am" (stored under `summary_text`, no node), which is semantically relevant to a sleep-related insight but would never surface in a node-name lookup. Semantic retrieval over the user's own stored observations closes this gap.

**On-device embedding:** User observation text is converted to a dense vector representation using a lightweight sentence embedding model bundled in the app (ONNX format, CPU-only, no network call required). The same model runs at storage time (when the observation is first saved) and at retrieval time (when the agent searches for relevant past context). The embedding model is a design-level choice — it must be small enough to bundle in a mobile app and fast enough to run on-device CPU without perceptible latency.

**On-device vector store:** Embeddings are stored in a vector index that lives on-device as part of the SQLite database. SQLite vector extension support (e.g. `sqlite-vss`) allows similarity queries to run without any additional server or process. The same database file that holds `self_reports` and `user_profile_attributes` also holds the vector index — no new storage infrastructure required.

**`UserMemory` class — design contract:**

The `UserMemory` class wraps both the SQLite structured store and the on-device vector index behind a single interface with four responsibilities:

- `store(observation, node, value, expires_days)` — writes to SQLite `user_profile_attributes` for structured lookup and upserts the embedding into the vector index for semantic retrieval.
- `retrieve_structured(node)` — exact node-name lookup against SQLite, respects expiry. Used when the agent needs a specific known attribute.
- `retrieve_relevant(context_string, top_k)` — embeds the context string, queries the vector index by cosine similarity, returns the top-k most relevant past observations regardless of node name. Used by the ReAct agent before synthesising an insight.
- `deduplicate_before_insert(raw_text)` — before any new self-report is inserted, embeds `raw_text` and queries the vector index. If an existing observation exceeds the similarity threshold, the new observation is treated as a near-duplicate: the existing record's confidence is updated rather than creating a new row. Prevents the same symptom phrased differently multiple times from inflating Layer 2 context.

**Integration with the ReAct agent:** `retrieve_relevant()` is exposed as the `get_user_memory` tool in the agent's tool registry. The agent decides whether to call it based on the current situation — it is not injected unconditionally into every insight.

**Privacy boundary:** The embedding model runs on-device. The vector index lives on-device. No user observation text or embedding vector is sent to any server at any point in this flow.

---

## 5.6. Rolling Inference Window and Dual Trigger Architecture

The DBN is not a batch system. It runs continuously in two modes — passive polling and on-demand chat response — both calling the same inference pipeline.

### Two Independent Triggers — One Shared Pipeline

| Trigger | When it fires | What changes in evidence | Output |
|---------|--------------|--------------------------|--------|
| Timer (Trigger 1) | Every N minutes — configurable via `inference_interval_minutes` | Passive nodes updated from latest sensor window. Self-reports carried forward from SQLite. | Silent snapshot written to `daily_snapshots`. Dashboard updates. |
| Chat (Trigger 2) | Immediately when user sends a message that passes confidence gates | New self-report node written to SQLite. Passive nodes unchanged from last timer snapshot. | Insight returned to user. Snapshot written with `trigger_type = "self_report"`. |

### Concrete Dual Trigger Flow

```
Minute 0:   Slice t created (timer fires)
            Passive nodes filled from latest sensor window
            Self-report nodes from SQLite (all valid, non-expired)
            DBN inference runs → posterior written
            trigger_type = 'timer'

Minute 7:   User sends message (chat trigger fires)
            SLM extracts entity → gates pass → self-report written to SQLite
            build_dbn_evidence() called immediately
            DBN inference runs → new posterior written
            trigger_type = 'self_report'
            Insight returned to user immediately

Minute 15:  Slice t+1 created (timer fires)
            Passive nodes updated from new sensor window
            Temporal edge: slice t posterior (minute-7 version) → slice t+1 prior
            DBN inference runs → new posterior written
            trigger_type = 'timer'
            Dashboard updates silently
```

### Recency Decay — Sub-Daily Granularity

```python
def compute_node_confidence(node, data_source, minutes_since_updated):
    source_weight = {'self_report': 1.00, 'passive': 0.65,
                     'inferred': 0.40, 'prior_only': 0.20}[data_source]
    if data_source == 'passive':
        expiry_minutes = cfg['temporal']['expiry_days'] * 1440
        recency = max(0.0, 1.0 - (minutes_since_updated / expiry_minutes))
        proxy_strength = cfg.get('passive_proxy_strength', 1.0)
        return source_weight * recency * proxy_strength
    else:
        expiry_days = cfg['temporal']['expiry_days']
        recency = max(0.0, 1.0 - (minutes_since_updated / (expiry_days * 1440)))
        return source_weight * recency
```

---

## 5.7. Window-Aware Training Pipeline

The training data pipeline produces one row per `(uid, date, window_start)`. A single configuration constant `WINDOW_MINUTES` controls granularity across the entire pipeline. **The training granularity must match the runtime inference interval.**

### Configuration Constant

```python
# ROLLING WINDOW CONFIG
# Valid values (minutes): 15, 60, 360, 1440
# 1440 = daily — reproduces original row count with one extra window_start column
WINDOW_MINUTES = 1440
```

### Fixed, Midnight-Anchored Windows

Windows are fixed and anchored to midnight of each calendar date. They never shift based on when data arrives. A sensor reading is assigned to a window by checking whether its timestamp falls within `[window_start_unix, window_end_unix)`. This is non-negotiable — fixed windows make inference consistent and comparable across users and days.

### Two New Core Helper Functions

**`clip_intervals_to_window`** — clips a list of `(start, end)` unix-second intervals to a specific time window, deduplicates overlaps, returns total minutes covered. Used by dark sensing and phonelock sensing. The existing `deduplicate_intervals` function is preserved unchanged for nighttime daily aggregates.

**`get_window_slots`** — generates all W-minute window boundaries for a given date as `(window_start_unix, window_end_unix, window_start_dt)` tuples. For 1440 min: 1 slot. For 60 min: 24 slots. For 15 min: 96 slots.

### Output Schema Change

```
uid | date | window_start | <feature_columns>
```

Final merge joins on `['uid', 'date', 'window_start']` for windowed loaders, on `['uid']` only for surveys.

### Sensor-Alive Architecture

**Category A — Validator Sensors** (always-on): `activity/`, `wifi/`, `bluetooth/`, `gps/`, `audio/`

**Category B — Behavioural Sensors** (event-driven; zero = genuine inactivity): dark sensing, phonelock, app usage, call log, SMS

**Window-Alive Decision Rule:**
```
STEP 1: Sensor has reading in this window?
  YES → store value. Done. Real reading always trusted.

STEP 2: No reading. Is window alive?
  alive = (uid, w_start_unix) in all_alive_windows
  YES (alive) → store 0. Genuine inactivity.
  NO (dead)   → store NaN. Excluded from training.

EMA columns: never subject to this rule.
```

### Per-Sensor Details

- **Activity:** Validator. No readings = dead window. `sedentary_ratio` and `active_ratio` both NaN.
- **Dark sensing:** Quality filter `dark_entries >= 3` removed. New col: `dark_window_minutes`. Rule: intervals present → `clip_intervals_to_window`. Alive + no intervals → 0. Dead → NaN.
- **Phonelock:** Quality filters `lock_entries >= 3` and `unlock_span_h >= 6` removed. New col: `unlocked_window_minutes`. Same rule.
- **Screen time (derived):** Night slots (21:00-07:00) → `unlocked_window_minutes` only. Day slots → cross-validate dark + phonelock.
- **App usage:** New col: `avg_running_tasks_window`. Daily aggregates alive-filtered before computation.
- **Call log / SMS:** Readings present → count. Alive + no readings → 0. Dead → NaN.

### EMA Fall-Through Carry-Forward

EMA responses are state declarations not point events. Per uid per day per EMA type: walk window slots in order, carry last response forward. Day boundary is hard — never crosses midnight. Never fill with 0. NaN before first daily response = genuinely unknown state.

### Nighttime Carry-Forward

`nighttime_active_minutes` and `nighttime_unlock_minutes` shifted forward one calendar day, joined as `prev_night_active_minutes` and `prev_night_unlock_minutes` on all window rows of the following date.

### New Output Columns

| Column | Description |
|--------|-------------|
| `window_start` | Fixed W-minute window open boundary (datetime, midnight-anchored) |
| `dark_window_minutes` | Dark sensor minutes within this window |
| `unlocked_window_minutes` | Phone unlocked minutes within this window |
| `screen_time_window_minutes` | Estimated screen-on minutes (derived from dark + phonelock) |
| `avg_running_tasks_window` | Mean running app tasks within this window slot |
| `prev_night_active_minutes` | Phone-active minutes during previous night 20:00-02:00 |
| `prev_night_unlock_minutes` | Phone unlocked minutes during previous night |

---

## 5.8. K-Means Discretisation Pipeline

The DBN requires discrete states. All continuous node columns must be discretised before Structural EM runs. This section documents when, why, and exactly how.

### Position in Pipeline

```
clean → train/val split → winsorise → z-score → min-max normalise →
survey ffill → harmonised columns built → DISCRETISE → Structural EM
```

Thresholds fit on StudentLife training uids only, applied identically to val rows, LifeSnaps rows, and at runtime.

### Why K-Means Over Quantile or Equal-Width Splitting

Quantile split forces equal population per bin regardless of natural structure. Equal-width forces equal range. Both ignore where values actually cluster. K-Means finds natural cluster centres in the data — if students cluster at 5 min, 25 min, and 55 min of screen time per hour, K-Means finds those three centres. Cutting arbitrarily through a dense cluster produces weaker, noisier CPT edges.

### Implementation

```python
from sklearn.cluster import KMeans
import numpy as np
import json

discretisation_thresholds = {}

cols_to_discretise = {
    'screen_time_window_minutes': 'screen_time',
    'activity_unified':           'activity_level',
    'sedentary_unified':          'sedentary_time',
    'sleep_hours_unified':        'sleep_duration',
    'avg_running_tasks_window':   'app_usage',
    'call_count':                 'social_activity',
    'hourly_bpm_z':               'heart_rate',
}

for raw_col, node_name in cols_to_discretise.items():

    # Fit on training rows only — no val or LifeSnaps data
    train_vals = train_df[raw_col].dropna().values.reshape(-1, 1)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(train_vals)

    # Sort centres so labels are ordered low → medium → high
    centres = np.sort(km.cluster_centers_.flatten())

    # Boundaries = midpoints between sorted centres
    boundaries = [
        (centres[0] + centres[1]) / 2,
        (centres[1] + centres[2]) / 2,
    ]

    discretisation_thresholds[node_name] = {
        'boundaries': boundaries,
        'centres':    centres.tolist()
    }

    # Apply to full combined dataset
    def discretise(val, b):
        if pd.isna(val):
            return np.nan
        if val <= b[0]:
            return 'low'
        elif val <= b[1]:
            return 'medium'
        else:
            return 'high'

    combined[node_name] = combined[raw_col].apply(
        lambda v: discretise(v, boundaries)
    )

# Save — applied identically at runtime
with open('feature_node_config.json', 'w') as f:
    json.dump({'discretisation_thresholds': discretisation_thresholds}, f)
```

### Class Balance Check — Mandatory Before Structural EM

K-Means can produce severely imbalanced bins on skewed data. Any state with less than ~10% of rows will have CPT entries estimated from too few observations — unreliable. Check after discretisation:

```python
for node_name in cols_to_discretise.values():
    dist = combined[node_name].value_counts(normalize=True)
    print(f'{node_name}:\n{dist}\n')
    if (dist < 0.10).any():
        print(f'  WARNING: {node_name} has underrepresented state — adjust boundary')
```

Fix by shifting one boundary manually if needed. The goal is not perfect balance but no state so rare that CPT cannot learn it.

### Survey Scale Cols — Not Discretised

PHQ-9, PSS, PANAS, loneliness, VR-12, BigFive items are **never** K-Means discretised. They remain as normalised continuous values and feed directly into Structural EM as evidence for the latent node E-step. They are not runtime nodes — they shape CPT values during training and disappear. Discretising them would destroy the gradient of clinical meaning embedded in their validated scale design.

### Two-Track Discretisation — Which Track Per Column

Not all columns use K-Means. Two tracks apply based on whether established clinical cutoffs exist:

**Track 1 — Clinical/Validated Thresholds (fixed cutoffs, not data-driven):**

| Column | Cutoffs | States |
|--------|---------|--------|
| `phq_total` | 0–4 / 5–9 / 10–14 / 15–19 / 20–27 | none / mild / moderate / mod_severe / severe |
| `pss_total` | 0–13 / 14–26 / 27–40 | low / moderate / high |
| `sleep_hours` | < 6 / 6–9 / > 9 | short / normal / long |
| `BMI` | < 18.5 / 18.5–24.9 / 25–29.9 / ≥ 30 | underweight / normal / overweight / obese |
| `SMQ040` | 1 / 2 / 3 | daily / some_days / not_at_all |
| `vr_general_health` | 1 / 2 / 3 / 4 / 5 | poor / fair / good / very_good / excellent |
| `HSD010` (NHANES general health) | 1–2 / 3 / 4–5 | poor_fair / good / very_good_excellent |
| `diabetes_status` | already discrete 0–3 | none / prediabetes / diabetes / diabetes_insulin |
| `chronic_condition` | already binary | 0 / 1 |
| `EDUCATION` | 1–2 / 3 / 4 / 5 | less_than_HS / HS_grad / some_college / college_grad |

**Track 2 — K-Means (data-driven, see implementation above):** all behavioural and sensor columns — `screen_time_window_minutes`, `active_ratio`, `call_count`, `call_duration_total`, `sms_count`, `hourly_bpm`, `hourly_steps`, `prev_night_resting_hr`, `prev_night_minutesAsleep`, `panas_pa`, `panas_na`, `lonely_total`, `extraversion`, `neuroticism`, `stress_ema_level`, `mood_*`, and all other continuous NHANES activity/physical cols.

> For nodes that appear in both StudentLife and NHANES (e.g., `phq_total`), K-Means cutoffs are derived from the larger dataset (NHANES) and applied identically to StudentLife rows.

---

## 5.9. DBN Training — Phased Structural EM with Expert Knowledge

### Expert Knowledge — Forced and Forbidden Edges

Before EM begins, two edge sets are defined and validated:

**Forced edges** — domain knowledge that HillClimbSearch must never remove or reverse. Causal directions established by medical literature and DBN design intent (e.g., `mental_stress → general_health`, `sleep_duration → mental_stress`). Stored in `bn_structure_lbp.json` under `forced_edges`.

**Forbidden edges** — edges HillClimbSearch must never add. Rule: all dynamic→static edges are forbidden, *except* those explicitly listed as forced edges. This prevents the model from learning spurious reverse-causal paths from latent mediators back to their determinants.

**Startup validation** (runs once before training begins):
1. Forced edges must form no cycle — `nx.is_directed_acyclic_graph(DiGraph(forced_edges))` — raises `ValueError` if cyclic.
2. Forced ∩ Forbidden must be empty — raises `ValueError` if any forced edge also appears in the forbidden set. This catches the common error of the dynamic→static rule accidentally banning a forced edge.

### Phased Structural EM

Training runs three phases of increasing data volume to warm up the structure before committing to full-data HC:

| Phase | Data fraction | HC max steps | Perturbation after? | Purpose |
|-------|--------------|--------------|--------------------|---------|| 1 | 40% | 200 | Yes | Warm-up — find rough initial structure |
| 2 | 70% | 300 | Yes | Intermediate refinement |
| 3 | 100% | 500 | No | Final convergence on full dataset |

Each phase runs E-step (LBP on current structure to fill latent node soft labels) then M-step (HillClimbSearch from sanitized seed). Phases 1 and 2 apply perturbation after convergence to diversify starting point before advancing to more data.

### `_sanitize_structure()` — Pre-HC Guard

Called on both the seed structure and the current structure before every HillClimbSearch invocation:

1. For every forced edge `(p → c)`: if `(c → p)` exists, remove it.
2. For every forced edge `(p → c)`: if `(p → c)` does not exist, add it.

Without this guard, pgmpy HC can remove a required edge for a BIC gain, add its reverse, then every subsequent HC call fails trying to restore the required edge (creating a cycle). `_sanitize_structure()` closes this loop.

### `_perturb_structure()` — Local Optimum Escape (Phases 1 + 2)

After each phase converges, the structure is perturbed with `_PERTURB_N_OPS = 12` random operations to diversify the starting point for the next HC search:

- Candidate operations: add, remove, or reverse any edge
- Excluded candidates: reverses of forced edges + all pairs in `expert_knowledge.forbidden_edges`
- 12 random operations applied from the remaining valid candidate set

The forbidden edge set from `expert_knowledge` is used directly — not just forced-edge reverses — so perturbation cannot install an edge the M-step would immediately have to undo.

### `_hard_kick()` — Plateau Escape (Phases 1 + 2)

When the structure is unchanged across consecutive iterations (plateau), a hard kick removes `_PERTURB_KICK_N = 8` random non-forced edges before the next HC call. Stronger than perturbation; only triggered when the model is genuinely stuck.

The kicked structure passes through `_sanitize_structure()` before HC runs, so forced edges are always restored after the kick.

### Training Loop — Summary Flow

```
for each phase in [1, 2, 3]:
    data_subset = sample(training_data, phase_fraction)
    for each iteration:
        E-step: LBP on current_structure → soft labels for latent nodes
        M-step: seed = _sanitize_structure(current_structure)
                if plateau and phase < 3: seed = _hard_kick(seed)
                new_structure = HillClimbSearch(seed, max_iter=phase_hc_steps)
                current_structure = _sanitize_structure(new_structure)
        if converged: break
    if phase < 3: current_structure = _perturb_structure(current_structure)

save(current_structure, bn_structure_lbp.json)
```

### Output Format — `bn_structure_lbp.json`

```json
{
  "trainable_nodes":   ["node1", "node2", ...],
  "forced_edges":      [["parent", "child"], ...],
  "all_edges":         [["parent", "child"], ...],
  "inter_slice_edges": [["node", "node"], ...]
}
```

`inter_slice_edges` stores temporal self-loops as `[[node, node], ...]` pairs. Read by `visualize_bn.py` to draw orange arc loops on temporal nodes.

---

## 6. High-Weight Node Derivation from CPT

After DBN training, each parent node's influence on a target latent node is quantified using **CPT spread** — the difference between the maximum and minimum posterior probability of the high state across parent states.

```python
def compute_high_weight_nodes(dbn, target_node, top_x=3, threshold=0.30):
    spreads = {}
    cpd = dbn.get_cpds(target_node)
    for parent in dbn.get_parents(target_node):
        p_high_given_parent_high = cpd.get_value(**{parent: 2, target_node: 2})
        p_high_given_parent_low  = cpd.get_value(**{parent: 0, target_node: 2})
        spreads[parent] = abs(p_high_given_parent_high - p_high_given_parent_low)
    ranked = sorted(spreads, key=spreads.get, reverse=True)
    return [n for n in ranked[:top_x] if spreads[n] >= threshold]

# Example output for mental_stress:
# spreads: {mood: 0.61, sleep_duration: 0.55, screen_time: 0.48,
#           alcohol_use: 0.22, social_activity: 0.19, smoking: 0.10}
# top_3: ['mood', 'sleep_duration', 'screen_time']
```

> `top_x` starts at 3. Threshold 0.30 filters negligible parents. Run once after every retraining. Write to `feature_node_config.json` under `high_weight_contributors`.

---

## 7. Data Source Flag and Confidence System

Every node carries a `data_source` field and a computed `confidence` value.

### Data Source Field

| `data_source` value | Meaning | Example |
|--------------------|---------|---------|
| `self_report` | User explicitly stated this in chat | "I have knee pain" → `pain_level=mild` |
| `passive` | Phone sensor computed this | HealthKit: 5.1hrs → `sleep_duration=poor` |
| `inferred` | DBN computed this from other nodes | `mental_stress` inferred from `mood+sleep+screen` |
| `prior_only` | Only NHANES population prior — no user data | `alcohol_use` when user never mentioned alcohol |

### How Confidence Drives Behaviour

| Confidence range | Insight inclusion | Language style | Proactive trigger? |
|-----------------|-------------------|---------------|-------------------|
| 0.70 - 1.00 | Always included | Assertive: *"Your sleep was poor — 5.1hrs from HealthKit"* | No |
| 0.35 - 0.69 | Included with caveat | Hedged: *"Your social contact appears lower than usual"* | No |
| 0.00 - 0.34 | Excluded OR triggers proactive question | Not shown unless question answered | **Yes — if high-weight contributor** |

---

## 8. Onboarding — Radical Simplification

| Step | Collected | Why |
|------|-----------|-----|
| 1 | Age + biological sex | Stratifies NHANES priors |
| 2 | Notification time preference | When to send daily nudge |
| 3 | START | Everything else inferred passively or collected organically |

> Alcohol use, smoking, chronic pain, and personality traits are **NOT** asked at onboarding. Nodes sit at NHANES population priors until user mentions them in chat.

---

## 9. Extended Temporal Flag System

| Flag | Meaning | Expiry | Example utterance |
|------|---------|--------|------------------|
| `today` | Single-day event | 1 day | *I exercised this morning* |
| `week` | Pattern for current week | 7 days | *I've been using my laptop 8 hours a day this week* |
| `daily` | Habitual recurring behaviour | 7 days | *I usually smoke half a pack a day* |

### Persistence Follow-up Question

```python
# System asks: 'Is this how you've been feeling generally, or just today?'
User: 'just today'     → temporal_flag = 'today'  → expires tomorrow
User: 'this week'      → temporal_flag = 'week'   → expires in 7 days
User: 'always/usually' → temporal_flag = 'daily'  → treated as habitual
User skips:            → temporal_flag = 'today'  → safe default
```

### Backdated Self-Reports — `report_date` Field

When a user says *"I drank a lot last Wednesday"*, the SLM extracts the node and value but needs a mechanism to assign it to the correct past date.

```json
{
  "node":          "<node_name>",
  "value":         "<state>",
  "temporal_flag": "<today|week|daily>",
  "report_date":   "<YYYY-MM-DD or null>",
  "raw_minutes":   null,
  "confidence":    0.0,
  "raw_text":      "<exact phrase>",
  "reasoning":     "<one sentence>"
}
```

**Rules:** Today/habitual → `report_date = null`. Specific past date → resolved ISO date. Cannot resolve → null, confidence lowered. Maximum lookback: 30 days.

**Retroactive recomputation:** When backdated report written, all `daily_snapshots` from `report_date` to today are stale. Recomputation pass replays inference forward from that date. **Flagged as post-MVP** — requires complete runtime pipeline first.

---

## 10. Evidence Fusion — Self-Report vs Passive Priority

```python
def build_dbn_evidence(
    date_str:         str,
    passive_features: dict,
    window_start:     datetime,
    window_end:       datetime,
    trigger:          str         # 'timer' | 'self_report' | 'manual'
) -> dict:
    # Step 1: discretise passive features as baseline
    # Step 2: apply valid self-reports from SQLite (non-expired)
    # Step 3: aggregate sub-dimension attributes from user_profile_attributes
    # Step 4: assign confidence — recency uses minutes_since_updated, not days
    ...
```

### `raw_text` Forwarding to Insight Generator

`raw_text` forwarded from NER JSON through `chat_trigger.py`. Used for linguistic specificity only — never for DBN reasoning. The architectural boundary is absolute: `raw_text` never influences node selection, belief updates, or evidence fusion. Retained permanently even after symptom-specific Layer 2 nodes are added.

`raw_text` is included in the context provided to the Gemma 4-2B-IT ReAct agent when it begins reasoning. The agent uses `raw_text` to form more specific `get_user_memory` queries and to produce more linguistically precise insight language.

**Before:** *"This pain seems to be putting pressure on your physical stress levels."*
**After:** *"Eye pain and redness like this is often linked to high screen time and poor sleep — both of which have been elevated for you recently."*

---

## 11. Contextual Awareness — Chat vs Dashboard

| Surface | Purpose | What it shows |
|---------|---------|---------------|
| Dashboard | Overall state | Physical stress %, mental stress %, 7-day graphs, trends, alert history |
| Chat | Causal explanation | What this specific report caused + what the DBN sees as causing it |

---

## 12. Delta Inference — Contextual Insight Engine

`compute_delta()` runs **on-device** immediately after a self-report passes the confidence gates. It produces the `delta_snapshot` stored in `last_entity` in session state. This delta is passed directly to the Gemma 4-2B-IT ReAct agent as part of its initial context — specifically to decide whether to call `get_trend()` (if delta is large, a trend check adds temporal framing) and to frame the effects direction of the two-direction insight.

```python
def compute_delta(dbn, evidence_without, evidence_with, all_nodes):
    beliefs_before = dbn.query(variables=all_nodes, evidence=evidence_without)
    beliefs_after  = dbn.query(variables=all_nodes, evidence=evidence_with)
    delta = {}
    for node in all_nodes:
        if node in ['physical_stress', 'mental_stress']:
            delta[node] = abs(
                beliefs_after[node].values[2] -
                beliefs_before[node].values[2]
            )
    return dict(sorted(delta.items(), key=lambda x: x[1], reverse=True))
```

---

## 13. Two-Direction Insight Generation

| Direction | Source | Example |
|-----------|--------|---------|
| Effects (downstream) | Delta inference | *"Your heavy drinking shifted mental stress from 32% to 72%"* |
| Causes (upstream) | Top parent nodes by CPT contribution | *"Main contributors: low energy (5.1hrs sleep) and mild pain"* |
| Layer 2 context | Semantically relevant `summary_text` — retrieved by vector similarity, no time cutoff (v7.5) | *"You also mentioned feeling overwhelmed with deadlines"* |

---

## 14. Pre-Insight Proactive Question System

```python
def get_single_proactive_question(intent, node, evidence):
    target = node or infer_target_from_intent(intent)
    parents_by_spread = get_parents_ranked_by_cpt_spread(target)
    for parent in parents_by_spread:
        conf = evidence.get(parent, {}).get('confidence', 0.0)
        if (conf < CONFIDENCE_THRESHOLD
            and no_passive_proxy(parent)
            and not_asked_in_7_days(parent)
            and questions_asked_today() == 0):
            return load_node_config(parent)['proactive_question']
    return None
```

---

## 14b. Question Cascade — Post-NER Guided Interview

Triggered immediately after NER completes for any `symptom_report`, `status_change`, or `lifestyle_disclosure` intent that fills at least one original_column.

### Full Flow

```
User text
   ↓
NER (Llama 1B)
  → fills N original_cols (direct matches, confidence any)
  → fills correlated/inverse cols (SLM-inferred, only if internal confidence ≥ 0.7)
  → all written to DB immediately
   ↓
questionCascade.build(filledCols) → ordered question queue
   ↓
Ask one question at a time (chat UI renders each as a message bubble)
  → User answers  → write to DB, advance queue
  → User skips    → mark skipped, advance queue
  → User taps Stop button → flush queue, proceed to insight
   ↓
Queue empty OR stopped → trigger insight generation
```

### Correlated / Inverse Column Inference (NER step)

SLM fills correlated and inverse columns in the same NER pass when it is sufficiently confident. Rules baked into system prompt:

- If a column has a semantic inverse (e.g. `e_talkative` ↔ `e_reserved_r`, `e_quiet_r`), and user statement implies a direction, fill the inverse with the opposite polarity.
- If a column is part of a group where the user's statement implies a related state (e.g. "I'm very sociable" → also implies `e_quiet_r=low`), fill it.
- Only fill correlated/inverse cols when SLM internal confidence ≥ 0.7 for that inferred value.
- SLM outputs all inferred cols in the same JSON array as direct matches, with a `inferred: true` flag and confidence value.
- User cannot be asked to confirm an inverse of what they just said (poor UX) — these are written silently.

### Question Queue Construction (`questionCascade.build`)

Given the set of filled `original_col` values, build queue in priority order:

| Tier | What | Rule |
|------|------|------|
| 1 | Other original_cols sharing the same `source_col` (same composite) | Fill the composite more completely |
| 2 | Other source_cols in the same node | Same node, different composite or single col |
| 3 | Proactive: high-weight unfilled parent nodes with no passive proxy | Same as Section 14 proactive logic |

Within each tier, order by CPT spread weight (highest first). Skip any col already filled in this session or answered within 7 days.

### Queue State (managed by `questionCascade.ts`)

```typescript
interface CascadeQuestion {
  original_col: string;
  source_col:   string;
  node_name:    string;
  question:     string;   // from column_question_map.json
  tier:         1 | 2 | 3;
}

interface CascadeState {
  questions: CascadeQuestion[];
  current:   number;         // index into questions[]
  stopped:   boolean;
}
```

API:
- `build(filledCols, nodeConfig, cqMap) → CascadeState`
- `currentQuestion(state) → CascadeQuestion | null`
- `advance(state) → CascadeState`           // skip or after answer
- `stop(state) → CascadeState`              // user taps Stop
- `isDone(state) → boolean`                 // stopped or current === questions.length

### DB Writes During Cascade

Each user answer goes through `runNer()` with the specific question context injected into the prompt so the SLM can parse the answer against the correct options list. Confidence for cascade answers = 1.0 (user directly answered the question asked). Inferred cols written at SLM confidence.

### Insight Trigger

`isDone(state) === true` → `runInfer()` called → insight displayed. No insight fires before cascade completes or stops.

### Suppression

Cascade questions respect 7-day per-col suppression (same as proactive question system). A col answered in this cascade is suppressed for 7 days in future cascades.

---

## 15. Insight Button Logic

| Intent class | Response type | Insight button? |
|-------------|--------------|----------------|
| `symptom_report` | Causal — entity extracted, delta computed | Yes |
| `status_change` | Causal — direction extracted, delta computed | Yes |
| `stress_query` | Insight generated immediately | N/A — inline |
| `lifestyle_disclosure` | Acknowledgement only | No |
| `off_topic` | Redirect | No |

---

## 16. Daily Notification Design

One notification per day at user-set time (default 8pm). Question priority: (1) highest-weight unfilled node with no passive proxy, (2) node whose passive proxy has lowest confidence, (3) node not answered longest, (4) never repeat within 7 days.

Optional follow-up: when user answers and taps "Share more", pre-selected related questions appear. Each answer fills a separate node with its own 7-day suppression.

---

## 17. Turn Routing — Stage Pipeline

The Talk turn is processed through an 8-stage pipeline. Each stage has a precise trigger condition; only the stages whose conditions are met run. Stages short-circuit the pipeline when they produce a terminal response.

### Full Stage Flow

```
STAGE 0: isAckOnly
  — Exact wordlist match: ["ok","okay","k","got it","thanks","thank you",
                            "cool","alright","sure","noted","👍","👌"]
  — Exact match only (no partial/substring). Canned warm response. No model calls. DONE.

STAGE 1: detectUndoIntent (Llama, n_predict=3, temp=0.0) → isUndoTurn (bool)

STAGE 2: classifyIntent (Llama, n_predict=55, temp=0.0) → IntentResult — ALWAYS runs
  IntentResult: { hasReport, hasQuery, hasTrendQuery, isSocial, isThirdPartyQuery, queryNodes[] }
  socialFastPath = isSocial && !hasQuery && !hasReport
  Trend fetch: hasTrendQuery && queryNodes non-empty && !socialFastPath
    → get_belief_trend { node_names: queryNodes, window_days: 14, session_id }
    → trendSummary (injected as TREND HISTORY block in context)

STAGE 3: runNer — ONLY IF hasReport=true OR isUndoTurn=true
  Two-pass: segmentText → per-segment NER → deduplicateEntities
  → entities[], unmatched[], topics[]

STAGE 3b: runUndoWork(db, sessionId, turnId, nerResult) — ONLY IF isUndoTurn=true
  — soft-deletes previous turn rows (entity-level if targets resolved, full turn otherwise)
  — captures preUndoBeliefs from second-most-recent inference_snapshot
  — resolves correctedNodes vs newNodes
  isPureUndo = preUndoBeliefs !== null && unmatched.length === 0 && newNodes.length === 0
    → if isPureUndo: undoAck(), DONE

STAGE 4: run_dbn_inference — SKIP IF socialFastPath
  — throttled to 2 runs per turn
  — skips if no new evidence since last run
  → beliefs: BeliefResult | null

STAGE 5: selectBeliefWindow(beliefs, intentResult, isEarlyInteraction)
  isEarlyInteraction (getReportedTurnCount < 3):
    gate=0.0, budget=∞, all nodes, each annotated "[prior — no user data yet]"
  hasQuery (non-social):
    queryNodes forced in regardless of confidence, gate=0.35, budget=15
  report-only (hasReport && !hasQuery):
    gate=0.55, budget=8
  socialFastPath:
    → returns empty string (no belief block in context)

STAGE 6: buildContextBlock — stacks ALL applicable injections:
  isUndoTurn && preUndoBeliefs non-null  → CORRECTION TURN
  isEarlyInteraction                     → FIRST INTERACTION
  hasQuery                               → QUERY TURN
  isSocial && hasQuery                   → MIXED SOCIAL+QUERY
  isSocial && !hasQuery                  → SOCIAL TURN
  isThirdPartyQuery                      → THIRD-PARTY QUERY
  unresolved queryNodes (not in beliefs) → NOTE: not tracked

STAGE 7: Gemma
  socialFastPath  → glance forced, n_predict=80
  glance mode     → single runGlanceCall, n_predict=400
  reflect mode    → runReactLoop, up to 8 ReAct steps
```

### Routing Decision Matrix

| isUndo | hasReport | hasQuery | isSocial | isThirdParty | Path |
|--------|-----------|----------|----------|--------------|------|
| false | false | false | true | false | socialFastPath → Gemma n=80 |
| false | false | true | false | false | Stage 4 inference → queryNodes forced → Glance/Reflect |
| false | false | true | true | false | Stage 4 inference → MIXED SOCIAL+QUERY context → Glance/Reflect |
| false | true | false | false | false | Stage 3 NER → Stage 4 → report-only belief window → Glance/Reflect |
| false | true | true | false | false | Stage 3 NER → Stage 4 → queryNodes forced + report beliefs → Glance/Reflect |
| true | false | false | false | false | Stage 3 NER → Stage 3b → isPureUndo check → undoAck or Glance/Reflect |
| true | true | false | false | false | Stage 3 NER → Stage 3b → resolve correctedNodes/newNodes → Glance/Reflect |
| false | false | false | false | true | Stage 4 → THIRD-PARTY QUERY context → Glance/Reflect |
| false | true | false | false | true | Stage 3 NER → Stage 4 → THIRD-PARTY QUERY → Glance/Reflect |
| false | true | true | false | true | Stage 3 NER → Stage 4 → THIRD-PARTY QUERY + QUERY → Glance/Reflect |

---

## 18. Hybrid Node Structure — When to Separate vs Merge

Merge two training columns into one node if: same psychological construct, removing one loses no information, correlated r > 0.7.

| Example | Decision | Reason |
|---------|----------|--------|
| `lonely_leftout + lonely_isolated` | Merge → `social_exclusion` | Same construct |
| `lonely_companionship + lonely_group` | Separate | Different constructs |
| PSS items 1-10 | Separate training cols → one latent node | Distinct stress dimensions |
| PHQ-9 items 1-9 | Separate training cols → one latent node | Multi-dimensional: sleep ≠ concentration |
| `n_worries + n_nervous` (BigFive) | Separate → contribute to `mental_stress` CPT | Worry ≠ anxiety reactivity |

---

## 19. Confidence Gates — Three Gates + Confidence Check

```python
CONFIDENCE_THRESHOLD = 0.70
NODE_CONFIDENCE_THRESHOLD = 0.35

VALID_NODES = {
    'pain_level','mood','physical_exercise','energy_level',
    'activity_level','sleep_duration','sleep_quality',
    'screen_time','social_activity','sedentary_time',
    'alcohol_use','smoking','heart_rate'
}

def process_extraction(slm_output: dict) -> dict:
    if confidence < CONFIDENCE_THRESHOLD:
        return {'action': 'acknowledge_only', 'reason': 'low_confidence', 'clarify': True}
    if node not in VALID_NODES:
        return {'action': 'acknowledge_only', 'reason': 'unknown_node', 'clarify': False}
    if value not in VALID_VALUES.get(node, []):
        return {'action': 'acknowledge_only', 'reason': 'invalid_state', 'clarify': False}
    if temporal_flag not in valid_flags:
        temporal_flag = 'today'
    return {'action': 'update_dbn', 'node': node, 'value': value, ...}
```

---

## 20. Hybrid Two-Layer Insight Generation

**Layer 1:** DBN-grounded — specific beliefs or delta values, confidence-appropriate language.
**Layer 2:** Unstructured context from past user statements. Omitted entirely if no clearly relevant unmatched context exists. Retrieved via semantic similarity search (v7.5) rather than 7-day time window.

**Example:** *"Your heavy drinking today has shifted your mental stress estimate from 32% to 72% — a 40-point increase. Sleep quality posterior also worsened. You also mentioned feeling overwhelmed with deadlines earlier this week."*

Unmatched raw text summarised **at storage time**, not at insight generation time. The summary is embedded and stored in the on-device vector index. At insight generation time the ReAct agent calls `get_user_memory()` to retrieve semantically relevant summaries — no sequential SLM calls at query time.

> **v7.5 upgrade:** Layer 2 context is now retrieved by semantic similarity over all stored `summary_text` embeddings, not by 7-day cutoff. A statement from 15 days ago that is strongly relevant to today's report will surface; a vague statement from 3 days ago that is unrelated will not. See Section 20b, Pattern A for how the ReAct agent calls this retrieval as a tool.

---

## 20b. Agentic Architecture — Four Patterns

All four patterns run on-device. Llama 3.2-1B-Instruct (Q4_K_M) handles two-pass NER (segmentation + per-segment extraction), intent classification (`classifyIntent`), undo detection (`detectUndoIntent`), and storage-time summarisation — tasks that are well-scoped and single-output. Gemma 4-2B-IT (Q4_K_M) handles the dual-mode insight agent (Glance and Reflect), reflection loop, and doctor brief planning — tasks that require multi-step reasoning or longer context. FastAPI hosts only the DBN inference endpoint, because pgmpy is Python-only and cannot run in React Native.

```
Phone (React Native)
  ├── Llama 3.2-1B-Instruct (Q4_K_M):  two-pass NER (seg → per-segment NER),
  │                                     classifyIntent (Stage 2), detectUndoIntent (Stage 1)
  ├── Gemma 4-2B-IT (Q4_K_M):          dual-mode insight (Glance | Reflect), reflection loop,
  │                                     plan-and-execute (brief)
  ├── TS Orchestrator (agent.ts):       8-stage turn pipeline driver, ReAct loop driver,
  │                                     MCP dispatcher, context builder
  ├── Tool Dispatcher (mcp.ts):         dispatchTool() — in-process TypeScript switch;
  │                                     same tool-name/args interface as MCP JSON-RPC
  ├── chrono-node (TypeScript):         raw_date_text phrase → ISO date resolution
  ├── Embedding model (ONNX, on-device): text → vector for UserMemory
  ├── SQLite:       structured store — self_reports, snapshots, sensor_windows
  └── Vector index (on-device): semantic memory — UserMemory.retrieve_relevant()

FastAPI Server (DBN host only):
  └── /query_dbn  → pgmpy inference, returns posteriors to phone
      (No user health data stored or processed server-side)
```

> **Note on MCP transport:** The current implementation uses direct in-process TypeScript function dispatch (`dispatchTool()` in mcp.ts), not JSON-RPC over stdio MCP servers. The tool name + arguments object interface is identical to the target MCP design. The stdio-server architecture (described below) is the long-term target and can be adopted without changing tool names or argument schemas.

---

### MCP Tool Layer — Current Implementation

All agent tools are defined in `mcp.ts` as `McpTool` entries and dispatched via `dispatchTool(name, args, db, turnId)`. The tool interface (name + `inputSchema`) matches the MCP JSON-RPC contract; the transport is currently in-process TypeScript dispatch rather than stdio.

**Complete MCP tool registry:**

| Tool | Callable from | Input | Output |
|------|--------------|-------|--------|
| `run_dbn_inference` | Stage 4 (agent.ts), Reflect ReAct | `turn_id` | `{ beliefs, skipped }` |
| `get_changed_nodes` | Stage 7 inline, Phase 2, Reflect ReAct | (none) | `{ changed_nodes[] }` |
| `undo_last_entry` | Stage 3b (runUndoWork) | `turn_id`, `node_names?` | `{ undone, pre_undo_beliefs }` |
| `store_indirect_evidence` | Reflect ReAct (max 1/turn) | node, value, confidence, summary | `{ stored }` |
| `get_user_memory` | Stage 1 setup (agent.ts), Reflect ReAct | `window_days` | `{ summaries[] }` |
| `get_cascade_questions` | Legacy (direct call in agent.ts) | `turn_id` | `{ questions[] }` |
| `get_belief_trend` | Stage 2 (hasTrendQuery, agent.ts), Reflect ReAct | `node_names[]`, `window_days`, `session_id` | `{ trends: Record<string,string> }` |

**`get_belief_trend` detail:**
- Input: `{ node_names: string[], window_days: number = 14, session_id: string }`
- Output: `{ trends: Record<string, string> }` — one summary string per node
- Format: `"stress: avg moderate, started high, currently low"`
- Stats: mode of `raw_text` (most frequent label) = avg; first and last `raw_text` in window; directional comparison (above/below avg) from numeric `raw_value` when available
- Skip node if < 3 entries in window (returns empty string for that node)
- In Glance: fetched by `startTurn` when `hasTrendQuery=true && queryNodes.length > 0 && !socialFastPath`; result injected as TREND HISTORY block
- In Reflect: Gemma calls autonomously via ReAct (in `GEMMA_TOOLS`)

**ReAct loop with tool dispatch:**

```
agent.ts runReactLoop
    │
    ├── build prompt: gemmaPrompt(systemContent, contextBlock)
    │   systemContent = PERSONA + RESPONSE_GUIDE + REACT_FORMAT + buildToolsBlock()
    │
    └── loop (up to MAX_STEPS = 8):
          │
          ├── _ctx.completion({ prompt, n_predict: 512, temperature: 0.3 }) → text
          │
          ├── scan for RESPONSE: [text]  → return final response, exit
          │
          ├── scan for TOOL_CALL: {"name":"...","arguments":{...}}
          │       if found:
          │         ├── validate name ∈ GEMMA_TOOLS
          │         ├── apply per-tool constraints (store_indirect_evidence cap=1)
          │         ├── dispatchTool(name, arguments, db, turnId)
          │         └── append OBSERVATION: {result} via new user turn
          │
          └── if no RESPONSE: and no TOOL_CALL: → extract text as fallback response
```

**GEMMA_TOOLS set (tools Gemma may call in Reflect mode):**
`run_dbn_inference`, `get_changed_nodes`, `store_indirect_evidence`, `get_user_memory`, `get_belief_trend`

**When tools are NOT called:**
- Glance mode: context pre-injected, single completion call, no tool dispatch
- Stage 0 (ack-only): no model calls at all
- socialFastPath: Gemma glance at n_predict=80, no tool calls

---

### Pattern A — ReAct Agent for Insight Generation

**Goal:** Replace the fixed Template 2 prompt with an on-device Gemma 4-2B-IT agent that produces grounded, context-aware insights. Two modes — Glance and Reflect — cover the full latency-vs-depth trade-off.

**Pre-injection of context (both modes):** Before any Gemma call, the system injects current DBN beliefs + memory summaries + changed-nodes delta into the system prompt context block. Glance mode never needs a `run_dbn_inference` tool call — the beliefs are already in context.

---

#### Glance Mode *(formerly "Rapid")*

**When to use:** Default. Fast feedback after a user report. Single structured Gemma call, no ReAct loop.

**How it works:**
1. System prompt: PERSONA + RESPONSE_GUIDE (see agent.ts).
2. Context block: built by `buildContextBlock` — stacks applicable injections (CORRECTION TURN, FIRST INTERACTION, QUERY TURN, etc.), then RECENT CONTEXT, MEMORY, CURRENT HEALTH STATE (belief window from `selectBeliefWindow`), optional TREND HISTORY, WHAT CHANGED THIS TURN, USER MESSAGE.
3. `selectBeliefWindow` gates: `isEarlyInteraction` → all nodes at gate=0.0; `hasQuery` → gate=0.35, budget=15, queryNodes forced in; report-only → gate=0.55, budget=8; socialFastPath → empty. Replaces the old fixed P > 0.55 threshold.
4. Single completion call — Gemma writes the response directly. `n_predict=400`, `temperature=0.5`. socialFastPath forces `n_predict=80`.
5. Output returned to UI directly (no streaming in current implementation — streaming can be added later).
6. Latency: ~5 s end-to-end on-device.

**Prompt format (Gemma instruct):**
```
<bos><start_of_turn>system
{PERSONA}\n\n{RESPONSE_GUIDE}<end_of_turn>
<start_of_turn>user
{context_block}<end_of_turn>
<start_of_turn>model
```

---

#### Reflect Mode *(formerly "Deep-think")*

**When to use:** User explicitly selects Reflect via the mode toggle in TalkScreen header.

**How it works:**
1. Same prompt format as Glance, with `REACT_FORMAT + tool descriptions` appended to the system prompt.
2. ReAct loop: reason → act → observe, up to `MAX_STEPS = 8` iterations.
3. Gemma signals tool use via plain-text format: `TOOL_CALL: {"name": "...", "arguments": {...}}`. No XML tags.
4. TS Orchestrator parses `TOOL_CALL:` lines, dispatches to `dispatchTool()`, appends `OBSERVATION: {result}` via a new user turn.
5. Loop exits when Gemma emits `RESPONSE: [text]`. Fallback: extract last model output if `MAX_STEPS` reached.
6. `n_predict=512`, `temperature=0.3` per step.

**ReAct turn format:**
```
THOUGHT: [reasoning]
TOOL_CALL: {"name": "tool_name", "arguments": {"arg": value}}
  → system appends:
<start_of_turn>user
OBSERVATION: {result}<end_of_turn>
<start_of_turn>model
THOUGHT: [continues]
RESPONSE: [final reply]
```

**Tool Registry (Reflect mode only — `GEMMA_TOOLS` set in agent.ts):**

Tool dispatch is handled directly by `dispatchTool()` in mcp.ts — an in-process TypeScript switch. This is functionally equivalent to the MCP JSON-RPC design but without the stdio transport layer in the current implementation.

| Tool | Input | Output | Constraint |
|------|-------|--------|-----------|
| `run_dbn_inference` | `turn_id` | `{ beliefs, skipped }` | Only call after storing new evidence; inference count cap applies |
| `get_changed_nodes` | (none) | `{ changed_nodes[] }` | Only call after new evidence stored |
| `store_indirect_evidence` | node, value, confidence, summary | `{ stored }` | Max 1 call per turn |
| `get_user_memory` | window_days | `{ summaries[] }` | Anytime |
| `get_belief_trend` | node_names[], window_days, session_id | `{ trends: Record<string,string> }` | Anytime; use when user asks about trends/history |

> `run_dbn_inference` count quota applies across both phases. Phase 1 fires count 1. Reflect mode Phase 2 skips a direct inference call, leaving count 1→2 available for the ReAct loop after `store_indirect_evidence` runs. Glance mode Phase 2 fires count 1→2 directly before the Gemma call.

**Note on `<|think|>` chain-of-thought:** The current implementation does not use the `<|think|>` system prompt token or capture `<|channel>thought` blocks. Reasoning is instead expressed through `THOUGHT:` lines in the ReAct format, which the TS Orchestrator parses directly. The `<|think|>` / capture-block approach remains the target for a future refactor once llama.cpp post-April 2026 build is confirmed on all target devices.

---

#### Phase-Split Architecture (Talk Feature)

The Talk feature exposes the two-phase turn split to the UI. This split exists so the UI can show proactive follow-up and cascade questions to the user — collecting answers that enrich inference — before Gemma runs.

**Phase 1 — `startTurn(db, sessionId, userMessage, mode)`:**

```
[STAGE 0]  isAckOnly(userMessage) — exact wordlist match
           → storeChatMessage(user) + storeChatMessage(model, ackResponse())
           → maybeEvictOldest → return { done: true, response }

           setCurrentUserMessage(userMessage)
           getRecentPairs(db, sessionId) → recentPairs
           setRecentTopics(...)  ← must run before get_user_memory
           get_user_memory { window_days } → memorySummaries
           isEarlyInteraction = getReportedTurnCount(db, sessionId) < 3

[STAGE 1]  detectUndoIntent(userMessage) → isUndoTurn (bool)

[STAGE 2]  classifyIntent(userMessage) → intentResult: IntentResult
           socialFastPath = intentResult.isSocial && !hasQuery && !hasReport
           if hasTrendQuery && queryNodes.length > 0 && !socialFastPath:
             get_belief_trend { node_names: queryNodes, window_days: 14, session_id }
             → trendSummary

[STAGE 3]  if hasReport || isUndoTurn:
             runNer(db, userMessage, '', turnId, mode, sessionId) → nerResult
           else: nerResult = emptyNer
           topic = nerResult.topics.join(', ')

[STAGE 3b] if isUndoTurn:
             runUndoWork(db, sessionId, turnId, nerResult)
             → { preUndoBeliefs, isPureUndo }
             if isPureUndo:
               → storeChatMessage(user, topic) + storeChatMessage(model, undoAck())
               → maybeEvictOldest → return { done: true, response: ack }

[STAGE 4]  if !socialFastPath:
             run_dbn_inference { turn_id } (count → 1) → beliefs
           else: beliefs = null

[STAGE 5]  selectBeliefWindow(beliefs, intentResult, isEarlyInteraction)
           ← computed inside buildContextBlock

[STAGE 6]  build question lists (only if !socialFastPath):
           buildFollowUps(nerResult.entities) → followUps
           buildCascade(filledCols, db)       → cascadeQs
           questions: DisplayQuestion[] = [...followUps, ...cascadeQs]

[STAGE 7]  if questions.length === 0:
             get_changed_nodes → changedNodes
             if socialFastPath:
               Gemma glance, n_predict=80
             else if mode=glance:
               runGlanceCall(...)
             else:
               runReactLoop(...)
             → storeChatMessage(user, topic) + storeChatMessage(model, response)
             → maybeEvictOldest → return { done: true, response }
           else:
             → _pendingTurns.set(turnId, { sessionId, userMessage, isUndoTurn: isUndoContext,
                                            beliefs, memorySummaries, recentPairs, topic,
                                            isEarlyInteraction, intentResult, trendSummary })
             → return { done: false, turnId, questions }
```

**Phase 2 — `completeTurn(db, turnId, mode, answers)`:**

```
1. pending = _pendingTurns.get(turnId); _pendingTurns.delete(turnId)
2. for each answer: writeProactiveAnswer(db, turnId, ans)
3. if answers.length > 0: markEvidenceWritten()
4. get_changed_nodes → changedNodes
5. Glance: run_dbn_inference (count → 2) → updated beliefs
           → runGlanceCall(recentPairs, memorySummaries, beliefs, changedNodes, userMessage,
                           isUndoTurn, isEarlyInteraction, intentResult, trendSummary)
   Reflect: skip direct inference so ReAct loop gets its full quota (1→2)
           → runReactLoop(db, turnId, recentPairs, memorySummaries, phase1Beliefs, ...)
6. storeChatMessage(user, topic) + storeChatMessage(model, response)
7. maybeEvictOldest(db, sessionId)
8. return response
```

**`_pendingTurns` Map:** `Map<string, PendingTurn>` keyed by `turnId`. Entry deleted immediately on `completeTurn` call. In-process only — not persisted across app restarts. `intentResult` and `trendSummary` are forwarded to Phase 2 so `selectBeliefWindow` and `buildContextBlock` behave identically whether the turn completed inline or via Phase 2.

---

#### Journal Feature (`runJournalTurn`)

Journal skips Gemma entirely. The user writes privately; the system updates its internal model silently.

```
1. setCurrentUserMessage(userMessage)
2. runNer(db, userMessage, '', turnId, 'reflect', sessionId)  ← reflect-mode gates
3. topic = nerResult.topics.join(', ')
4. run_dbn_inference(turn_id) → beliefs (silent)
5. ack = random acknowledgement string (4 options, no Gemma)
6. storeChatMessage(db, sessionId, turnId, 'user',  userMessage, topic)
   storeChatMessage(db, sessionId, turnId, 'model', ack)
7. maybeEvictOldest(db, sessionId)
8. return ack    ← UI displays this one-liner; user knows entry was received
```

Journal entries are stored in `chat_messages` with both a `user` and `model` row (the model row holds the short ack). Both are retrievable by `get_user_memory` in future turns — the semantic similarity search will surface journal content when relevant to a later conversation. Gemma is never called; the ack is drawn from a hardcoded pool of four short phrases. Topics are derived and stored exactly as in the Talk flow.

**Process — Reflect ReAct Loop:**

The agent runs a THOUGHT → TOOL_CALL → OBSERVATION cycle up to `MAX_STEPS = 8`. Each tool call is conditional — the agent only calls tools it judges relevant to the current report.

**Example Agent Trace (Reflect mode):**

```
User: "I've been having knee pain"

THOUGHT: User mentions knee pain. Current beliefs already pre-injected.
         Check if any past observations about pain stored.
TOOL_CALL: {"name": "get_user_memory", "arguments": {"window_days": 90}}
OBSERVATION: {"summaries": ["knee hurts when walking (12 days ago)", "pain after sitting long (8 days ago)"]}

THOUGHT: Recurring pattern confirmed from memory. I have enough context to respond.
RESPONSE: It sounds like this knee discomfort might be building over time...
```

**Output:** A single response string returned to `completeTurn`, then stored and returned to the UI.

---

### Pattern B — Reflection Loop on Insight Quality

**Goal:** Before the insight string reaches the user, a second LLM call critiques it against the DBN snapshot. Catches the most common failure: asserting certainty about a node whose confidence is below threshold.

**Why a separate critique pass:** The ReAct agent optimises for a rich, grounded insight. It may still over-claim — e.g., asserting that social isolation is a cause when `social_activity` confidence is 0.28. A dedicated critique pass enforces the confidence-language contract mechanically, rather than relying on the agent to self-regulate.

**Process:**

```python
async def generate_insight_with_reflection(
    draft: str,
    dbn_snapshot: dict,
    entity: dict
) -> str:

    critique_prompt = f"""
    Review this health insight against the DBN snapshot provided.
    Flag any of the following issues:
    1. Any claim about a node whose confidence in the snapshot is below 0.35
       — these nodes must not be referenced assertively.
    2. Any claim not derivable from the delta or posterior values in the snapshot.
    3. Missing the mandatory disclaimer line.
    4. Assertive language ('your stress IS high') where confidence is 0.35–0.69
       — must use hedged language ('appears elevated', 'suggests').

    DBN snapshot: {json.dumps(dbn_snapshot)}
    Draft insight: {draft}

    Return JSON only:
    {{
      "issues_found": true | false,
      "issues": ["<issue 1>", ...],
      "revised": "<corrected insight string or original if no issues>"
    }}
    """

    critique = await llm.agenerate(critique_prompt)
    result = json.loads(critique)
    return result['revised']
```

**Integration point:** Called immediately after the ReAct agent produces a draft, before the insight is displayed to the user. One additional model inference pass on-device.

**What this enforces in practice:**
- `social_activity` at confidence 0.28 → removed from assertive claim or reframed as "passive data is limited here"
- Disclaimer missing → appended automatically
- "Your sleep IS poor" when sleep confidence = 0.41 → rewritten to "sleep appears to have been disrupted"

---

### Pattern C — Plan-and-Execute Agent for Doctor Brief

> *See also: Section 26 — Doctor PDF Report for output format and mandatory data limitations section.*

**Goal:** Replace the single Template 5 LLM call with a structured two-stage agent: a planner that writes an execution plan tailored to the user's complaint, followed by an executor that runs each step independently and passes outputs forward.

**Why plan-and-execute over ReAct here:** The doctor brief is a deterministic, structured task — it always needs a symptom timeline, a passive data summary, and a data limitations section. ReAct is better for uncertain, interactive tasks where the next step depends on what the previous tool returned. Plan-and-execute is better when the task structure is known in advance but the content of each section depends on the user's specific data.

**Process:**

```python
async def generate_doctor_brief(complaint: str, user_data: dict) -> str:

    # Stage 1 — Planner: write a structured execution plan
    plan_prompt = f"""
    You are planning a clinical pre-consultation brief for a doctor.
    Patient complaint: {complaint}
    Available data: 30-day DBN snapshots, all self-reports, alert history,
                    passive sensor summaries, node confidence values.

    Write a 5-step execution plan. Each step must specify:
    - step_id: integer
    - data_needed: which fields from user_data to use
    - section_title: the section this step produces
    - instruction: what to write in that section

    Return JSON list of steps only.
    """
    plan = json.loads(await llm.agenerate(plan_prompt))

    # Stage 2 — Executor: run each step, passing results forward
    results = {}
    for step in plan:
        step_prompt = f"""
        Task: {step['instruction']}
        Data: {json.dumps({k: user_data[k] for k in step['data_needed'] if k in user_data})}
        Previous sections for context: {json.dumps(results)}
        Write this section only. Plain clinical language. Under 150 words.
        """
        results[step['section_title']] = await llm.agenerate(step_prompt)

    # Stage 3 — Synthesiser: assemble final brief
    synthesis_prompt = f"""
    Assemble these sections into a single clinical pre-consultation brief.
    Sections: {json.dumps(results)}
    Always append the data limitations section last:
    "Data limitations: steps undercount gym/swimming/cycling; sleep estimated
    from screen-off proxy; alcohol and smoking self-reported only; social
    activity may undercount encrypted VOIP; stress percentages are
    probabilistic estimates not clinical measurements."
    """
    return await llm.agenerate(synthesis_prompt)
```

**Example planner output for complaint "persistent knee pain":**

```json
[
  {"step_id": 1, "data_needed": ["self_reports_30d"],
   "section_title": "Chief complaint",
   "instruction": "Summarise the patient's chief complaint in their own words."},
  {"step_id": 2, "data_needed": ["self_reports_30d", "daily_snapshots"],
   "section_title": "Symptom timeline",
   "instruction": "List dates and values for all pain_level reports in the last 30 days."},
  {"step_id": 3, "data_needed": ["daily_snapshots"],
   "section_title": "Passive data patterns",
   "instruction": "Summarise activity_level, sleep_duration, sedentary_time trends."},
  {"step_id": 4, "data_needed": ["daily_snapshots", "self_reports_30d"],
   "section_title": "Behavioural changes",
   "instruction": "Note any notable changes in lifestyle nodes around the symptom onset dates."},
  {"step_id": 5, "data_needed": ["daily_snapshots"],
   "section_title": "Contributing factors",
   "instruction": "List probable contributing factors with confidence levels. Probabilistic only — not diagnostic."}
]
```

**Key design constraint:** The planner output is validated before execution — required sections (`chief_complaint`, `data_limitations`) are checked for presence and injected if the planner omits them. This prevents the agent from skipping mandatory clinical sections.

---

### Trend Tool — `get_belief_trend` (MCP tool, mcp.ts)

**Goal:** Give the ReAct agent (and Glance mode pre-fetch) temporal awareness — the ability to describe whether a metric has been higher, lower, or stable relative to its average, rather than presenting only the current snapshot.

**Implementation:** TypeScript function `handleGetBeliefTrend` in `mcp.ts`. Reads `user_data_sensorless` (not `inference_snapshots`) — operates on user-reported values, not DBN posteriors.

**Algorithm:**
1. Query `user_data_sensorless` for each node in `node_names` within the `window_days` lookback
2. If < 3 entries: skip node (return empty string)
3. `mean` = mode of `raw_text` (most frequent label in the window)
4. `first` = earliest `raw_text` in window; `last` = most recent `raw_text`
5. If ≥ 3 entries with non-null `raw_value`: compute numeric average; classify first-third vs average and last-third vs average as `above` / `below` / `at` (10% threshold)
6. Format: `"${node}: avg ${mean}, started ${first}${firstDir !== 'at' ? ` (${firstDir} avg)` : ''}, currently ${last}${lastDir !== 'at' ? ` (${lastDir} avg)` : ''}"`

**Example output:**
```
{ "trends": {
    "stress_ema": "stress_ema: avg moderate, started high (above avg), currently low (below avg)",
    "sleep_quality": "sleep_quality: avg moderate, started moderate, currently low"
} }
```

**Two call paths:**
- **Glance:** `startTurn` calls directly when `hasTrendQuery=true && queryNodes.length > 0 && !socialFastPath`. Result injected as `TREND HISTORY (internal — use only if user asked about trends)` block in context.
- **Reflect:** Gemma calls autonomously via `TOOL_CALL: {"name": "get_belief_trend", ...}` in the ReAct loop.

**PERSONA guardrail:** Gemma never asserts the user is "improving," "making progress," "trending," or "doing better/worse over time" based on this data. The trend block informs natural-language framing; the persona rule prevents overclaiming direction as clinical progress.

---

## 21. SQLite Schema

Schema is implemented in `Codebase/backend/database.py` (`init_db()`). Three tables.

```sql
CREATE TABLE IF NOT EXISTS user_data_sensorless (
    -- Row identity
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                  TEXT NOT NULL,      -- when the entry was recorded (user local time)

    -- Node mapping
    node_name                  TEXT,              -- DBN node this entry contributes to (e.g. 'depression')
    original_column            TEXT,              -- item/base column matched by SLM Level 1 NER (e.g. 'phq_psychomotor', 'DPQ080'); null for Level 2/3
    source_column              TEXT,              -- composite or direct source column matched by SLM Level 2 NER (e.g. 'phq_total'); null for Level 1/3; derived at processing time from original_column via column_question_map.csv when Level 1

    -- Question content
    question_text              TEXT,              -- exact question shown to user (for audit + re-rendering)
    raw_text                   TEXT,              -- user's raw free-text or selected option label

    -- Discretized value (what the DBN consumes)
    node_value                 TEXT,              -- final discretized state string (e.g. 'mild', 'high')
    raw_value                  REAL,              -- numeric value before discretization (e.g. 2.0 for PHQ item)
    summary_text               TEXT,              -- human-readable summary of this entry (shown in UI recap)

    -- Evidence quality
    confidence                 REAL,              -- [0,1] certainty of node_value; fraction of scale items answered for multi-item nodes
    data_source                TEXT,              -- origin tag: 'self_report' | 'proactive' | 'onboarding'
    merge_mode                 TEXT,              -- how to combine multiple rows for same node: 'latest' | 'vote' | 'scale'

    -- Temporal handling
    temporal_flag              TEXT,              -- 'persistent' (trait) | 'decaying' (state) — controls staleness behaviour
    report_date                TEXT,              -- date the entry refers to (may differ from created_at for retrospective reports)
    expires_date               TEXT,              -- explicit expiry override; NULL = use STALENESS_DAYS from evidence layer

    -- Row lifecycle
    is_active                  INTEGER DEFAULT 1,          -- 0 = soft-deleted / superseded by newer answer
    was_proactive              INTEGER DEFAULT 0,          -- 1 = system initiated the question, 0 = user-initiated
    answered                   INTEGER DEFAULT 1,          -- 0 = question shown but skipped; kept for suppression tracking
    proactive_suppressed_until TEXT,                       -- ISO timestamp; system won't re-ask this node until after this time

    created_at                 TEXT DEFAULT (datetime('now'))  -- row write time (UTC)
);

CREATE TABLE IF NOT EXISTS sensor_windows (
    -- Window identity (composite PK = one row per inference window per day)
    date          TEXT NOT NULL,        -- calendar date of the window (YYYY-MM-DD)
    snapshot_time TEXT NOT NULL,        -- time inference was triggered within that day (HH:MM:SS)
    window_start  TEXT,                 -- start of the sensor collection window (ISO timestamp)

    -- Payload
    sensor_data   TEXT,                 -- JSON: {node_name: {node_value, confidence, data_source, created_at}} — node-level evidence from passive sensors

    created_at    TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, snapshot_time)
);

CREATE TABLE IF NOT EXISTS inference_snapshots (
    -- Snapshot identity (composite PK = one snapshot per inference run)
    date                TEXT NOT NULL,
    snapshot_time       TEXT NOT NULL,

    -- Trigger context
    trigger_type        TEXT,                 -- 'scheduled' | 'sensor_event' | 'user_query'

    -- Inputs (stored for full reproducibility)
    prior_beliefs       TEXT,                 -- JSON: {node: [p0, p1, ...]} — inter-slice temporal priors from t-1 snapshot
    sensor_snapshot     TEXT,                 -- JSON: {node: {node_value, confidence, data_source, created_at}}
    sensorless_snapshot TEXT,                 -- JSON: {node: {node_value, confidence, data_source, created_at}}

    -- Outputs
    dbn_beliefs         TEXT,                 -- JSON: {node: {state: prob, ...}} — full posterior from LBP
    node_confidences    TEXT,                 -- JSON: {node: float}
    node_data_sources   TEXT,                 -- JSON: {node: str}
    summary_line        TEXT,                 -- one-sentence natural language summary

    created_at          TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, snapshot_time)
);
```

**Column routing logic for `user_data_sensorless`:**
- Level 1 match (SLM finds a specific survey item): `original_column` = item column, `source_column` = null (derived later via `column_question_map.csv`)
- Level 2 match (SLM finds a composite/direct source column): `original_column` = null, `source_column` = composite column name
- Level 3 match (direct node): both null, `node_value` = valid state string

Never both `original_column` and `source_column` non-null simultaneously.

---

## 22. JSON Session State Schema

```json
{
  "session_id":        "2024-01-15-14-23",
  "turn":              3,
  "intent":            "symptom_report",
  "inference_trigger": "self_report",
  "snapshot_time":     "2024-01-15T14:23:00",
  "window_start":      "2024-01-15T14:08:00",
  "window_end":        "2024-01-15T14:23:00",
  "entities": [{
    "symptom_text":    "knee pain last tuesday",
    "node":            "pain_level",
    "value":           "mild",
    "confidence":      0.88,
    "node_confidence": 1.00,
    "data_source":     "self_report",
    "temporal_flag":   "today",
    "report_date":     "2024-01-09",
    "raw_minutes":     null,
    "merge_mode":      "replace"
  }],
  "awaiting_response":    "insight_offer | proactive_fill | persistence_followup | null",
  "proactive_question":   null,
  "persistence_followup": "Is this how you have been feeling generally, or just today?",
  "last_entity": {
    "node":            "pain_level",
    "value":           "mild",
    "turn":            2,
    "data_source":     "self_report",
    "confidence":      1.00,
    "snapshot_time":   "2024-01-15T14:23:00",
    "delta_snapshot":  {
      "physical_stress": 0.32,
      "mental_stress":   0.08
    }
  },
  "dbn_snapshot": {
    "pain_level":      {"none": 0.15, "mild": 0.72, "high": 0.13,
                        "data_source": "self_report", "confidence": 1.00},
    "sleep_quality":   {"poor": 0.61, "adequate": 0.30, "good": 0.09,
                        "data_source": "passive", "confidence": 0.55},
    "mental_stress":   {"low": 0.38, "moderate": 0.42, "high": 0.20,
                        "data_source": "inferred", "confidence": 0.40}
  }
}
```

---

## 23. Prompt Templates

### Template 0 — Segmentation (Pre-NER)

Runs first on every user message. Splits a compound message into individual health claim segments. Llama 3.2-1B-Instruct, `n_predict=128`, `temperature=0.0`. KV cache is pre-warmed on this prompt.

```
System: Split the user message into individual health claim segments.
Rules:
- Each segment must contain exactly one health-relevant claim.
- Keep the user's exact wording — do not paraphrase.
- If the message is a single claim, return it as the only item.
- If no health-relevant content, return an empty array.
Return ONLY a valid JSON array of strings.

Examples:
Input:  "I was really stressed today and drank coffee to feel good"
Output: ["I was really stressed today", "drank coffee to feel good"]

Input:  "bad headache and couldn't sleep last night"
Output: ["bad headache", "couldn't sleep last night"]

Input:  "feeling fine"
Output: ["feeling fine"]

Input:  "what's the weather like"
Output: []

User: {USER_INPUT}
```

**Parse contract:**
- `failed=true` (malformed JSON) → fallback: treat entire message as one segment, run NER on it.
- `failed=false, segments=[]` → skip NER entirely (no health-relevant content).
- `failed=false, segments=[...]` → run NER per segment.

---

### Template 1 — NER Extraction

#### Column-to-Node Linkage

Three levels, most specific to least specific:

```
original_column  →  source_column  →  node_name
  (item/base)       (composite or      (DBN node)
                     direct col)
```

- `original_column` — finest granularity; a specific survey item (e.g. `phq_psychomotor`, `DPQ080`). Defined with English question labels in `configs/column_question_map.csv`. SLM matches free text against `question_label`.
- `source_column` — the column that feeds the DBN node directly; either a composite aggregate (e.g. `phq_total`, multiple items sum into it) or a direct column with no items below it (e.g. `mood_how`). For composite source columns, English descriptions are in `configs/source_column_descriptions.json`. For direct source columns, descriptions are already in `column_question_map.csv`.
- `node_name` — DBN node (e.g. `depression`). Discretization thresholds and state labels in `configs/feature_node_config.json` under `source_column_bins`.

**Derivation:** when `original_column` is filled, `source_column` is derived at processing time from `column_question_map.csv` (`composite_column` field if non-empty, else `harmonized_column`). SLM never needs to output both simultaneously.

#### SLM Routing Cascade (most specific → least specific)

1. **Item-level** — text matches a specific `original_column` question label. Output `original_column = matched`, `source_column = null`, `raw_value = numeric score`, `node_value = null`. Processing derives `source_column`, aggregates items, discretizes. `confidence = 1 / total_items_in_composite`.
2. **Source-column-level** — text matches a composite or direct source column description (e.g. "my PHQ score is 15"). Output `original_column = null`, `source_column = matched composite/direct column`, `raw_value = score`, `node_value = null`. Processing discretizes directly to node. `confidence` based on data quality.
3. **Node-level** — no column match. Output `original_column = null`, `source_column = null`, `node_value = valid state string`. Direct DBN insert; no aggregation. `confidence` = SLM estimate.

**Rule:** never skip a level that fits. If "I feel hostile" matches `panas_hostile` → Level 1, not Level 3. DBN handles downstream implication — SLM must not make holistic node-state judgments when a column match exists.

**Lookup files injected into SLM prompt at runtime:**
- `column_question_map.csv` — `original_column`, `composite_column`, `question_label` for all items and direct columns
- `source_column_descriptions.json` — composite source column name → description
- `{NODE_STATES_JSON}` — valid state labels per node

#### Template

```
System: You are a health data extraction assistant.
One message may produce multiple entities. Return ONLY a valid JSON array.

Each entity:
{
  "original_column": "<item column if Level 1 match — null otherwise>",
  "source_column":   "<composite/direct column if Level 2 match — null otherwise>",
  "node_name":       "<DBN node name>",
  "node_value":      "<state string — null if Level 1 or Level 2>",
  "raw_value":       <numeric score or null>,
  "temporal_flag":   "<today|week|daily>",
  "raw_date_text":   "<exact date phrase as spoken, e.g. 'last Wednesday', '1st April' — or null>",
  "raw_minutes":     <integer or null>,
  "confidence":      <0.0-1.0>,
  "raw_text":        "<exact symptom phrase>",
  "reasoning":       "<one sentence>"
}

Routing rules (apply in order, stop at first match):
1. Match text to an item column question (column_question_map) →
   original_column = that column, source_column = null, raw_value = score, node_value = null
2. Match text to a composite/direct source column description (source_column_descriptions) →
   original_column = null, source_column = that column, raw_value = score, node_value = null
3. No column match →
   original_column = null, source_column = null, node_value = valid state from NODE_STATES

Never output both original_column and source_column as non-null simultaneously.
Never infer node state holistically when a column match exists.
For Level 1: confidence = 1 / total_items_in_composite.

Item columns and their questions (original_column → composite_column → node_name):
{COLUMN_QUESTION_MAP_JSON}

Composite/direct source column descriptions (source_column → node_name):
{SOURCE_COLUMN_DESCRIPTIONS_JSON}

Valid node states:
{NODE_STATES_JSON}

Today's date: {TODAY_DATE_ISO}
User: {USER_INPUT}
```

> **`raw_date_text` resolution:** The SLM outputs the date phrase exactly as the user said it (e.g. `"last Wednesday"`, `"two weeks ago"`). TypeScript resolves this phrase to an ISO date string using `chrono-node` with the current date as reference (`chrono.parseDate(raw_date_text, now)`). The SLM never computes dates — chrono-node handles any natural language expression. The resolved ISO date is stored as `report_date` in SQLite.

---

### Indirect-Effect Routing (v7.6, updated v7.9)

Some user statements describe behaviors that influence DBN nodes without directly naming them (e.g. "I drank a lot of coffee today"). The NER SLM (Llama 3.2-1B) does **not** attempt causal inference — it only extracts direct L1/L2/L3 matches. Indirect downstream effects are handled by Gemma 4-2B-IT.

**`NluResult.unmatched` field:** After two-pass NER completes, any health-relevant text from a segment that produced no column or node match is captured in `NluResult.unmatched: string[]`. This is the handoff from the 1B NER pass to the 2B Gemma call. The segmentation prompt already separates direct-match and indirect segments.

**How Gemma handles `unmatched` content (current implementation):**

`unmatched` segments are passed into `buildContextBlock` as part of the USER MESSAGE. In Reflect mode, Gemma observes this content and may reason about indirect effects in its THOUGHT steps, then call `store_indirect_evidence` (capped at 1 per turn, confidence ≤ 0.5) to write the inferred DBN effect. In Glance mode, Gemma receives the context and may acknowledge the indirect content in its response without writing evidence — no separate background inference call.

> The dedicated background Gemma inference call (fired immediately after NER whenever `unmatched != null`, described in v7.6) is not implemented in the current codebase. Indirect-effect reasoning is integrated into the main Gemma call via the Reflect ReAct loop.

**Segmentation + NluResult.unmatched flow:**

| User statement | Segmentation | NER | `unmatched` | Gemma handling |
|---|---|---|---|---|
| "I feel hopeless" | `["I feel hopeless"]` | `depression: moderate` (L3) → DB | `[]` | No indirect content |
| "I drank lots of coffee today" | `["I drank lots of coffee today"]` | no match | `["I drank lots of coffee today"]` | Reflect: may store_indirect_evidence for sleep/stress |
| "I ran a lot and drank lots of coffee" | `["I ran a lot", "drank lots of coffee"]` | `exercise: moderate` → DB; no match on second | `["drank lots of coffee"]` | Reflect: may reason about sleep/stress effects |

**Indirect-effect writes (Reflect mode only):** When Gemma judges that `unmatched` content implies a DBN node effect, it calls `store_indirect_evidence(node_name, node_value, confidence, summary)` with confidence ≤ 0.5. Max 1 call per turn. This is always a Gemma Reflect ReAct call, never a NER output.

**Why not teach the NER SLM causal inference?** A 1B-parameter model produces unreliable confidence estimates for multi-step causal chains. Gemma 4-2B-IT has sufficient capacity to reason about indirect effects in context and can observe the current DBN state before deciding.

---

### Template 2 — Contextual Insight (two-direction)

> **v7.5 status:** Superseded by the ReAct Agent (Pattern A, Section 20b) for all `symptom_report`, `status_change`, and `stress_query` intents. Template 2 is retained as the **fallback** for offline mode (no server connection) and as the **reference specification** for the confidence language rules and two-direction framing that the ReAct agent's final synthesis step must still follow. The confidence language guide and disclaimer requirement below apply to both paths.

```
System: You are a health insight assistant.
ROUTING: {show_effects | show_causes | general_snapshot}

If show_effects:
  User just reported: {ENTITY_TEXT} (node: {NODE}, value: {VALUE})
  User's exact words: {RAW_TEXT}
  What this report moved (delta, ranked): {DELTA_SNAPSHOT_JSON}
  DIRECTION 1 — Effects: Explain downstream causation.
  DIRECTION 2 — Causes: Explain what DBN sees driving this report.

Confidence language guide:
  > 0.70: assertive — 'Your sleep was poor (5.1hrs from HealthKit)'
  0.35-0.70: hedged — 'appears lower than usual'
  < 0.35: exclude or caveat

Always append: 'This is an estimate based on your phone behaviour
and self-reports — not a clinical measurement.'
```

### Template 3 — Stress Query Response

> **v7.5 note:** `{UNMATCHED_SUMMARIES}` is now populated by `UserMemory.retrieve_relevant()` (semantic similarity search) rather than a 7-day time-window SQL query. The placeholder name is unchanged; the retrieval mechanism behind it is not.

```
System: You are a health summary assistant.
DBN state: {DBN_SNAPSHOT_JSON}
Physical stress: {PHYSICAL_PCT}%  Mental stress: {MENTAL_PCT}%
Top physical contributors: {PHYSICAL_TOP_NODES}
Top mental contributors: {MENTAL_TOP_NODES}
Unmatched context: {UNMATCHED_SUMMARIES}

LAYER 1: State percentages. Name top 2-3 factors. Under 100 words.
LAYER 2: If unmatched context relevant, add one sentence prefixed 'You also mentioned...'.
Always append disclaimer.
```

### Template 4 — Raw Text Summarisation (storage-time)

```
System: Summarise in 1-2 lines. Preserve symptoms, body locations,
emotional states, behaviours, time references. Discard filler.
If nothing health-relevant, return empty string.
Text: {RAW_TEXT}
```

### Template 5 — Doctor Brief

> **v7.5 status:** Superseded by the Plan-and-Execute Agent (Pattern C, Section 20b). Template 5 is retained as the **fallback** for offline mode and as the **reference specification** for the mandatory sections and data limitations language the agent's synthesiser step must always produce. The six sections below map directly to the planner's expected `section_title` values.

```
System: Generate structured clinical pre-consultation brief.
30-day DBN history: {30_DAY_SNAPSHOTS_JSON}
Pre-appointment complaint: {USER_COMPLAINT}
All self-reported symptoms: {SELF_REPORTS_30_DAYS}

1. Chief complaint (patient's words)
2. Symptom timeline
3. Passive data patterns
4. Notable behavioural changes
5. Probable contributing factors (probabilistic, not diagnostic)
6. Data limitations (always include)
```

---

## 24. Alert System

Alert config in `alert_config.json`. Screen time alerts compare against total screen time — passive dark sensor plus self-reports. Alerts do not fire for nodes with confidence below 0.35.

---

## 25. Dashboard Design

**7-day view:** one chart per tracked metric.

**Per-day drill-down:** full DBN snapshot with confidence per node, stress percentages, all self-reports including `raw_text`, unmatched summaries, alerts fired.

**Intra-day timeline:** time series of stress estimates across all snapshots within each day, labelled by `trigger_type`.

**Confidence indicators:** self-report → solid colour. Passive → slight transparency. Prior-only → dashed border.

---

## 26. Doctor PDF Report

Generated on demand before appointment. Data limitations section mandatory: steps undercount gym/swimming/cycling; sleep estimated from screen-off; alcohol and smoking self-reported only; social activity may undercount encrypted VOIP; system is monitoring tool not diagnostic tool; stress percentages are probabilistic estimates not clinical measurements.

### Plan-and-Execute Agent Integration

The doctor brief is generated by the Plan-and-Execute Agent (Pattern C, Section 20b). The single Template 5 LLM call is replaced by a two-stage process: a planner that writes a 5-step execution plan tailored to the user's specific complaint, followed by an executor that runs each step independently and passes section outputs forward.

**Why this matters for the brief:** A generic template produces the same section structure for knee pain and for sleep disorder complaints. The planner adapts — a complaint about pain triggers heavier weighting on `pain_level` history and `physical_stress` trend; a mental health complaint triggers heavier weighting on `mood`, `social_activity`, and `phq_total` history.

**Mandatory sections the validator always enforces** (even if planner omits):
- Chief complaint in patient's own words
- Symptom timeline with dates and values
- Data limitations (verbatim, always last)

**Output format:** PDF generated via `reportlab`. Section headers correspond to the planner's `section_title` values. Confidence values included in parentheses next to each passive data claim — e.g., "activity_level: low (confidence: 0.71)". Self-reported values labelled as such.

---

## 27. Evaluation Strategy

### DBN Accuracy
- Log-likelihood on held-out StudentLife val uids
- Log-likelihood on held-out LifeSnaps ids (cross-population generalisation)
- Calibration: PHQ-9 >= 10 participants should have elevated `mental_stress` posteriors
- Cross-dataset edge confidence: flag edges between SL-only and LS-only nodes, verify they are not over-relied upon

### SLM Extraction Accuracy
- 60 hand-labelled test utterances, precision and recall
- Gate 2 and Gate 3 rejection rates under 5% on legitimate utterances
- Temporal flag accuracy: 20 utterances
- `report_date` resolution accuracy: 20 past-date utterances

### System Latency

| Component | Target |
|-----------|--------|
| Intent router | < 100ms (rule-based) or < 300ms (zero-shot) |
| SLM NER extraction | < 2s on target device |
| Storage-time summarisation | < 2s (background, not blocking UI) |
| Full insight generation (fixed template — offline fallback) | < 4s |
| ReAct agent insight (Gemma 4-2B-IT, on-device) | Target set by device benchmarking during Phase 4 |
| Reflection loop (Gemma 4-2B-IT, on-device) | Included within insight generation budget |
| Doctor brief — plan-and-execute (Gemma 4-2B-IT, on-device) | Background task; user not waiting in real-time |
| Dashboard load | < 1s |

### User Study (minimum 3 people)
- SUS (System Usability Scale)
- Rate each insight on 1-5 relevance scale after 1 week
- Rate Layer 2 context insights — relevant or intrusive?
- Rate notification questions — purposeful or annoying?
- Count daily notification response rate

---

## 28. Critical Build Order

> **Ordering principle:** The UI is built last, after the complete backend — data pipeline, DBN, runtime inference engine, and agentic layer — is working and tested via direct API calls and simulated data. The UI wires against the finished system once, correctly, rather than being built twice against an intermediate state. The integration test and user study run after the UI is connected to the full system.

**Phase 0 — Data Pipeline**

1. NHANES cleaning — sentinel codes → NaN, impossible values → NaN, drop age < 18
2. NHANES prior table — available-case weighted frequency per node per age/sex stratum, save to `feature_node_config.json`
3. StudentLife cleaning — fix `sleep_hours` broadcast bug first, then all physiological bounds and scale violations
4. LifeSnaps cleaning — fix age join, cast mixed types, drop structurally absent cols, physiological bounds
5. StudentLife uid-level train/val split (80/20, seed 42) — **before any threshold computation**
6. StudentLife per-uid winsorsation (sensor cols, 1st-99th percentile within uid)
7. StudentLife within-uid z-score (continuous sensor cols only)
8. StudentLife min-max normalisation (survey scale cols, fixed known ranges)
9. StudentLife survey forward-fill within uid (ffill + bfill)
10. StudentLife training-eligible row flagging (sensor alive AND survey present)
11. LifeSnaps per-id winsorsation and z-score
12. Distribution overlap check for each harmonised node (StudentLife train vs LifeSnaps)
13. Harmonised column construction — scale based on overlap result, add dataset_source flag
14. Concatenate StudentLife + LifeSnaps into combined training CSV

**Phase 1 — Discretisation**

15. K-Means discretisation (k=3) on each continuous node column — fit on StudentLife training rows only
16. Derive cut boundaries as midpoints between sorted cluster centres
17. Class balance audit — no state below ~10% of rows
18. Apply boundaries to full combined CSV (train + val + LifeSnaps)
19. Save boundaries to `feature_node_config.json`
20. Survey cols: do NOT discretise — stay continuous for E-step

**Phase 2 — DBN Training**

21. `WINDOW_MINUTES` constant in `data_harmon.py` — controls pipeline granularity
22. `clip_intervals_to_window` + `get_window_slots` helper functions
23. Sensor-alive architecture: `all_alive_windows` from validator + behavioural sensors
24. Per-sensor migration: activity, dark, phonelock, screen time, app usage, call log, SMS
25. EMA fall-through carry-forward per uid per day
26. Nighttime carry-forward columns joined to following day rows
27. `feature_node_config.json` — `merge_mode`, `temporal.valid_flags`, `proactive_ask`, `passive_proxy_strength`, `allows_self_report`, `inference_interval_minutes`, `discretisation_thresholds`, `priors`
28. Manual edge injection for any self-report node lacking training col coverage
28a. Define `forced_edges` (domain knowledge), `forbidden_edges` (dynamic→static rule minus forced), max-parents cap — write edge sets to `bn_structure_lbp.json`
28b. Startup validation: assert `forced_edges` form no cycle; assert forced ∩ forbidden = ∅ — fail fast before any EM iteration
28c. Run `data_likelihood_tables.py` — compute `P(observed_bin | node_state)` per node from training rows, save lookup tables for LBP soft evidence injection
29. Phased Structural EM — Phase 1 (40% data, HC 200 steps, perturb), Phase 2 (70%, HC 300, perturb), Phase 3 (100%, HC 500, no perturb) — see Section 5.9 for full loop specification
29a. Each HC call preceded by `_sanitize_structure()` on seed and result
29b. Plateau detection triggers `_hard_kick()` (strip 8 non-forced edges) in phases 1+2 before retrying HC
29c. After phase 1 and phase 2: `_perturb_structure()` applies 12 random edge ops (excluding forced reverses and `expert_knowledge.forbidden_edges`)
29d. Save converged structure and CPTs — `bn_structure_lbp.json` (structure JSON) + pickled model file
30. CPT spread computation — derive `HIGH_WEIGHT_NODES` per latent node, write to `feature_node_config.json`
31. Validate on StudentLife val uids → log-likelihood
32. Validate on LifeSnaps held-out ids → cross-population log-likelihood
33. Flag cross-dataset edges (SL-only parent + LS-only child) — lower confidence, document in model card
33a. Run `visualize_bn.py --struct bn_structure_lbp.json` — inspect rendered graph, verify node count matches config, all temporal self-loops visible, no isolated nodes

**Phase 3 — Runtime Inference Engine (Python/FastAPI — no UI)**

> All steps in this phase are tested via direct API calls, scripted evidence dicts, and simulated sensor data. No UI is needed or built here.

34. FastAPI server — stand up `/query_dbn` endpoint, load pickled DBN at startup. Test: mock evidence dict → valid posteriors returned.
35. Confidence computation function — `source_weight x recency x proxy_strength`, sub-daily recency
36. Prompt templates — Template 0 (segmentation), Template 1 (NER with `raw_date_text` field), Template 2–5. `raw_text` forwarding, confidence language guide
37. JSON session state schema — all fields including `report_date`, `delta_snapshot`, `inference_trigger`
38. SQLite schema — all tables: `daily_metrics`, `daily_snapshots` (composite primary key `date + snapshot_time`, `trigger_type` column), `self_reports` (with `report_date` for backdated entries), `user_profile_attributes`, `proactive_questions`
39. Evidence fusion layer — passive discretisation + merge + confidence + sub-dimension aggregation
40. `build_dbn_evidence()` — updated signature with `window_start`, `window_end`, `trigger`
41. `raw_date_text` extraction in NER pipeline — SLM extracts exact date phrase; `chrono-node` resolves phrase to ISO date in TypeScript. `Today: {YYYY-MM-DD}` injected into NER prompt. No calendar arithmetic in the SLM.
42. Confidence gate logic — all 3 gates + `NODE_CONFIDENCE_THRESHOLD`
43. Storage-time summarisation on Gate 2/3 failures
44. Intent router — rule-based first, test with 30 utterances via scripted input
45. Persistence follow-up handler
46. Proactive node check — `HIGH_WEIGHT_NODES` + confidence + 7-day suppression
47. Delta inference engine — `compute_delta()`, store in `last_entity`
48. Cause vs symptom router — `route_insight_direction()`
49. Insight button logic — `should_show_insight_button()`
50. Confidence-gated insight language
51. Alert threshold checker
52. Stress inference engine — posterior marginalisation, percentage computation
53. Background task manager — fires inference every `inference_interval_minutes`
54. Chat trigger handler
55. Daily notification system logic (server-side scheduling only — push delivery wired in Phase 5)
56. Doctor brief generator — reportlab PDF
57. Gate: 7 days of simulated sensor data through the full pipeline — both triggers fire correctly, snapshots populate SQLite, delta computed, intent router passes 30-utterance eval

**Phase 4 — Agentic Layer (on-device, React Native / TypeScript)**

> All steps tested via scripted inputs and direct method calls before any UI is connected.

58. Select and bundle the on-device embedding model — ONNX format, CPU-only, bundled in app assets. Verify consistent embeddings for semantically similar phrases.
59. Set up on-device vector store — vector index backed by the existing SQLite database. Confirm similarity queries return correct nearest neighbours on test data.
60. Build `UserMemory` class — wraps SQLite `user_profile_attributes` (structured) and on-device vector index (semantic). Implement `store()`, `retrieve_structured()`, `retrieve_relevant()`, `deduplicate_before_insert()`. Unit test all four methods independently.
61. Integrate deduplication into self-report write path — before any `self_reports` INSERT, call `deduplicate_before_insert(raw_text)`. Duplicate detected → update confidence on existing record. Test with paraphrased symptom pairs.
62. Implement `get_belief_trend` MCP tool — reads `user_data_sensorless` per node in window, computes mode label (avg), first/last labels, directional comparison from numeric `raw_value` when available. Returns `{ trends: Record<string,string> }`. Skip node if < 3 entries. Test with simulated 14-day report sequences. **[DONE — `handleGetBeliefTrend` in mcp.ts]**
63. Build Llama 1B two-pass NER pipeline — (a) `segmentText()` calls Llama with segmentation prompt, warm KV cache on seg prompt; (b) NER per segment; (c) `deduplicateEntities()` across all segment results; (d) `raw_date_text` → `chrono-node` resolution to ISO date. Test with 20 compound and single-entity utterances. Verify `NluResult.unmatched` is populated for indirect-effect segments. **[DONE — `runNer` in nlu.ts]**
63b. Build `classifyIntent` — Llama, few-shot JSON classifier, n_predict=55, temp=0.0; returns `IntentResult`. Verify all 10 routing-matrix paths produce correct classification. **[DONE — `classifyIntent` in nlu.ts]**
64. Load Gemma 4-2B-IT (Q4_K_M) on-device alongside Llama 3.2-1B-Instruct (Q4_K_M). Confirm both models coexist within device RAM budget. Verify llama.cpp build is post-April 2026 for Gemma 4 tag parsing.
65. Build Glance mode — single structured Gemma call with pre-injected context. `buildContextBlock` with `selectBeliefWindow` (gate/budget based on intent). PERSONA + RESPONSE_GUIDE in system prompt. socialFastPath forces n_predict=80. Test with 10 scripted reports including social-only turns. **[DONE — `runGlanceCall` in agent.ts]**
66. Build Reflect mode — REACT_FORMAT + `buildToolsBlock()` (GEMMA_TOOLS) appended to system prompt; ReAct loop (THOUGHT/TOOL_CALL/OBSERVATION/RESPONSE text tokens, not XML); tool registry: `run_dbn_inference`, `get_changed_nodes`, `store_indirect_evidence` (max 1/turn), `get_user_memory`, `get_belief_trend`; plain-text `TOOL_CALL:` line parsing; max 8 steps; fallback extraction on step exhaustion. Test with 10 scripted scenarios requiring multi-step reasoning. **[DONE — `runReactLoop` in agent.ts]**
67. Wire `hasTrendQuery` Glance pre-fetch — in `startTurn`, when `hasTrendQuery=true && queryNodes.length > 0 && !socialFastPath`, call `get_belief_trend` and inject result as TREND HISTORY block. Test with 5 trend-query utterances in Glance mode. **[DONE — agent.ts Stage 2]**
68. Build reflection loop — Gemma 4-2B-IT critique pass after draft insight produced. Test against 20 draft insights with known confidence violations.
69. Wire full Talk flow — `startTurn` (NER → inference → questions) → question panel (if questions > 0) → `completeTurn` (write answers → inference → Glance/Reflect → response). If no questions: inline completion in `startTurn`. Glance/Reflect mode toggle persists per screen session. **[DONE — agent.ts + TalkScreen.tsx]**
70. Implement plan-and-execute agent for doctor brief — planner call, executor loop, synthesiser call, mandatory section validator. All on-device via Gemma 4-2B-IT. Test with 5 simulated 30-day histories.
71. Gate: end-to-end agentic test — scripted compound NER (seg → per-segment NER → dedup → unmatched) → DBN update → delta → Glance insight (via full 8-stage pipeline) → insight string produced. Then same flow with Reflect (ReAct loop, store_indirect_evidence for unmatched segment). Verify `isPureUndo` bug fix by testing "stress wrong — also chest pain" retains chest pain entity. Verify no user health data leaves device except the DBN evidence dict sent to `/query_dbn`.

**Phase 5 — React Native UI**

> The UI wires against the complete, tested backend. No backend logic changes in this phase. Testing is against the full system from day one.

72. React Native Expo app scaffold — connect expo-health, expo-sqlite, expo-notifications, react-native-llama.cpp
73. Load both on-device models (Llama 3.2-1B-Instruct + Gemma 4-2B-IT, both Q4_K_M) — verify RAM budget on target device. Model paths (`NLU_MODEL_PATH`, `AGENT_MODEL_PATH`) are constants in App.tsx — set to absolute device GGUF paths before running.
74. Wire sensor collectors to window engine — activity, screen state, call log, GPS, audio, wifi, BT
75. Connect `notification_count` proxy for social activity
76. Wire timer trigger to background task manager
77. **[DONE]** State-based router in App.tsx — `screen: 'home' | 'talk' | 'journal'` state, no react-navigation dependency. DB init (`openDb` + `initDb`) and model init (`initNlu` + `warmupNlu` + `initAgent`) in `useEffect`. Session ID generated once per app session.
78. **[DONE]** HomeScreen — two cards: Talk (indigo `#4f46e5`) and Journal (violet `#7c3aed`). Props: `{ onTalk, onJournal }`.
79. **[DONE]** TalkScreen — chat bubbles (user right/model left), Glance/Reflect mode toggle in header, two-phase turn split wired (`startTurn` → question panel → `completeTurn`), question panel with opts buttons and range numeric input, Skip button, progress indicator ("N of M"), input bar hidden during question phase. Props: `{ db, sessionId, onBack }`.
80. **[DONE]** JournalScreen — large freeform TextInput, "Save Entry" button calls `runJournalTurn`, saved confirmation with "Write another" reset. Props: `{ db, sessionId, onBack }`.
81. Onboarding screens — age, sex, notification time preference only
82. Dashboard — stress percentages, 7-day trend charts, intra-day timeline, confidence indicators (solid/transparent/dashed per data_source)
83. Daily notification delivery — connect server-side scheduling to expo-notifications push
84. Report (PDF) export — trigger plan-and-execute agent on doctor brief flow, display reportlab PDF. **[NOT YET BUILT]** Planned as a third feature on HomeScreen.
85. Alert display — surface alerts from `alert_config.json` thresholds in dashboard

**Phase 6 — Integration Test and User Study**

84. End-to-end integration test — 7 days simulated data through full stack including UI, both triggers, backdated report flow, two-pass NER, Glance + Reflect insight, doctor brief export
85. SLM extraction eval — 60 hand-labelled utterances, precision and recall on intent + NER (including compound messages with unmatched segments)
86. Reflection loop eval — 20 draft insights with known violations, confirm all caught
87. User study — 3+ users, 1 week, SUS + insight ratings (Glance vs Reflect) + notification engagement rate + Layer 2 context relevance ratings

---

## 29. Architecture Decision Log

| Decision | Rationale | Version |
|----------|-----------|---------|
| Two-layer personalisation architecture | DBN with 40+ nodes overfits on ~60k training rows. Sub-dimension attribute layer decouples personalisation from DBN size. | v7.0 |
| All survey scale items restored to training data | Dropping items introduces bias. EM decides redundancy; sub-dimension layer decides runtime usage. | v7.0 |
| `app_usage numRunning` as cognitive load proxy | High numRunning during evening = mental busyness signal distinct from screen_time. | v7.0 |
| Sub-dimension expiry semantically motivated | Stable traits expire in 90-180 days. Daily states expire in 1-3 days. Single temporal_flag too coarse. | v7.0 |
| Rolling inference replaces daily batch | Insights must reflect current window, not stale daily average. | v7.1 |
| Dual trigger architecture | Two independent triggers call same pipeline. Self-reports persist in SQLite — automatically included in next timer pass. | v7.1 |
| `daily_snapshots` PRIMARY KEY changed to `(date, snapshot_time)` | Multiple snapshots per day expected. DATE alone silently overwrites earlier snapshots. | v7.1 |
| Recency decay now sub-daily | 45-min-old passive reading should be slightly less confident than 2-min-old. | v7.1 |
| Window-aware training pipeline | Training and runtime must operate at same granularity. One constant switches entire pipeline. | v7.2 |
| Sensor-alive architecture replaces quality filters | Daily filters cannot distinguish dead sensor from day with genuinely low activity. Window-level validator sensors answer this unambiguously. | v7.2 |
| EMA fall-through carry-forward | EMA responses are state declarations not point events. Treating as point events makes training data artificially sparse. | v7.2 |
| `report_date` for backdated self-reports | Without it, past-date reports mis-timestamped as today — corrupts all subsequent temporal edge propagations. | v7.2 |
| `raw_text` forwarding for linguistic specificity | DBN coarse nodes lose symptom detail. Forwarding original phrase to SLM preserves detail without violating two-brain architecture. | v7.2 |
| LifeSnaps promoted to secondary training source | StudentLife has no wearable data. LifeSnaps has heart rate, Fitbit sleep, Fitbit activity. Nodes for these constructs cannot be trained without LifeSnaps rows. | v7.3 |
| Four node training source categories | Different nodes have different data availability across datasets. Flat concatenation creates unnecessary cross-dataset NaN. Category architecture minimises NaN while maximising signal. | v7.3 |
| Distribution overlap check before harmonisation | Same construct from different instruments may not share distribution. Concatenating without checking means discretisation thresholds cut the two populations differently. | v7.3 |
| K-Means discretisation over quantile/equal-width | K-Means finds natural cluster centres. Quantile and equal-width cut through dense clusters arbitrarily, producing weaker CPT edges. | v7.3 |
| Discretisation thresholds fit on training uids only | Fitting on full dataset leaks val uid distributions into CPT boundaries — silent leakage that inflates validation metrics. | v7.3 |
| Manual edge injection for coverage gaps | An isolated node with no edges is inert — it cannot affect inference regardless of its state. Any self-report node without training col coverage requires explicit edge assertion from medical literature. | v7.3 |
| NHANES never enters training CSV | No date column, no temporal structure. Cannot teach temporal edges. Prior lookup table is its only valid contribution. Old statement that StudentLife, LifeSnaps, and GLOBEM merge into one CSV was incorrect and removed. | v7.3 |
| Passive-first philosophy | Health apps that ask too many questions get abandoned. Sensors before questions always. | v6.0 |
| Data source flag + confidence system | Solves WhatsApp problem: zero call_log calls with WhatsApp usage → low confidence flags that insight should not assert loneliness. | v6.0 |
| Two-direction insight | Users want downstream effects AND upstream causes. Single direction feels incomplete. | v6.0 |
| Confidence-gated insight language | Asserting loneliness at 0.33 confidence trains users to distrust the app. | v6.0 |
| Onboarding: age + sex only | Every additional question reduces completion ~10-15%. | v6.0 |
| Delta inference over fixed parent lists | Fixed rules ignore current passive state. Alcohol effect on stress larger when sleep already poor. | v5.0 |
| `mental_stress` and `physical_stress` remain Layer 3 | Allowing self-report collapses app into mood diary. | v5.0 |
| Storage-time summarisation | Query-time adds N sequential SLM calls → 15-20 second delays. | v4.0 |
| ReAct over fixed templates for insight generation | Fixed templates inject the same data every call. ReAct agent conditionally retrieves history, memory, and trend data only when relevant to the current report. Produces more accurate, less generic insights. | v7.5 |
| Reflection loop as separate critique pass | ReAct agent optimises for richness, not constraint enforcement. A dedicated Gemma 4-2B-IT critique pass mechanically enforces the confidence-language contract — assertive language only above 0.70 confidence. | v7.5 |
| Plan-and-execute over ReAct for doctor brief | Doctor brief has deterministic section structure — always needs timeline, passive summary, data limitations. Plan-and-execute suits structured multi-step tasks; ReAct suits uncertain interactive tasks. | v7.5 |
| On-device vector store for UserMemory | SQLite alone supports only exact node-name lookup. An on-device vector index adds semantic retrieval — surfaces a statement like "waking at 3am" when relevant to a sleep insight, even though it has no node mapping. All data stays on-device. | v7.5 |
| Embedding model bundled on-device | Sending user text to a server for embedding would breach the fully-local privacy guarantee. A lightweight ONNX embedding model bundled in app assets runs on-device CPU with no network call. | v7.5 |
| Self-report deduplication via vector similarity | Repeated near-identical phrasings of the same symptom pollute Layer 2 retrieval. Similarity-based deduplication catches rephrasing without conflating genuinely distinct reports. | v7.5 |
| Layer 2 retrieval upgraded from 7-day window to semantic search | Time-window retrieval misses relevant statements older than 7 days and includes irrelevant recent statements. Semantic similarity retrieves by relevance, not recency. | v7.5 |
| Gemma 4-2B-IT (Q4_K_M) for agent reasoning, Llama 3.2-1B-Instruct (Q4_K_M) for NER/intent | NER and intent routing are well-scoped single-output tasks the 1B model handles reliably. Multi-step agent reasoning requires a more capable model. Llama 3.2-1B stays loaded at all times; Gemma 4-2B-IT loads only when insight generation is triggered. | v7.5 |
| FastAPI role narrowed to DBN inference only | pgmpy is Python-only and cannot run in React Native. FastAPI exists solely to host the DBN inference endpoint. No user health data is stored or processed server-side beyond the evidence dict required for inference. | v7.5 |
| UI built after agentic layer, not before | Building the UI against an intermediate pipeline would require wiring it twice. Building it last means it wires against the complete tested system once. Integration test and user study then evaluate the real final system, not a prototype. | v7.5 |
| Two-pass NER segmentation (seg → per-segment NER) | Single NER call on a compound message produces unreliable multi-entity JSON. Splitting first with a short seg call guarantees one entity per NER call — the 1B model's reliable operating mode. | v7.6 |
| `raw_date_text` extracted by SLM; chrono-node resolves to ISO date | Pre-computing a fixed set of date offsets (yesterday, last Monday, etc.) fails for arbitrary user input. Offloading resolution to chrono-node handles any natural language temporal expression without burdening the 1B model with calendar arithmetic. | v7.6 |
| KV cache warmed on segmentation prompt, not NER prompt | Segmentation always runs first on every message and busts the NER cache anyway. Warming the seg prompt eliminates the always-cold first-call penalty. NER call 1 per message is unavoidably cold; calls 2+ hit the warm cache. | v7.6 |
| `NluResult.unmatched` as explicit 1B→2B connector | Indirect effects (e.g. coffee → sleep) cannot be reliably inferred by a 1B model. Capturing unmatched health-relevant text as a typed field on NluResult makes the handoff to Gemma explicit and testable — rather than implicitly relying on storage-time summarisation. | v7.6 |
| Two distinct Gemma calls (background inference + insight) | Combining indirect-effect inference and insight generation into one call produces confused output and blocks the user. Separating them lets background inference fire immediately after NER while the user reads cascade questions, then insight fires on demand. | v7.6 |
| Dual-mode agent: Rapid (~5 s) vs Deep-think (~30 s, streamed) | A single mode either feels too slow for quick reports or too shallow for complex multi-factor questions. Dual mode lets the user choose depth. Streaming makes Deep-think's latency imperceptible. | v7.6 |
| Pre-injection of context before both modes | Pre-injecting DBN state + recent obs + sensor snapshot into the system prompt eliminates `query_dbn` from the Rapid hot path (a full network round-trip) and reduces the number of ReAct tool calls needed in Deep-think. | v7.6 |
| Streaming output for both Rapid and Deep-think | A 30-second response perceived as a 30-second wait causes abandonment. Streaming begins within ~1 second and makes the experience feel fast regardless of total generation time. | v7.6 |
| Capture thinking block, do not strip | Stripping `<|channel>thought`...`<channel|>` discards the model's reasoning chain — useful for debugging, quality audits, and future training signal. Capturing it costs nothing at inference time. | v7.6 |
| Rename Rapid/Deep-think to Glance/Reflect | "Rapid" and "Deep-think" introduced user-facing phrasing before latency characteristics were locked. "Glance" and "Reflect" describe the user's intent (quick check vs deeper review), are shorter to display in a toggle, and are more durable as the feature evolves. | v7.7 |
| Two-phase turn split (startTurn / completeTurn) | Single-phase flow requires either blocking on question collection (bad UX) or discarding answers entirely (weaker inference). Phase split allows the UI to collect proactive answers asynchronously while preserving full inference context across both calls via the `_pendingTurns` map. | v7.7 |
| Inline completion when no questions built | If Phase 1 builds zero questions, a round-trip through the UI and a `completeTurn` call add latency with no benefit. Completing inline and returning `{ done: true }` keeps the fast path fast. | v7.7 |
| `_pendingTurns` Map for inter-phase state | Phase 1 beliefs and memory must reach Phase 2 without a database round-trip. An in-process Map is the lowest-latency option. The Map entry is deleted on consumption — no leak path exists. | v7.7 |
| `markEvidenceWritten()` separate from `writeProactiveAnswer` | `writeProactiveAnswer` writes to `user_data_sensorless` via a direct SQL insert, bypassing the normal evidence write path (`handleStoreIndirectEvidence`) that updates `_session.lastEvidenceWriteAt`. Without `markEvidenceWritten()`, the inference throttle treats Phase 2 as "no new evidence" and skips the second inference run. The explicit marker decouples the throttle bypass from the write path. | v7.7 |
| `DisplayQuestion` unified type for UI rendering | `FollowUpQuestion` and `CascadeQuestion` are distinct internal types with different fields. Exporting a unified `DisplayQuestion` interface means the question panel in TalkScreen contains no conditional logic based on question origin. Adding a new question source in future requires only populating the shared fields. | v7.7 |
| Journal skips Gemma entirely | Journal entries are private, self-directed, not conversational. A model response would shift the experience from journalling to chatting. NER + inference still runs so the entry enriches the DBN model silently — the user gets the benefit of structured understanding without being responded to. | v7.7 |
| Journal uses reflect-mode NER confidence gates | Journal text is often more introspective and ambiguous than chat messages. Reflect-mode gates require higher confidence before writing evidence — reduces false positives from unclear phrasing in private entries. | v7.7 |
| ReAct format uses THOUGHT/TOOL_CALL/OBSERVATION/RESPONSE text tokens, not XML tags | XML tag parsing with `<tool_call>...</tool_call>` requires the model to reliably close tags — Gemma 4-2B-IT on-device produces more consistent output with plain-text token prefixes. THOUGHT:/TOOL_CALL:/OBSERVATION:/RESPONSE: are unambiguous line-start markers that regex-parse cleanly. | v7.7 |
| Direct TypeScript tool dispatch (dispatchTool) over MCP stdio in current implementation | MCP stdio transport requires spawning child processes, establishing stdio pipes, and handling process lifecycle — overhead not justified until the tool surface grows beyond the current four tools. `dispatchTool()` calls the same underlying functions via an in-process switch. The MCP architecture remains the migration target when tool domains proliferate or need independent testing. | v7.7 |
| State-based router over react-navigation | A three-screen app with no deep linking, no tab bars, and no stack history has no routing requirements that react-navigation solves. A `screen` state variable is zero-dependency and trivially testable. react-navigation can be adopted later without touching screen components. | v7.7 |

---

## 30. Future Scope

### Wearable Integration
If wearable connected, `sleep_quality` confidence rises to 0.90 from HRV, `social_activity` gains Bluetooth proximity proxy. Confidence system reflects this automatically — no architectural change needed.

### Passive Personality Inference
After 2-4 weeks passive data, DBN infers stable personality traits from behavioural patterns without ever asking personality questions.

### App Package Categorisation for Screen Time Split
`RUNNING_TASKS_baseActivity_mPackage` splits `screen_time` into `productive`, `social`, `passive`. Android-only with graceful fallback on iOS.

### Symptom-Specific Layer 2 Nodes
Long-term solution to `raw_text` forwarding — `eye_strain` with causal parents `screen_time`, `sleep_quality`, `sedentary_time`. `raw_text` forwarding retained permanently even after symptom-specific nodes exist.

### Retroactive Snapshot Recomputation
Full recomputation pass for backdated self-reports — post-MVP. Replays inference from stated date through today propagating corrected prior through all temporal slices.

### Crisis Pipeline
Distress classifier detects crisis language → bypass SLM entirely → hardcoded human-reviewed safe response with crisis resource numbers. **Never generate crisis responses with an LLM.**

### SLM Model Upgrade
If Llama 3.2-1B-Instruct (Q4_K_M) produces unacceptable gate failure rates, upgrade to a stronger model in the same GGUF format via the same `react-native-llama.cpp` interface. Gemma 4-2B-IT is already used for agent reasoning; a parallel upgrade path applies to the NER model independently.

---

*Version 7.5 | Confidential*
