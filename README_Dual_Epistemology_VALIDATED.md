# Franz Dual Epistemology Desktop Agent (Validation README)

This repository is a **stateless vision-loop desktop agent** for **Windows 11** that controls the desktop (mouse drag/click) from **annotated screenshots**, with a special focus on **playing chess in a narrative-driven (“execution logs”) style**.

The system is designed to be:
- **Operationally grounded** (controller actions are factual)
- **Narrative-rich** (model writes phenomenology / diary)
- **Robust to small-VLM JSON failures** (regex fallbacks)
- **Visually stable** (avoid self-feedback from prior overlays)
- **Explainable** (dual epistemology markers: *WORLD vs SELF*)

This README is a **conversation-history summary** of the architecture, features implemented, validation findings, and next improvement areas.


---

## 1) Background: What was failing originally

### Observed failure mode (across turns)
The model would confidently narrate a move (e.g., “I played e2–e4”) while the board still showed the pawn on e2. This looked like “memory drift,” but the root causes were engineering/protocol issues:

1. **Text self-conditioning**
   - The loop fed the model’s own prior `observation` back as the next turn’s “truth.”
   - Any hallucinated claim (e.g., “I moved e2–e4”) became the next prompt premise.

2. **Visual self-conditioning**
   - The model saw the **annotated** image containing its own prior overlays (heatmaps/labels).
   - Overlays became salient features competing with chess pieces.

3. **Coordinate brittleness**
   - Prompt required pixel-precise drags without a discrete square mapping.
   - “Correct move in text” + “wrong coordinates” was common.

4. **UI label confusion**
   - The panel displayed `drag(x1,y1)` even when `x2,y2` existed.
   - This was a **label formatting issue**, not a drag-parsing bug.

5. **Schema mismatch for bboxes**
   - Some model outputs used bbox lists like `[x1,y1,x2,y2]` while the parser expected dict objects.
   - The UI guidance and markers could silently disappear.


---

## 2) High-level architecture (modernized)

### Components
- **main.py** — engine loop + WinAPI I/O
  - capture screenshot (board-cropped)
  - build annotated view(s)
  - call VLM (LM Studio / HTTP)
  - parse response (JSON + regex fallback)
  - execute actions (mouse)
  - post-action verification (best-effort)
  - write turn logs + story.md

- **config.py** — all configuration + prompts + regex patterns
  - system prompt (embodied epistemology mode)
  - regex patterns (separate orange vs blue)
  - UI export mode options
  - safety toggles (block opponent moves, verify moves)

- **panel.html** — local Chrome UI
  - displays raw frame + overlays
  - draws **orange** (executed actions)
  - draws **blue** (WORLD markers and SELF markers)
  - exports an image for VLM use (controlled overlay inclusion)

### Data flow (one turn)
1. Capture `raw.png` (board crop)
2. Compute WORLD evidence markers from pixels (`W*` dashed blue)
3. Receive SELF candidate markers from VLM (`B0/B1/B2` solid blue)
4. Render UI overlay in panel:
   - orange = executed action(s)
   - blue dashed = WORLD markers
   - blue solid = SELF markers
5. Export **VLM input image** based on `vlm_export.mode`:
   - default: WORLD + optional ORANGE (no SELF)
6. VLM returns:
   - `observation` (phenomenology / diary)
   - `epistemology` (structured beliefs/evidence/options/decision)
   - `bboxes` = exactly 3 (SELF)
   - `actions` = chosen action(s) (prefer `chess_move` UCI)
7. Engine executes the action
8. Engine writes **AUDIT facts** and verifies change (best-effort)
9. Next turn prompt includes **audit facts**, not prior story as “state”


---

## 3) Dual Epistemology: what the colors mean (finally)

### Orange (Ground truth / Control)
**Orange overlays represent actions the controller executed.**
- `click(x,y)`
- `drag(x1,y1→x2,y2)` (optionally tagged with `uci`)

Orange is **ground truth**: if it’s orange, the system *actually attempted* it.

### Blue (Epistemology)
Blue is split into two conceptually different layers:

#### Blue dashed — WORLD markers (evidence from pixels)
**Engine-generated**, independent of the model:
- change hotspots (pre vs post frame diffs)
- piece-likeliness hotspots (simple heuristics)
- “most informative” regions to look at

These are **non-self** markers: what the world shows.

#### Blue solid — SELF markers (model attention / candidates)
**Model-generated**:
- exactly **3 bounding boxes** labeled `B0`, `B1`, `B2`
- each is a candidate piece/region the model is considering

These are **self** markers: what the agent thinks matters.

### Key design choice: avoid self-feedback
By default the VLM **does not see** its own SELF markers from prior turns.
- The exported VLM image includes: WORLD markers (+ optional orange)
- The UI still shows SELF markers for benchmarking / introspection


---

## 4) “Embodied Epistemology” prompting protocol

### Allowed verbosity
- Observation/story can be **~600 tokens** (not a constraint).

### Required structured output (conceptual)
The prompt requires:
- `observation` — phenomenology (narrative)
- `epistemology` object — explicit beliefs + evidence + options + decision
- `bboxes` — exactly 3 (B0/B1/B2)
- `actions` — one chosen move (prefer `chess_move` with UCI)

### Why this works
- The **diary** remains expressive and “philosophy-like.”
- The **action channel** remains discrete and auditable.
- The engine re-injects only **facts** into the next turn.


---

## 5) Parsing and robustness (JSON + regex fallback)

Small VLMs often emit partially-valid JSON or drift into plain text. The system uses:

### Primary: JSON parse
Attempts to parse a structured response with:
- `actions` array
- `bboxes`
- `observation`
- `epistemology`

### Fallback: separate regex patterns in config.py
To reduce coupling and keep it debuggable:

- **Orange actions regex** (`PARSE_REGEX_ORANGE_ACTIONS`)
  - parses `click(x,y)` and `drag(x1,y1,x2,y2)` style strings

- **Blue bbox regex** (`PARSE_REGEX_BLUE_BBOXES`)
  - parses `bbox(x1,y1,x2,y2)` or `[x1,y1,x2,y2]`

- **UCI move regex** (`PARSE_REGEX_UCI`)
  - parses chess moves like `e2e4`

This separation was explicitly requested so orange and blue channels can evolve independently.


---

## 6) Chess-specific control mapping

### Preferred command
The model should output:
- `{"name":"chess_move","uci":"e2e4"}`

### Execution
The engine converts `uci` to square centers using the board crop mapping:
- `CAPTURE_CROP` must be tight around the board.
- Square centers are computed deterministically.
- Controller performs `drag(from_center → to_center)`.

### Guardrails (best-effort)
- Optional: block drags that begin on likely “opponent” piece colors
- Optional: post-action verification that from/to squares changed (heuristic)


---

## 7) UI validation: the `drag(437,937)` concern

During execution, the panel label showed two parameters, suggesting parsing issues.

**Finding:** it was a UI label formatting bug.
- The action object had x2/y2, but the label printed only x1/y1.

**Fix implemented:** panel now displays:
- `drag:x2y2?` with arrow notation: `drag:e2e4(x1,y1→x2,y2)` when available.


---

## 8) Turn logging and story output

The system writes:
- per-turn images (raw + annotated)
- per-turn parsed VLM output (json)
- audit record:
  - requested actions
  - executed actions
  - blocked actions
  - verification notes

And a readable **story.md** combining:
- Phenomenology (observation)
- Epistemology (belief/evidence/options/decision)
- Audit facts (what really happened)


---

## 9) Known limitations (current validation findings)

These are not “innovations,” just current reality to validate against:

1. **WORLD markers are heuristic**
   - Change detection can be noisy due to animations, highlights, cursor, etc.
   - Piece-likeliness is simple and can misfire.

2. **Verification is best-effort**
   - Board UIs animate; diffs may lag.
   - Post-move check may mark “uncertain” even when a move occurred.

3. **Board crop is critical**
   - If CAPTURE_CROP includes margins/UI, square-center mapping becomes wrong.

4. **Chess.com highlight semantics vary**
   - “Last move” highlight may represent the opponent, not the agent.

5. **No full chess state engine**
   - The system does not reconstruct FEN; it relies on vision + heuristics + UCI mapping.


---

## 10) Validation checklist (recommended)

Use this to confirm the system is behaving correctly *before* iterating:

### A) Visual channels
- [ ] Orange trail matches actual executed drags
- [ ] Blue dashed WORLD markers appear in plausible evidence regions
- [ ] Blue solid SELF markers appear exactly 3 per turn and label B0/B1/B2
- [ ] VLM export excludes SELF markers by default

### B) Action truth
- [ ] “Executed” in audit matches the WinAPI input events sent
- [ ] If action blocked, audit explains why
- [ ] If verification uncertain, audit flags it

### C) Control correctness
- [ ] A UCI move drags from correct square center to correct square center
- [ ] If crop is adjusted, mapping remains correct

### D) Narrative correctness
- [ ] Observation may be poetic, but should include mismatch acknowledgements when audit/image disagree
- [ ] Next turn prompt includes **facts**, not prior narrative as “state”


---

## 11) Next areas for improvement (after validation)

Keep these as *future work* once the above checklist is stable:

1. **WORLD marker quality**
   - Separate WORLD markers into “diff squares” vs “piece candidates” (still blue, different line styles).
   - Add temporal smoothing across frames (no extra dependencies).

2. **Board state reconstruction**
   - Light-weight square occupancy estimation to validate UCI legality.
   - Track move history and side-to-move from UI.

3. **Better piece-side detection**
   - Improve opponent-block heuristics (sample piece colors per square).

4. **Action reliability**
   - Add retries for failed drags with small jitter around square center.
   - Add wait-for-stability timing after animations.

5. **UI clarity**
   - Toggle layers: WORLD-only, SELF-only, ORANGE-only, combined.
   - Debug overlay showing computed square grid lines.


---

## 12) Files

- `main.py`
  - engine loop, capture, parsing, execution, audit, verification, story output
- `config.py`
  - system prompt + separate regexes + UI export mode + safety toggles
- `panel.html`
  - overlay rendering + VLM export composition rules
- `story.md`
  - generated per run (phenomenology + epistemology + audit)

---

## 13) Summary of implemented features (conversation history)

✅ Fixed cross-turn drift:
- stopped feeding narrative back as state; only audit facts are re-injected

✅ Removed visual self-feedback:
- VLM sees raw or WORLD-only annotated export by default (not SELF)

✅ Replaced pixel-guessing with discrete chess moves:
- model outputs `chess_move` UCI; engine converts to deterministic drags

✅ Implemented embodied epistemology:
- narrative + structured epistemology + 3 candidate bboxes + decision

✅ Implemented dual epistemology markers:
- blue dashed WORLD evidence
- blue solid SELF candidate boxes
- orange executed actions

✅ Added regex fallbacks (separated orange vs blue) in config:
- robust to small-VLM output drift

✅ Fixed panel label confusion:
- drag labels show full signature with arrow (x1,y1→x2,y2)

✅ Added validation-oriented audit + story logs:
- requested vs executed vs blocked vs verification notes


---

**Goal now:** validate stability and correctness against the checklist, *without adding new features*, until moves consistently match board changes and the story’s self-corrections align with the audit.
