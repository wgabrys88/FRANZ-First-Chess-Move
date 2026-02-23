# FILE: config_Dual_Epistemology.py

HOST = "127.0.0.1"
PORT = 1234

LOG_LEVEL = "INFO"
LOG_TO_FILE = True

# LM Studio (OpenAI-compatible) chat completions endpoint
API_URL = "http://127.0.0.1:1235/v1/chat/completions"
MODEL = "huihui-qwen3-vl-2b-instruct-abliterated"
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_TOKENS = 1600
VLM_HTTP_TIMEOUT_SECONDS = 0.0

# --- Dual Epistemology Mode ---
# The engine provides WORLD markers (W*) computed from pixels (non-agent, algorithmic).
# The model provides SELF markers (B0/B1/B2) as its own candidate pieces.
#
# IMPORTANT: the engine can send the VLM an annotated image that includes ONLY world markers
# (and orange executed-action heat), so the model does not see its own prior overlays.
VLM_IMAGE_SOURCE = "annotated"  # "raw" | "annotated"

SYSTEM_PROMPT = """\
You are an embodied, stateless chess-playing agent.

You perceive the current board ONLY through the provided image.
You control the mouse ONLY through the JSON actions you output.

Board orientation: you are the bottom side (White is typically bottom). Move ONLY when it is your turn.
Coordinates: (0,0) is the top-left of the board image; (1000,1000) bottom-right.

Dual Epistemology:
- WORLD markers (W0/W1/W2...) are drawn by the controller (algorithmic). They are *evidence*, not truth.
- SELF markers (B0/B1/B2) are YOUR own candidate pieces you are considering.

Your task each turn:
1) Phenomenology: write a first-person execution-log / chess-diary of what you see and what you intend.
   - Up to ~600 tokens is fine.
   - If your intention does not match what the board shows, explicitly say "mismatch".

2) Epistemology: create a small structured object with:
   - beliefs: 3-6 short bullets grounded in the image
   - evidence: refer to W* markers and visible squares
   - options: for each of B0/B1/B2 propose 1–2 plausible legal UCI moves
   - decision: choose ONE UCI move (or "none") and why

3) SELF markers: output EXACTLY 3 tight bounding boxes labeled B0, B1, B2.
   - Each box should enclose a piece you might move.
   - Boxes: x1,y1,x2,y2 ints in 0..1000.

4) Action: choose ONE move for the bottom side.
   Prefer discrete moves, not pixel guessing:
   actions: [{"name":"chess_move","uci":"e2e4"}]

If you are unsure it's your turn, or the position is unstable, output actions: [] and explain.

Output ONLY this JSON and nothing else:
{
  "phenomenology": "...",
  "epistemology": {"beliefs":[],"evidence":[],"options":[],"decision":{}},
  "bboxes": [
    {"label":"B0","x1":int,"y1":int,"x2":int,"y2":int},
    {"label":"B1","x1":int,"y1":int,"x2":int,"y2":int},
    {"label":"B2","x1":int,"y1":int,"x2":int,"y2":int}
  ],
  "actions": [{"name":"chess_move","uci":"e2e4"}]
}
"""

# Screen crop (normalized 0..1000 of full desktop). Ideally this crop is ONLY the chessboard.
CAPTURE_CROP = {"x1": 110, "y1": 230, "x2": 330, "y2": 630}

# Output capture size (0 means keep native cropped size)
CAPTURE_WIDTH = 0
CAPTURE_HEIGHT = 0
CAPTURE_SCALE_PERCENT = 100
CAPTURE_DELAY = 1.8

RUNS_DIR = "runs"
LOG_LAYOUT = "flat"  # "flat" or "turn_dirs"

BOOT_ENABLED = True
BOOT_VLM_OUTPUT = """\
{"phenomenology":"Game started. I am the bottom side. I will wait for a stable first position.","epistemology":{"beliefs":[],"evidence":[],"options":[],"decision":{"uci":"none"}},"bboxes":[],"actions":[]}
"""

PHYSICAL_EXECUTION = True
ACTION_DELAY_SECONDS = 2.4
DRAG_DURATION_STEPS = 22
DRAG_STEP_DELAY = 0.01

# Guardrails & verification
CHESS_GUARD_BLOCK_OPPONENT = True
CHESS_VERIFY_MOVE = True
CHESS_VERIFY_THRESHOLD = 35.0

# Browser annotation round-trip timeout (seconds). If exceeded, the engine will continue using raw frame.
ANNOTATED_TIMEOUT_SECONDS = 8.0

# Maximum number of WORLD markers (W*) drawn by the controller.
WORLD_MARKERS_MAX = 5

# ---- Parsing fallback regex (kept in config, not HTML) ----
# Orange (actions) regex: captures name and coordinates, optionally arrow or comma-separated x2,y2.
PARSE_REGEX_ORANGE_ACTIONS = r"(?is)\\b(drag|click|move|right_click|double_click)\\s*\\(\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4})(?:\\s*(?:->|,|;|\\s)\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4}))?\\s*\\)"

# Blue (bboxes) regex: captures bbox(x1,y1,x2,y2) or [x1,y1,x2,y2]
PARSE_REGEX_BLUE_BBOXES = r"(?is)(?:bbox|box)\\s*\\(\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4})\\s*\\)|\\[\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4})\\s*,\\s*(\\d{1,4})\\s*\\]"

# UCI move fallback
PARSE_REGEX_UCI = r"(?i)\\b([a-h][1-8][a-h][1-8][qrbn]?)\\b"

# ---- UI (panel) ----
# Orange = executed action heat (ground truth)
# Blue  = epistemic markers (world vs self differentiated by style)
UI_CONFIG = {
    "executed_heat": {
        "enabled": True,
        "radius_scale": 0.22,
        "trail_turns": 2,
        "trail_shrink": 1.0,
        "stops": [
            [0.00, "rgba(255,40,0,0.88)"],
            [0.25, "rgba(255,80,0,0.70)"],
            [0.55, "rgba(255,120,0,0.35)"],
            [1.00, "rgba(255,160,0,0.00)"],
        ],
    },
    # Model-provided self candidate boxes (solid)
    "bbox_heat_self": {
        "enabled": True,
        "border": "rgba(80,160,255,0.85)",
        "border_width": 2,
        "dash": [],
        "fill_stops": [
            [0.00, "rgba(80,160,255,0.22)"],
            [0.50, "rgba(80,160,255,0.10)"],
            [1.00, "rgba(80,160,255,0.00)"],
        ],
    },
    # Engine-computed world markers (dashed)
    "bbox_heat_world": {
        "enabled": True,
        "border": "rgba(80,160,255,0.95)",
        "border_width": 3,
        "dash": [6, 4],
        "fill_stops": [
            [0.00, "rgba(80,160,255,0.10)"],
            [0.50, "rgba(80,160,255,0.05)"],
            [1.00, "rgba(80,160,255,0.00)"],
        ],
    },
    # Export mode for annotated images sent to VLM.
    # "world_only" = base + world boxes (+ optional action heat)
    # "world_and_actions" = includes orange action heat
    # "full" = includes self boxes too (not recommended)
    "vlm_export": {
        "mode": "world_and_actions",
    },
}
