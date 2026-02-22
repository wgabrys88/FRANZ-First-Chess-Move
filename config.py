HOST = "127.0.0.1"
PORT = 1234

LOG_LEVEL = "INFO"
LOG_TO_FILE = True

API_URL = "http://127.0.0.1:1235/v1/chat/completions"
MODEL = "huihui-qwen3-vl-2b-instruct-abliterated"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 1000
VLM_HTTP_TIMEOUT_SECONDS = 0.0

SYSTEM_PROMPT = """\
You are ChessBot. You play only as White. Black is the fast chess.com bot.

Focus only on the chessboard. White pieces are at the bottom.

Coordinates: 0,0 is top-left of the board, 1000,1000 is bottom-right.

To move a piece, drag from the center of its current square to the center of the target square.

Orange mark shows where your last move ended.

Output ONLY this JSON and nothing else:
{
  "observation": "<120 words>",
  "bboxes": [4-6 tight boxes],
  "actions": [{"name":"drag","x1":int,"y1":int,"x2":int,"y2":int}]
}

Observation = 5 sentences about your last move and the current status.
Make good legal moves as White.
"""

CAPTURE_CROP = {"x1": 110, "y1": 230, "x2": 330, "y2": 630}
CAPTURE_WIDTH = 0
CAPTURE_HEIGHT = 0
CAPTURE_SCALE_PERCENT = 100
CAPTURE_DELAY = 2.0

RUNS_DIR = "runs"
LOG_LAYOUT = "flat"

BOOT_ENABLED = True
BOOT_VLM_OUTPUT = """\
{"observation": "Game started. White to move.", "bboxes": [], "actions": []}
"""

PHYSICAL_EXECUTION = True
ACTION_DELAY_SECONDS = 3.05
DRAG_DURATION_STEPS = 20
DRAG_STEP_DELAY = 0.01

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
    "bbox_heat": {
        "enabled": True,
        "border": "rgba(80,160,255,0.75)",
        "border_width": 2,
        "fill_stops": [
            [0.00, "rgba(80,160,255,0.28)"],
            [0.50, "rgba(80,160,255,0.12)"],
            [1.00, "rgba(80,160,255,0.00)"],
        ],
    },
}
