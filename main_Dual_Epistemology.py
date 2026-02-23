# FILE: main_Dual_Epistemology.py

from __future__ import annotations

import asyncio
import base64
import ctypes
import ctypes.wintypes as W
import http.client
import json
import logging
import os
import re
import signal
import struct
import time
import urllib.parse
import webbrowser
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

HERE: Final[Path] = Path(__file__).resolve().parent
CONFIG_PATH: Final[Path] = HERE / "config.py"
PANEL_HTML: Final[Path] = HERE / "panel.html"


def _load_config() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", str(CONFIG_PATH))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


CFG: Any = _load_config()

HOST: Final[str] = str(getattr(CFG, "HOST", "127.0.0.1"))
PORT: Final[int] = int(getattr(CFG, "PORT", 1234))

log = logging.getLogger("franz")


def _cfg(name: str, default: Any = None) -> Any:
    return getattr(CFG, name, default)


def setup_logging(run_dir: Path) -> None:
    level = getattr(logging, str(_cfg("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    fmt = logging.Formatter(
        "[%(name)s][%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if bool(_cfg("LOG_TO_FILE", True)):
        fh = logging.FileHandler(run_dir / "main.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    log.info("logging ready level=%s run_dir=%s", logging.getLevelName(level), run_dir)


def make_run_dir() -> Path:
    runs_base = HERE / str(_cfg("RUNS_DIR", "runs"))
    runs_base.mkdir(exist_ok=True)
    existing = sorted(
        [d for d in runs_base.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    run_dir = runs_base / f"run_{len(existing) + 1:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------- State ----------------


@dataclass
class EngineState:
    phase: str = "init"
    error: str | None = None
    turn: int = 0
    run_dir: Path | None = None

    raw_b64: str = ""
    raw_w: int = 0
    raw_h: int = 0
    raw_bgra: bytes = b""  # cropped frame

    annotated_b64: str = ""
    pending_seq: int = 0
    annotated_seq: int = -1
    annotated_event: asyncio.Event = field(default_factory=asyncio.Event)

    # latest model output
    vlm_json: str = ""
    phenomenology: str = ""
    epistemology: dict[str, Any] = field(default_factory=dict)

    # dual epistemology overlays
    self_bboxes: list[dict[str, Any]] = field(default_factory=list)
    world_bboxes: list[dict[str, Any]] = field(default_factory=list)

    # actions
    requested_actions: list[dict[str, Any]] = field(default_factory=list)
    executed_actions: list[dict[str, Any]] = field(default_factory=list)

    msg_id: int = 0

    next_vlm_json: str | None = None
    next_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


S: EngineState
STOP: asyncio.Event


def set_phase(phase: str, error: str | None = None) -> None:
    S.phase = phase
    S.error = error
    log.info("phase=%s error=%s", phase, error)


# ---------------- WinAPI screenshot + mouse ----------------

SRCCOPY: Final[int] = 0x00CC0020
CAPTUREBLT: Final[int] = 0x40000000
BI_RGB: Final[int] = 0
DIB_RGB: Final[int] = 0
HALFTONE: Final[int] = 4

try:
    ctypes.WinDLL("shcore", use_last_error=True).SetProcessDpiAwareness(2)
except Exception:
    pass

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)


def _sig(dll: Any, name: str, argtypes: list[Any], restype: Any) -> None:
    fn = getattr(dll, name)
    fn.argtypes = argtypes
    fn.restype = restype


_sig(_user32, "GetDC", [W.HWND], W.HDC)
_sig(_user32, "ReleaseDC", [W.HWND, W.HDC], ctypes.c_int)
_sig(_user32, "GetSystemMetrics", [ctypes.c_int], ctypes.c_int)
_sig(_gdi32, "CreateCompatibleDC", [W.HDC], W.HDC)
_sig(
    _gdi32,
    "CreateDIBSection",
    [W.HDC, ctypes.c_void_p, W.UINT, ctypes.POINTER(ctypes.c_void_p), W.HANDLE, W.DWORD],
    W.HBITMAP,
)
_sig(_gdi32, "SelectObject", [W.HDC, W.HGDIOBJ], W.HGDIOBJ)
_sig(
    _gdi32,
    "BitBlt",
    [
        W.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        W.HDC,
        ctypes.c_int,
        ctypes.c_int,
        W.DWORD,
    ],
    W.BOOL,
)
_sig(
    _gdi32,
    "StretchBlt",
    [
        W.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        W.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        W.DWORD,
    ],
    W.BOOL,
)
_sig(_gdi32, "SetStretchBltMode", [W.HDC, ctypes.c_int], ctypes.c_int)
_sig(_gdi32, "SetBrushOrgEx", [W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_void_p], W.BOOL)
_sig(_gdi32, "DeleteObject", [W.HGDIOBJ], W.BOOL)
_sig(_gdi32, "DeleteDC", [W.HDC], W.BOOL)
_sig(_user32, "SetCursorPos", [ctypes.c_int, ctypes.c_int], W.BOOL)
_sig(_user32, "mouse_event", [W.DWORD, W.DWORD, W.DWORD, W.DWORD, ctypes.c_ulong], None)

MOUSEEVENTF_LEFTDOWN: Final[int] = 0x0002
MOUSEEVENTF_LEFTUP: Final[int] = 0x0004
MOUSEEVENTF_RIGHTDOWN: Final[int] = 0x0008
MOUSEEVENTF_RIGHTUP: Final[int] = 0x0010


class _BIH(ctypes.Structure):
    _fields_ = [
        ("biSize", W.DWORD),
        ("biWidth", W.LONG),
        ("biHeight", W.LONG),
        ("biPlanes", W.WORD),
        ("biBitCount", W.WORD),
        ("biCompression", W.DWORD),
        ("biSizeImage", W.DWORD),
        ("biXPelsPerMeter", W.LONG),
        ("biYPelsPerMeter", W.LONG),
        ("biClrUsed", W.DWORD),
        ("biClrImportant", W.DWORD),
    ]


class _BMI(ctypes.Structure):
    _fields_ = [("bmiHeader", _BIH), ("bmiColors", W.DWORD * 3)]


def _make_bmi(w: int, h: int) -> _BMI:
    bmi = _BMI()
    hdr = bmi.bmiHeader
    hdr.biSize = ctypes.sizeof(_BIH)
    hdr.biWidth = w
    hdr.biHeight = -h
    hdr.biPlanes = 1
    hdr.biBitCount = 32
    hdr.biCompression = BI_RGB
    return bmi


def _clampi(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _screen_size() -> tuple[int, int]:
    w, h = int(_user32.GetSystemMetrics(0)), int(_user32.GetSystemMetrics(1))
    return (w, h) if w > 0 and h > 0 else (1920, 1080)


NORM_MAX: Final[int] = 1000


def _nedge(v: int, span: int) -> int:
    v = _clampi(v, 0, NORM_MAX)
    return (v * span + NORM_MAX // 2) // NORM_MAX


def _npt(v: int, span: int) -> int:
    v = _clampi(v, 0, NORM_MAX)
    return 0 if span <= 1 else (v * (span - 1) + NORM_MAX // 2) // NORM_MAX


def _crop_px(base_w: int, base_h: int) -> tuple[int, int, int, int]:
    c = _cfg("CAPTURE_CROP", {"x1": 0, "y1": 0, "x2": NORM_MAX, "y2": NORM_MAX})
    if not isinstance(c, dict):
        return 0, 0, base_w, base_h
    x1 = int(c.get("x1", 0))
    y1 = int(c.get("y1", 0))
    x2 = int(c.get("x2", NORM_MAX))
    y2 = int(c.get("y2", NORM_MAX))
    x1, x2 = (_clampi(x1, 0, NORM_MAX), _clampi(x2, 0, NORM_MAX))
    y1, y2 = (_clampi(y1, 0, NORM_MAX), _clampi(y2, 0, NORM_MAX))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px1 = _nedge(x1, base_w)
    py1 = _nedge(y1, base_h)
    px2 = _nedge(x2, base_w)
    py2 = _nedge(y2, base_h)
    px1 = max(0, min(px1, base_w))
    py1 = max(0, min(py1, base_h))
    px2 = max(px1, min(px2, base_w))
    py2 = max(py1, min(py2, base_h))
    return px1, py1, px2, py2


def _norm_to_screen_xy(nx: int, ny: int) -> tuple[int, int]:
    sw, sh = _screen_size()
    x1, y1, x2, y2 = _crop_px(sw, sh)
    return x1 + _npt(nx, x2 - x1), y1 + _npt(ny, y2 - y1)


def _create_dib(dc: Any, w: int, h: int) -> tuple[Any, int]:
    bits = ctypes.c_void_p()
    hbmp = _gdi32.CreateDIBSection(dc, ctypes.byref(_make_bmi(w, h)), DIB_RGB, ctypes.byref(bits), None, 0)
    return (hbmp, int(bits.value)) if hbmp and bits.value else (None, 0)


def _capture_bgra_full() -> tuple[bytes, int, int] | None:
    sw, sh = _screen_size()
    sdc = _user32.GetDC(0)
    if not sdc:
        return None
    memdc = _gdi32.CreateCompatibleDC(sdc)
    if not memdc:
        _user32.ReleaseDC(0, sdc)
        return None
    hbmp, bits = _create_dib(sdc, sw, sh)
    if not hbmp:
        _gdi32.DeleteDC(memdc)
        _user32.ReleaseDC(0, sdc)
        return None
    old = _gdi32.SelectObject(memdc, hbmp)
    _gdi32.BitBlt(memdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY | CAPTUREBLT)
    raw = bytes((ctypes.c_ubyte * (sw * sh * 4)).from_address(bits))
    _gdi32.SelectObject(memdc, old)
    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(memdc)
    _user32.ReleaseDC(0, sdc)
    return raw, sw, sh


def _crop_bgra(bgra: bytes, sw: int, sh: int, crop: dict[str, int]) -> tuple[bytes, int, int]:
    x1, y1 = max(0, min(crop["x1"], sw)), max(0, min(crop["y1"], sh))
    x2, y2 = max(x1, min(crop["x2"], sw)), max(y1, min(crop["y2"], sh))
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return bgra, sw, sh
    out = bytearray(cw * ch * 4)
    ss, ds = sw * 4, cw * 4
    for y in range(ch):
        so, do = (y1 + y) * ss + x1 * 4, y * ds
        out[do : do + ds] = bgra[so : so + ds]
    return bytes(out), cw, ch


def _stretch_bgra(bgra: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes | None:
    sdc = _user32.GetDC(0)
    if not sdc:
        return None
    src_dc = _gdi32.CreateCompatibleDC(sdc)
    dst_dc = _gdi32.CreateCompatibleDC(sdc)
    if not src_dc or not dst_dc:
        if src_dc:
            _gdi32.DeleteDC(src_dc)
        if dst_dc:
            _gdi32.DeleteDC(dst_dc)
        _user32.ReleaseDC(0, sdc)
        return None

    src_bmp, src_bits = _create_dib(sdc, sw, sh)
    if not src_bmp:
        _gdi32.DeleteDC(src_dc)
        _gdi32.DeleteDC(dst_dc)
        _user32.ReleaseDC(0, sdc)
        return None
    ctypes.memmove(src_bits, bgra, sw * sh * 4)
    old_src = _gdi32.SelectObject(src_dc, src_bmp)

    dst_bmp, dst_bits = _create_dib(sdc, dw, dh)
    if not dst_bmp:
        _gdi32.SelectObject(src_dc, old_src)
        _gdi32.DeleteObject(src_bmp)
        _gdi32.DeleteDC(src_dc)
        _gdi32.DeleteDC(dst_dc)
        _user32.ReleaseDC(0, sdc)
        return None
    old_dst = _gdi32.SelectObject(dst_dc, dst_bmp)

    _gdi32.SetStretchBltMode(dst_dc, HALFTONE)
    _gdi32.SetBrushOrgEx(dst_dc, 0, 0, None)
    _gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, SRCCOPY)

    result = bytes((ctypes.c_ubyte * (dw * dh * 4)).from_address(dst_bits))

    _gdi32.SelectObject(dst_dc, old_dst)
    _gdi32.SelectObject(src_dc, old_src)
    _gdi32.DeleteObject(dst_bmp)
    _gdi32.DeleteObject(src_bmp)
    _gdi32.DeleteDC(dst_dc)
    _gdi32.DeleteDC(src_dc)
    _user32.ReleaseDC(0, sdc)
    return result


def _bgra_to_png(bgra: bytes, w: int, h: int) -> bytes:
    stride = w * 4
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        row = bgra[y * stride : (y + 1) * stride]
        for i in range(0, len(row), 4):
            raw.extend((row[i + 2], row[i + 1], row[i], 255))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def capture_frame(with_bgra: bool = True) -> tuple[str, int, int, bytes]:
    if (delay := float(_cfg("CAPTURE_DELAY", 0.0))) > 0:
        time.sleep(delay)
    cap = _capture_bgra_full()
    if cap is None:
        return "", 0, 0, b""
    bgra, sw, sh = cap

    # crop
    crop = _cfg("CAPTURE_CROP")
    if crop and isinstance(crop, dict) and all(k in crop for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = _crop_px(sw, sh)
        bgra, sw, sh = _crop_bgra(bgra, sw, sh, {"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # scale
    out_w, out_h = int(_cfg("CAPTURE_WIDTH", 0)), int(_cfg("CAPTURE_HEIGHT", 0))
    dw = dh = 0
    if out_w > 0 and out_h > 0:
        dw, dh = out_w, out_h
    else:
        p = int(_cfg("CAPTURE_SCALE_PERCENT", 100) or 100)
        if p > 0 and p != 100:
            dw = max(1, (sw * p + 50) // 100)
            dh = max(1, (sh * p + 50) // 100)

    if dw > 0 and dh > 0 and (sw, sh) != (dw, dh):
        s = _stretch_bgra(bgra, sw, sh, dw, dh)
        if s:
            bgra, sw, sh = s, dw, dh

    b64 = base64.b64encode(_bgra_to_png(bgra, sw, sh)).decode("ascii")
    log.info("capture done %dx%d b64len=%d", sw, sh, len(b64))
    return b64, sw, sh, (bgra if with_bgra else b"")


def _move_to(x: int, y: int) -> None:
    _user32.SetCursorPos(x, y)


def _mouse(flags: int) -> None:
    _user32.mouse_event(flags, 0, 0, 0, 0)


def execute_actions(actions: list[dict[str, Any]]) -> None:
    if not bool(_cfg("PHYSICAL_EXECUTION", True)):
        log.info("PHYSICAL_EXECUTION=False, skipping %d actions", len(actions))
        return

    action_delay = float(_cfg("ACTION_DELAY_SECONDS", 0.05))
    drag_steps = int(_cfg("DRAG_DURATION_STEPS", 20))
    drag_step_d = float(_cfg("DRAG_STEP_DELAY", 0.01))

    for a in actions:
        name = str(a.get("name", "")).lower()
        nx1, ny1 = int(a.get("x1", 0)), int(a.get("y1", 0))
        nx2, ny2 = int(a.get("x2", nx1)), int(a.get("y2", ny1))
        x1, y1 = _norm_to_screen_xy(nx1, ny1)
        x2, y2 = _norm_to_screen_xy(nx2, ny2)
        log.info(
            "execute action=%s uci=%s nx1=%d ny1=%d nx2=%d ny2=%d px1=%d py1=%d px2=%d py2=%d",
            name,
            a.get("uci"),
            nx1,
            ny1,
            nx2,
            ny2,
            x1,
            y1,
            x2,
            y2,
        )

        match name:
            case "move":
                _move_to(x1, y1)
            case "click":
                _move_to(x1, y1)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_LEFTDOWN)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_LEFTUP)
            case "right_click":
                _move_to(x1, y1)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_RIGHTDOWN)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_RIGHTUP)
            case "double_click":
                _move_to(x1, y1)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_LEFTDOWN)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_LEFTUP)
                time.sleep(0.06)
                _mouse(MOUSEEVENTF_LEFTDOWN)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_LEFTUP)
            case "drag":
                _move_to(x1, y1)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_LEFTDOWN)
                time.sleep(0.03)
                steps = max(1, drag_steps)
                for i in range(1, steps + 1):
                    tx = x1 + (x2 - x1) * i // steps
                    ty = y1 + (y2 - y1) * i // steps
                    _move_to(tx, ty)
                    time.sleep(drag_step_d)
                time.sleep(0.03)
                _mouse(MOUSEEVENTF_LEFTUP)
            case _:
                log.warning("unknown action name=%r", name)

        time.sleep(action_delay)


# ---------------- Chess coordinate helpers ----------------

_FILES: Final[str] = "abcdefgh"


def _parse_sq(s: str) -> tuple[int, int] | None:
    s = (s or "").strip().lower()
    if len(s) != 2:
        return None
    f, r = s[0], s[1]
    if f not in _FILES or r not in "12345678":
        return None
    file_i = _FILES.index(f)
    rank = int(r)
    row = 8 - rank  # row 0 is top (rank 8)
    return file_i, row


def _square_rect_norm(sq: str) -> tuple[int, int, int, int] | None:
    pr = _parse_sq(sq)
    if not pr:
        return None
    col, row = pr
    x1 = int(col * NORM_MAX / 8)
    x2 = int((col + 1) * NORM_MAX / 8)
    y1 = int(row * NORM_MAX / 8)
    y2 = int((row + 1) * NORM_MAX / 8)
    return _clampi(x1, 0, NORM_MAX), _clampi(y1, 0, NORM_MAX), _clampi(x2, 0, NORM_MAX), _clampi(y2, 0, NORM_MAX)


def _square_center_norm(sq: str) -> tuple[int, int] | None:
    r = _square_rect_norm(sq)
    if not r:
        return None
    x1, y1, x2, y2 = r
    return (x1 + x2) // 2, (y1 + y2) // 2


def _norm_center_to_square(nx: int, ny: int) -> str:
    nx = _clampi(int(nx), 0, NORM_MAX)
    ny = _clampi(int(ny), 0, NORM_MAX)
    col = min(7, (nx * 8) // (NORM_MAX + 1))
    row = min(7, (ny * 8) // (NORM_MAX + 1))
    file_c = _FILES[col]
    rank = 8 - row
    return f"{file_c}{rank}"


def _uci_ok(u: str) -> bool:
    u = (u or "").strip().lower()
    if len(u) not in (4, 5):
        return False
    if not (_parse_sq(u[:2]) and _parse_sq(u[2:4])):
        return False
    if len(u) == 5 and u[4] not in "qrbn":
        return False
    return True


def _drag_action_from_uci(uci: str) -> dict[str, Any] | None:
    uci = (uci or "").strip().lower()
    if not _uci_ok(uci):
        return None
    c1 = _square_center_norm(uci[:2])
    c2 = _square_center_norm(uci[2:4])
    if not (c1 and c2):
        return None
    x1, y1 = c1
    x2, y2 = c2
    return {"name": "drag", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "uci": uci}


def _uci_from_action(a: dict[str, Any]) -> str | None:
    if isinstance(a.get("uci"), str) and _uci_ok(a["uci"]):
        return a["uci"].strip().lower()
    if str(a.get("name", "")).lower() == "drag" and all(k in a for k in ("x1", "y1", "x2", "y2")):
        f = _norm_center_to_square(int(a["x1"]), int(a["y1"]))
        t = _norm_center_to_square(int(a["x2"]), int(a["y2"]))
        return f + t
    return None


# ---------------- Visual heuristics (world markers + guardrails) ----------------


def _b_at(bgra: bytes, w: int, x: int, y: int) -> int:
    i = (y * w + x) * 4
    if i < 0 or i + 3 >= len(bgra):
        return 0
    b, g, r = bgra[i], bgra[i + 1], bgra[i + 2]
    return (int(r) + int(g) + int(b)) // 3


@dataclass(frozen=True)
class _SqStats:
    mean: float
    var: float
    bg: float


def _square_rect_px(w: int, h: int, col: int, row: int) -> tuple[int, int, int, int]:
    x0 = (col * w) // 8
    x1 = ((col + 1) * w) // 8
    y0 = (row * h) // 8
    y1 = ((row + 1) * h) // 8
    return x0, y0, x1, y1


def _square_stats(bgra: bytes, w: int, h: int, col: int, row: int) -> _SqStats | None:
    if not bgra or w <= 0 or h <= 0:
        return None
    x0, y0, x1, y1 = _square_rect_px(w, h, col, row)
    sw, sh = x1 - x0, y1 - y0
    if sw < 8 or sh < 8:
        return None

    # corners approximate background
    corners = [
        (x0 + 2, y0 + 2),
        (x1 - 3, y0 + 2),
        (x0 + 2, y1 - 3),
        (x1 - 3, y1 - 3),
    ]
    bg_vals = [_b_at(bgra, w, _clampi(cx, 0, w - 1), _clampi(cy, 0, h - 1)) for cx, cy in corners]
    bg = sum(bg_vals) / len(bg_vals)

    # central patch
    px0 = x0 + sw * 3 // 10
    px1 = x1 - sw * 3 // 10
    py0 = y0 + sh * 3 // 10
    py1 = y1 - sh * 3 // 10
    if px1 <= px0 or py1 <= py0:
        return None

    step = max(1, min(sw, sh) // 12)
    vals: list[int] = []
    for yy in range(py0, py1, step):
        for xx in range(px0, px1, step):
            vals.append(_b_at(bgra, w, _clampi(xx, 0, w - 1), _clampi(yy, 0, h - 1)))
    if not vals:
        return None

    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return _SqStats(mean=mean, var=var, bg=bg)


def _estimate_piece_color(bgra: bytes, w: int, h: int, sq: str) -> str:
    pr = _parse_sq(sq)
    if not pr:
        return "unknown"
    col, row = pr
    st = _square_stats(bgra, w, h, col, row)
    if not st:
        return "unknown"

    # Heuristic: pieces increase variance.
    # White pieces tend to have highlights (mean above bg); black pieces tend to be darker (mean below bg).
    if st.var < 60 and abs(st.mean - st.bg) < 10:
        return "empty"
    if (st.mean - st.bg) > 18:
        return "white"
    if (st.bg - st.mean) > 18:
        return "black"
    return "unknown"


def compute_world_markers(cur_bgra: bytes, w: int, h: int, prev_bgra: bytes | None = None) -> list[dict[str, Any]]:
    """Compute WORLD bboxes (blue dashed) from pixels.

    W0/W1: squares with largest patch change vs previous frame (if available)
    W2/W3/W4: squares with highest "piece-likeliness" score
    """

    cur_stats: dict[str, _SqStats] = {}
    prev_stats: dict[str, _SqStats] = {}

    for row in range(8):
        for col in range(8):
            sq = f"{_FILES[col]}{8-row}"
            st = _square_stats(cur_bgra, w, h, col, row)
            if st:
                cur_stats[sq] = st
            if prev_bgra:
                stp = _square_stats(prev_bgra, w, h, col, row)
                if stp:
                    prev_stats[sq] = stp

    # piece-likeliness score
    piece_scores: list[tuple[float, str]] = []
    for sq, st in cur_stats.items():
        score = float(st.var) + 1.2 * float(abs(st.mean - st.bg))
        piece_scores.append((score, sq))
    piece_scores.sort(reverse=True)

    # change score
    delta_scores: list[tuple[float, str]] = []
    if prev_bgra and prev_stats:
        for sq, st in cur_stats.items():
            pst = prev_stats.get(sq)
            if not pst:
                continue
            d = abs(st.mean - pst.mean) + 0.08 * abs(st.var - pst.var)
            delta_scores.append((float(d), sq))
        delta_scores.sort(reverse=True)

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def add_marker(label: str, sq: str, kind: str, score: float) -> None:
        if sq in used:
            return
        rect = _square_rect_norm(sq)
        if not rect:
            return
        x1, y1, x2, y2 = rect
        color = _estimate_piece_color(cur_bgra, w, h, sq)
        chosen.append(
            {
                "label": label,
                "kind": kind,
                "sq": sq,
                "score": round(score, 3),
                "color_hint": color,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        used.add(sq)

    # W0/W1 from deltas if available
    for i, (sc, sq) in enumerate(delta_scores[:2]):
        add_marker(f"W{i}", sq, "delta", sc)

    # W2.. from piece scores
    start_idx = len(chosen)
    max_total = int(_cfg("WORLD_MARKERS_MAX", 5) or 5)
    max_total = max(3, min(10, max_total))

    for j, (sc, sq) in enumerate(piece_scores):
        if len(chosen) >= max_total:
            break
        # prefer non-empty/unknown squares
        ch = _estimate_piece_color(cur_bgra, w, h, sq)
        if ch == "empty":
            continue
        add_marker(f"W{start_idx + (len(chosen) - start_idx)}", sq, "piece", sc)

    # If we couldn't compute anything, return empty.
    return chosen


def _square_patch_stats(bgra: bytes, w: int, h: int, sq: str) -> tuple[float, float] | None:
    pr = _parse_sq(sq)
    if not pr:
        return None
    col, row = pr
    st = _square_stats(bgra, w, h, col, row)
    if not st:
        return None
    return st.mean, st.var


def verify_move_effect(pre_bgra: bytes, post_bgra: bytes, w: int, h: int, uci: str) -> bool | None:
    uci = (uci or "").strip().lower()
    if not _uci_ok(uci):
        return None
    f, t = uci[:2], uci[2:4]
    a1 = _square_patch_stats(pre_bgra, w, h, f)
    a2 = _square_patch_stats(post_bgra, w, h, f)
    b1 = _square_patch_stats(pre_bgra, w, h, t)
    b2 = _square_patch_stats(post_bgra, w, h, t)
    if not (a1 and a2 and b1 and b2):
        return None
    (m1, v1), (m2, v2) = a1, a2
    (n1, u1), (n2, u2) = b1, b2
    delta = abs(m2 - m1) + abs(n2 - n1) + 0.08 * (abs(v2 - v1) + abs(u2 - u1))
    thr = float(_cfg("CHESS_VERIFY_THRESHOLD", 35.0))
    return delta >= thr


# ---------------- Parsing (JSON + regex fallback) ----------------


def _json_extract(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # try to pull the outermost object
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


def _ni(v: Any) -> int:
    try:
        return _clampi(int(v), 0, NORM_MAX)
    except Exception:
        return 0


def parse_vlm_payload(raw: str) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (phenomenology, epistemology, self_bboxes, requested_actions)."""

    obj = _json_extract(raw)

    phenomenology = ""
    epistemology: dict[str, Any] = {}
    self_bboxes: list[dict[str, Any]] = []
    requested_actions: list[dict[str, Any]] = []

    if obj is not None:
        phenomenology = str(obj.get("phenomenology") or obj.get("observation") or "")
        ep = obj.get("epistemology")
        if isinstance(ep, dict):
            epistemology = cast(dict[str, Any], ep)

        # bboxes
        bbs = obj.get("bboxes", [])
        if isinstance(bbs, list):
            for i, b in enumerate(bbs):
                if isinstance(b, dict) and all(k in b for k in ("x1", "y1", "x2", "y2")):
                    self_bboxes.append(
                        {
                            "label": str(b.get("label") or f"B{i}"),
                            "x1": _ni(b["x1"]),
                            "y1": _ni(b["y1"]),
                            "x2": _ni(b["x2"]),
                            "y2": _ni(b["y2"]),
                        }
                    )
                elif isinstance(b, list) and len(b) >= 4:
                    self_bboxes.append(
                        {
                            "label": f"B{i}",
                            "x1": _ni(b[0]),
                            "y1": _ni(b[1]),
                            "x2": _ni(b[2]),
                            "y2": _ni(b[3]),
                        }
                    )

        # actions
        acts = obj.get("actions", [])
        if isinstance(acts, list):
            for a in acts:
                if not isinstance(a, dict) or "name" not in a:
                    continue
                name = str(a.get("name", "")).lower()
                if name == "chess_move":
                    uci = str(a.get("uci", "")).strip().lower()
                    if _uci_ok(uci):
                        requested_actions.append({"name": "chess_move", "uci": uci})
                elif name in ("drag", "move", "click", "right_click", "double_click") and "x1" in a and "y1" in a:
                    entry: dict[str, Any] = {"name": name, "x1": _ni(a.get("x1")), "y1": _ni(a.get("y1"))}
                    if "x2" in a and "y2" in a:
                        entry["x2"] = _ni(a.get("x2"))
                        entry["y2"] = _ni(a.get("y2"))
                    uci = _uci_from_action(entry)
                    if uci:
                        entry["uci"] = uci
                    requested_actions.append(entry)

    # Regex fallback (orange actions, blue bboxes, uci)
    if obj is None or (not requested_actions and not self_bboxes):
        rx_a = str(_cfg("PARSE_REGEX_ORANGE_ACTIONS", "") or "")
        rx_b = str(_cfg("PARSE_REGEX_BLUE_BBOXES", "") or "")
        rx_u = str(_cfg("PARSE_REGEX_UCI", "") or "")

        if not phenomenology:
            phenomenology = raw.strip()

        # bboxes
        if rx_b:
            try:
                patb = re.compile(rx_b)
                for i, m in enumerate(patb.finditer(raw)):
                    g = [x for x in m.groups() if x is not None]
                    if len(g) >= 4:
                        x1, y1, x2, y2 = map(_ni, g[:4])
                        self_bboxes.append({"label": f"B{i}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                        if len(self_bboxes) >= 3:
                            break
            except Exception:
                pass

        # actions
        if rx_a:
            try:
                pata = re.compile(rx_a)
                for m in pata.finditer(raw):
                    name = str(m.group(1)).lower()
                    x1, y1 = _ni(m.group(2)), _ni(m.group(3))
                    x2 = m.group(4)
                    y2 = m.group(5)
                    entry: dict[str, Any] = {"name": name, "x1": x1, "y1": y1}
                    if x2 is not None and y2 is not None:
                        entry["x2"] = _ni(x2)
                        entry["y2"] = _ni(y2)
                    uci = _uci_from_action(entry)
                    if uci:
                        entry["uci"] = uci
                    requested_actions.append(entry)
            except Exception:
                pass

        # uci
        if not requested_actions and rx_u:
            try:
                patu = re.compile(rx_u)
                m = patu.search(raw)
                if m:
                    uci = m.group(1).strip().lower()
                    if _uci_ok(uci):
                        requested_actions.append({"name": "chess_move", "uci": uci})
            except Exception:
                pass

    # clamp self bboxes to exactly 3 if more
    if len(self_bboxes) > 3:
        self_bboxes = self_bboxes[:3]

    log.info(
        "parse_vlm_payload phen_len=%d self_bboxes=%d requested_actions=%d",
        len(phenomenology),
        len(self_bboxes),
        len(requested_actions),
    )
    return phenomenology, epistemology, self_bboxes, requested_actions


# ---------------- Saving ----------------


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    except Exception as e:
        log.warning("append jsonl failed: %s", e)


def save_turn_data(
    run_dir: Path,
    turn: int,
    payload: dict[str, Any],
    raw_b64: str,
) -> None:
    layout = str(_cfg("LOG_LAYOUT", "turn_dirs")).lower()

    if layout == "flat":
        raw_name = f"turn_{turn:04d}_raw.png"
        if raw_b64:
            try:
                (run_dir / raw_name).write_bytes(base64.b64decode(raw_b64))
            except Exception as e:
                log.warning("save raw png failed: %s", e)
        _append_jsonl(run_dir / "turns.jsonl", {"turn": turn, "stage": "raw", **payload, "raw_png": raw_name})
        return

    td = run_dir / f"turn_{turn:04d}"
    td.mkdir(exist_ok=True)
    (td / "turn_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if raw_b64:
        try:
            (td / "screenshot_raw.png").write_bytes(base64.b64decode(raw_b64))
        except Exception as e:
            log.warning("save raw png failed: %s", e)


def save_annotated(run_dir: Path, turn: int, annotated_b64: str) -> None:
    if not annotated_b64:
        return
    layout = str(_cfg("LOG_LAYOUT", "turn_dirs")).lower()
    if layout == "flat":
        ann_name = f"turn_{turn:04d}_annotated.png"
        try:
            (run_dir / ann_name).write_bytes(base64.b64decode(annotated_b64))
        except Exception as e:
            log.warning("save annotated png failed: %s", e)
        _append_jsonl(run_dir / "turns.jsonl", {"turn": turn, "stage": "annotated", "annotated_png": ann_name})
        return
    td = run_dir / f"turn_{turn:04d}"
    td.mkdir(exist_ok=True)
    try:
        (td / "screenshot_annotated.png").write_bytes(base64.b64decode(annotated_b64))
    except Exception as e:
        log.warning("save annotated png failed: %s", e)


def _append_story(run_dir: Path, text: str) -> None:
    try:
        (run_dir / "story.md").open("a", encoding="utf-8").write(text)
    except Exception:
        pass


# ---------------- VLM call ----------------


def call_vlm(user_text: str, image_b64: str) -> tuple[str, dict[str, Any], str | None]:
    url = str(_cfg("API_URL", ""))
    u = urllib.parse.urlparse(url)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    path = u.path or "/v1/chat/completions"
    t = float(_cfg("VLM_HTTP_TIMEOUT_SECONDS", 0) or 0)
    timeout = None if t <= 0 else t

    system_prompt = str(_cfg("SYSTEM_PROMPT", ""))
    payload = {
        "model": str(_cfg("MODEL", "")),
        "temperature": float(_cfg("TEMPERATURE", 0.7)),
        "top_p": float(_cfg("TOP_P", 0.9)),
        "max_tokens": int(_cfg("MAX_TOKENS", 1200)),
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    log.info("vlm POST %s:%d%s text_len=%d img_len=%d", host, port, path, len(user_text), len(image_b64))

    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        if resp.status < 200 or resp.status >= 300:
            return "", {}, f"HTTP {resp.status}"
        obj = json.loads(data.decode("utf-8", "replace"))
        text = cast(str, obj["choices"][0]["message"]["content"])
        usage = cast(dict[str, Any], obj.get("usage", {}) or {})
        return text, usage, None
    except Exception as e:
        log.error("vlm error: %s", e)
        return "", {}, str(e)


# ---------------- Engine loop ----------------


def _world_summary(world_bboxes: list[dict[str, Any]]) -> str:
    if not world_bboxes:
        return "none"
    parts: list[str] = []
    for wbb in world_bboxes[:8]:
        lbl = str(wbb.get("label", "W?"))
        sq = str(wbb.get("sq", "?"))
        kind = str(wbb.get("kind", ""))
        ch = str(wbb.get("color_hint", ""))
        sc = wbb.get("score")
        parts.append(f"{lbl}~{sq}({kind},{ch},{sc})")
    return ", ".join(parts)


def _self_summary(self_bboxes: list[dict[str, Any]]) -> str:
    if not self_bboxes:
        return "none"
    parts: list[str] = []
    for bb in self_bboxes[:6]:
        lbl = str(bb.get("label") or "B?")
        cx = (int(bb.get("x1", 0)) + int(bb.get("x2", 0))) // 2
        cy = (int(bb.get("y1", 0)) + int(bb.get("y2", 0))) // 2
        parts.append(f"{lbl}~{_norm_center_to_square(cx, cy)}")
    return ", ".join(parts)


async def engine_loop(run_dir: Path) -> None:
    S.run_dir = run_dir

    boot_enabled = bool(_cfg("BOOT_ENABLED", True))
    boot_text = str(_cfg("BOOT_VLM_OUTPUT", ""))

    set_phase("boot" if boot_enabled else "running")
    if boot_enabled and boot_text.strip():
        log.info("engine: injecting boot VLM text len=%d", len(boot_text))
        S.next_vlm_json = boot_text
        S.next_event.set()
    else:
        log.info("engine: waiting for first /inject")
        set_phase("waiting_inject")

    # initial capture so UI has a frame
    raw_b64, w, h, bgra = await asyncio.get_event_loop().run_in_executor(None, capture_frame, True)
    async with S.lock:
        S.raw_b64, S.raw_w, S.raw_h, S.raw_bgra = raw_b64, w, h, bgra
        S.world_bboxes = compute_world_markers(bgra, w, h, None) if bgra else []

    while not STOP.is_set():
        try:
            await asyncio.wait_for(S.next_event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        async with S.lock:
            vlm_raw = S.next_vlm_json or ""
            S.next_vlm_json = None
            S.next_event.clear()

        if not vlm_raw.strip():
            continue

        S.turn += 1
        turn = S.turn
        log.info("engine: === TURN %d ===", turn)

        set_phase("running")
        phenomenology, epistemology, self_bboxes, requested_actions = parse_vlm_payload(vlm_raw)

        # Snapshot the pre-action frame (what the model was looking at).
        async with S.lock:
            pre_bgra = S.raw_bgra
            pre_w = S.raw_w
            pre_h = S.raw_h
            pre_world = S.world_bboxes

        # Convert requested actions into executable actions.
        converted: list[dict[str, Any]] = []
        for a in requested_actions:
            name = str(a.get("name", "")).lower()
            if name == "chess_move":
                da = _drag_action_from_uci(str(a.get("uci", "")))
                if da:
                    converted.append(da)
            elif name == "drag":
                entry = {"name": "drag", "x1": _ni(a.get("x1")), "y1": _ni(a.get("y1")), "x2": _ni(a.get("x2")), "y2": _ni(a.get("y2"))}
                uci = _uci_from_action(entry)
                if uci:
                    entry["uci"] = uci
                converted.append(entry)
            elif name in ("click", "move", "right_click", "double_click"):
                converted.append({"name": name, "x1": _ni(a.get("x1")), "y1": _ni(a.get("y1"))})

        requested_uci = [u for a in converted if (u := _uci_from_action(a))]
        requested_desc = ", ".join(requested_uci) if requested_uci else "none"

        # Guardrail: optionally block moves that appear to start on a black piece.
        blocked_notes: list[str] = []
        exec_actions = list(converted)
        if bool(_cfg("CHESS_GUARD_BLOCK_OPPONENT", False)) and pre_bgra and pre_w > 0 and pre_h > 0:
            filtered: list[dict[str, Any]] = []
            for a in exec_actions:
                if str(a.get("name", "")).lower() == "drag":
                    uci = _uci_from_action(a)
                    if uci:
                        from_sq = uci[:2]
                        color = _estimate_piece_color(pre_bgra, pre_w, pre_h, from_sq)
                        if color == "black":
                            blocked_notes.append(f"blocked {uci}: black piece at {from_sq}")
                            continue
                filtered.append(a)
            exec_actions = filtered

        executed_uci = [u for a in exec_actions if (u := _uci_from_action(a))]
        executed_desc = ", ".join(executed_uci) if executed_uci else "none"

        # Self/world consistency note
        if requested_uci and self_bboxes:
            from_sq = requested_uci[0][:2]
            self_sqs = {_norm_center_to_square((int(bb["x1"]) + int(bb["x2"])) // 2, (int(bb["y1"]) + int(bb["y2"])) // 2) for bb in self_bboxes}
            if from_sq not in self_sqs:
                blocked_notes.append(f"note: chosen move starts at {from_sq} but self boxes suggest {sorted(self_sqs)}")

        async with S.lock:
            S.vlm_json = vlm_raw
            S.phenomenology = phenomenology
            S.epistemology = epistemology
            S.self_bboxes = self_bboxes
            S.requested_actions = requested_actions
            S.executed_actions = exec_actions
            S.msg_id += 1

        set_phase("executing")
        await asyncio.get_event_loop().run_in_executor(None, execute_actions, exec_actions)

        set_phase("capturing")
        raw_b64, w, h, bgra = await asyncio.get_event_loop().run_in_executor(None, capture_frame, True)
        if not raw_b64:
            set_phase("error", "capture failed")
            continue

        # Compute WORLD markers on the NEW frame.
        world_bboxes = compute_world_markers(bgra, w, h, pre_bgra if pre_bgra else None) if bgra else []

        async with S.lock:
            S.raw_b64, S.raw_w, S.raw_h, S.raw_bgra = raw_b64, w, h, bgra
            S.world_bboxes = world_bboxes

        # Persist
        payload = {
            "turn": turn,
            "phenomenology": phenomenology,
            "epistemology": epistemology,
            "self_bboxes": self_bboxes,
            "world_bboxes": world_bboxes,
            "requested_actions": requested_actions,
            "executed_actions": exec_actions,
        }
        await asyncio.get_event_loop().run_in_executor(None, save_turn_data, run_dir, turn, payload, raw_b64)

        # Request browser annotation for VLM (world-only overlays)
        async with S.lock:
            S.pending_seq = turn
            S.annotated_seq = -1
            S.annotated_b64 = ""
            S.annotated_event.clear()

        set_phase("waiting_annotated")
        log.info("engine: waiting for browser annotated seq=%d", turn)

        annotated_b64 = ""
        ann_timeout = float(_cfg("ANNOTATED_TIMEOUT_SECONDS", 0) or 0)
        t0 = time.time()
        while not STOP.is_set():
            try:
                await asyncio.wait_for(S.annotated_event.wait(), timeout=0.5)
                break
            except asyncio.TimeoutError:
                if ann_timeout > 0 and (time.time() - t0) >= ann_timeout:
                    log.warning("annotated timeout (%.2fs); continuing", ann_timeout)
                    break
                continue

        if STOP.is_set():
            break

        async with S.lock:
            annotated_b64 = S.annotated_b64

        if annotated_b64:
            await asyncio.get_event_loop().run_in_executor(None, save_annotated, run_dir, turn, annotated_b64)

        # Verify the effect of the executed move.
        verified_line = "unknown"
        if bool(_cfg("CHESS_VERIFY_MOVE", True)) and pre_bgra and bgra and pre_w == w and pre_h == h and executed_uci:
            v = verify_move_effect(pre_bgra, bgra, w, h, executed_uci[0])
            if v is False:
                verified_line = "uncertain (from/to squares look unchanged)"
            elif v is True:
                verified_line = "changed (from/to squares differ)"

        # Build grounded next-turn context.
        context = (
            f"TURN {turn} AUDIT\n"
            f"Requested: {requested_desc}\n"
            f"Executed: {executed_desc}\n"
            f"Verified: {verified_line}\n"
        )
        if blocked_notes:
            context += "Notes: " + "; ".join(blocked_notes) + "\n"
        context += (
            f"WORLD markers (controller, algorithmic): {_world_summary(world_bboxes)}\n"
            f"Prior WORLD markers: {_world_summary(pre_world)}\n"
            f"Prior SELF boxes (model, not guaranteed): {_self_summary(self_bboxes)}\n\n"
            "On the NEW image: run Dual Epistemology Mode.\n"
            "- Produce rich phenomenology (up to ~600 tokens).\n"
            "- Produce epistemology object (beliefs/evidence/options/decision).\n"
            "- Mark EXACTLY 3 SELF boxes labeled B0/B1/B2.\n"
            "- Choose ONE bottom-side move with actions=[{name:'chess_move',uci:'....'}] or actions=[].\n"
        )

        # Story
        story = (
            f"\n# Turn {turn}\n"
            f"## Audit\n"
            f"- Requested: {requested_desc}\n"
            f"- Executed: {executed_desc}\n"
            f"- Verified: {verified_line}\n"
        )
        if blocked_notes:
            story += "- Notes: " + "; ".join(blocked_notes) + "\n"
        story += (
            f"- World markers: {_world_summary(world_bboxes)}\n"
            f"- Self boxes: {_self_summary(self_bboxes)}\n"
            "\n## Phenomenology (model)\n"
            + (phenomenology.strip() or "(empty)")
            + "\n"
        )
        if epistemology:
            story += "\n## Epistemology (model)\n" + json.dumps(epistemology, ensure_ascii=False, indent=2) + "\n"
        _append_story(run_dir, story)

        # Call VLM using selected image source.
        set_phase("calling_vlm")
        img_source = str(_cfg("VLM_IMAGE_SOURCE", "annotated")).lower()
        img_b64 = raw_b64 if img_source == "raw" else (annotated_b64 or raw_b64)

        new_vlm_text, usage, err = await asyncio.get_event_loop().run_in_executor(None, call_vlm, context, img_b64)
        if err:
            log.error("vlm error turn=%d: %s", turn, err)
            S.error = err
            set_phase("vlm_error")
            continue

        log.info("vlm ok turn=%d response_len=%d usage=%s", turn, len(new_vlm_text), usage)

        async with S.lock:
            S.next_vlm_json = new_vlm_text
            S.next_event.set()

        set_phase("running")


# ---------------- Minimal async HTTP server ----------------


class AsyncHTTPServer:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_connection, self._host, self._port)
        log.info("server http://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._process(reader, writer)
        except (ConnectionResetError, ConnectionAbortedError, asyncio.IncompleteReadError):
            pass
        except Exception as e:
            if isinstance(e, OSError) and getattr(e, "winerror", None) in (10053, 10054):
                pass
            else:
                log.warning("connection error: %s", e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _process(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        raw_line = await asyncio.wait_for(reader.readline(), timeout=30)
        if not raw_line:
            return
        request_line = raw_line.decode("utf-8", "replace").strip()
        parts = request_line.split(" ")
        if len(parts) < 2:
            return
        method, full_path = parts[0], parts[1]
        path = full_path.split("?", 1)[0]
        headers: dict[str, str] = {}
        while True:
            hl = await asyncio.wait_for(reader.readline(), timeout=10)
            if not hl or hl in (b"\r\n", b"\n"):
                break
            decoded = hl.decode("utf-8", "replace").strip()
            if ":" in decoded:
                k, v = decoded.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        body = b""
        if cl := int(headers.get("content-length", "0")):
            body = await asyncio.wait_for(reader.readexactly(cl), timeout=60)

        match method:
            case "GET":
                await self._do_get(path, writer)
            case "POST":
                await self._do_post(path, body, writer)
            case "OPTIONS":
                await self._send_json(writer, {}, 200)
            case _:
                await self._send_error(writer, 405)

    async def _do_get(self, path: str, writer: asyncio.StreamWriter) -> None:
        match path:
            case "/" | "/index.html":
                await self._send_raw(writer, 200, "text/html; charset=utf-8", PANEL_HTML.read_bytes())
            case "/config":
                await self._send_json(
                    writer,
                    {
                        "ui": _cfg("UI_CONFIG", {}),
                        "capture_width": int(_cfg("CAPTURE_WIDTH", 0)),
                        "capture_height": int(_cfg("CAPTURE_HEIGHT", 0)),
                        "parse_regex_orange": str(_cfg("PARSE_REGEX_ORANGE_ACTIONS", "") or ""),
                        "parse_regex_blue": str(_cfg("PARSE_REGEX_BLUE_BBOXES", "") or ""),
                        "parse_regex_uci": str(_cfg("PARSE_REGEX_UCI", "") or ""),
                    },
                )
            case "/state":
                async with S.lock:
                    await self._send_json(
                        writer,
                        {
                            "phase": S.phase,
                            "error": S.error,
                            "turn": S.turn,
                            "msg_id": S.msg_id,
                            "pending_seq": S.pending_seq,
                            "annotated_seq": S.annotated_seq,
                            "raw_b64": S.raw_b64,
                            "raw_w": S.raw_w,
                            "raw_h": S.raw_h,
                            "world_bboxes": S.world_bboxes,
                            "self_bboxes": S.self_bboxes,
                            "requested_actions": S.requested_actions,
                            "executed_actions": S.executed_actions,
                            "phenomenology": S.phenomenology,
                            "epistemology": S.epistemology,
                            "vlm_json": S.vlm_json,
                        },
                    )
            case _:
                await self._send_error(writer, 404)

    async def _do_post(self, path: str, body: bytes, writer: asyncio.StreamWriter) -> None:
        match path:
            case "/annotated":
                try:
                    obj = json.loads(body.decode("utf-8"))
                except Exception:
                    await self._send_json(writer, {"ok": False, "err": "invalid json"}, 400)
                    return

                seq = obj.get("seq")
                img = obj.get("image_b64", "")

                async with S.lock:
                    expected = S.pending_seq

                if seq != expected:
                    await self._send_json(
                        writer,
                        {"ok": False, "err": f"seq mismatch: got {seq} expected {expected}"},
                        409,
                    )
                    return

                if not isinstance(img, str) or len(img) < 100:
                    await self._send_json(writer, {"ok": False, "err": "image_b64 too short"}, 400)
                    return

                async with S.lock:
                    S.annotated_b64 = img
                    S.annotated_seq = int(seq)
                    S.annotated_event.set()

                await self._send_json(writer, {"ok": True, "seq": seq})

            case "/inject":
                try:
                    obj = json.loads(body.decode("utf-8"))
                except Exception:
                    await self._send_json(writer, {"ok": False, "err": "invalid json"}, 400)
                    return

                text = obj.get("vlm_text", "")
                if not isinstance(text, str) or not text.strip():
                    await self._send_json(writer, {"ok": False, "err": "vlm_text empty"}, 400)
                    return

                async with S.lock:
                    S.next_vlm_json = text
                    S.next_event.set()

                await self._send_json(writer, {"ok": True})

            case _:
                await self._send_error(writer, 404)

    async def _send_raw(self, writer: asyncio.StreamWriter, code: int, content_type: str, data: bytes) -> None:
        status = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
        }.get(code, "OK")
        hdr = (
            f"HTTP/1.1 {code} {status}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(data)}\r\n"
            f"Cache-Control: no-cache\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            f"Access-Control-Allow-Headers: Content-Type\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(hdr.encode("utf-8") + data)
        await writer.drain()

    async def _send_json(self, writer: asyncio.StreamWriter, obj: Any, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        await self._send_raw(writer, code, "application/json", data)

    async def _send_error(self, writer: asyncio.StreamWriter, code: int) -> None:
        await self._send_json(writer, {"error": code}, code)


async def async_main() -> None:
    global S, STOP
    S = EngineState()
    STOP = asyncio.Event()

    run_dir = make_run_dir()
    setup_logging(run_dir)

    log.info("Franz (Dual Epistemology) starting run_dir=%s", run_dir)
    log.info("panel=%s config=%s", PANEL_HTML, CONFIG_PATH)

    server = AsyncHTTPServer(HOST, PORT)
    await server.start()

    loop = asyncio.get_event_loop()
    if hasattr(loop, "add_signal_handler") and os.name != "nt":
        loop.add_signal_handler(signal.SIGINT, lambda: STOP.set())

    try:
        webbrowser.open(f"http://{HOST}:{PORT}")
    except Exception as e:
        log.warning("webbrowser.open failed: %s", e)

    engine_task = asyncio.create_task(engine_loop(run_dir))

    try:
        await STOP.wait()
    except KeyboardInterrupt:
        STOP.set()

    engine_task.cancel()
    await server.stop()
    log.info("Franz stopped")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
