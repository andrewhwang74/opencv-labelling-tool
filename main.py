import cv2
import sys
import os
import ctypes
import json
import datetime
import numpy as np
from ultralytics import YOLO

WINDOW_NAME     = "Video Player"
SIDEBAR_W       = 220            # left sidebar width (pixels)
TRACKS_PER_PAGE = 7              # saved-track rows visible at once

# Colours assigned to saved tracks in the preview list (cycles)
PREVIEW_COLORS = [
    (255, 140,   0),   # orange
    (  0, 230, 118),   # green
    (220,   0, 180),   # magenta
    (160,  80, 255),   # purple
    (255, 220,   0),   # yellow
    (  0, 180, 255),   # sky
    (255,  60,  60),   # red
    (  0, 255, 200),   # teal
]

# Colours cycling for named timeline markers
MARKER_COLORS = [
    (  0, 255, 255),   # yellow
    (255, 180,   0),   # blue
    (  0, 255, 160),   # green
    (255,  80, 200),   # pink
    (128, 255,   0),   # lime
    (  0, 200, 255),   # orange
    (200, 200, 255),   # lavender
]

# --- Annotation config: edit these to customise modes and states ---

# Maps YOLO class name → mode category shown in popup
YOLO_CLASS_MAP = {
    'person':     'person',
    'car':        'vehicle',
    'truck':      'vehicle',
    'bus':        'vehicle',
}
DEFAULT_MODE = 'person'   # fallback when YOLO class not in map

# Per-mode state categories
# Each entry: (category_name, [options], scope)
# scope = 'per_frame'  → stored per frame; popup accessible on every paused frame
# scope = 'per_track'  → set once, applied to all frames; propagated retroactively
# scope defaults to 'per_frame' if omitted.
TRACK_STATES = {
    'person': [
        ('movement',              ['walking', 'waiting', 'rushing', 're-routing'],              'per_frame'),
        ('crosswalk position',    ['outside crosswalk', 'inside crosswalk'],                    'per_frame'),
        ('distraction',           ['distacted', 'not distracted'],                              'per_frame'),
        ('interaction',           ['no interaction', 'gesture', 'verbal', 'eye-contact'],       'per_frame'),
        ('wait time',             ['just arrived', 'short wait', 'long wait'],                  'per_track'),
        ('start position',        ['near', 'far'],                                              'per_track'),
        ('group size',            ['alone', 'with group'],                                      'per_track'),
        ('carrying object',       ['no object', 'stroller', 'cart', 'other'],                   'per_track'),
    ],
    'vehicle': [
        ('movement',              ['turning', 'waiting', 'straight'],                                           'per_frame'),
        ('position',              ['outside crosswalk', 'inside crosswalk'],                                    'per_frame'),
        ('interaction',           ['no interaction', 'honk', 'gap-find', 'gesture', 'verbal', 'eye-contact'],   'per_frame'),
        ('type',                  ['suv', 'truck', 'bus', 'sedan', 'van', 'pickup'],                            'per_track'),
        ('wait time',             ['just arrived', 'short wait', 'long wait'],                                  'per_track'),
    ],
}

# Popup appearance
POPUP_W            = 200    # popup panel width (canvas pixels)
POPUP_ROW_H        = 26     # height of each row (pixels)
POPUP_FONT         = 0.38   # font scale
POPUP_MODE_BG      = (0,   160, 180)   # mode row background
POPUP_MODE_COLOR   = (10,  10,  10)
POPUP_FRAME_BG     = (55,  55,  55)    # per_frame state row background
POPUP_FRAME_COLOR  = (210, 210, 210)
POPUP_TRACK_BG     = (70,  50,  10)    # per_track state row background (amber)
POPUP_TRACK_COLOR  = (240, 200, 100)

# waitKeyEx key codes
KEY_LEFT  = (2424832, 65361)   # Windows, Linux/macOS
KEY_RIGHT = (2555904, 65363)
KEY_ENTER = (13, 10)   # CR / LF
KEY_BKSP  = (8, 127)   # Backspace / Delete

VK_ALT   = 0x12   # GetAsyncKeyState virtual-key code for Alt
VK_SHIFT = 0x10   # GetAsyncKeyState virtual-key code for Shift

SCRUB_STEP = 1

TRACK_COLOR   = (0, 220, 255)   # cyan   – live tracked object
HISTORY_COLOR = (80, 150, 255)  # blue   – historical (seeked-to) box
DIM_COLOR     = (160, 160, 160) # grey   – "showing all" fallback detections

# Shared mutable state (written by mouse callback, read by main loop)
_state = {
    "seek_frame":    None,   # timeline click    → seek to frame
    "detect_click":  None,   # video area click  → (canvas_x, canvas_y)
    "sidebar_click": None,   # sidebar click     → (canvas_x, canvas_y)
    "drag_start":    None,   # LBUTTONDOWN in video area → (cx, cy) canvas coords
    "drag_current":  None,   # MOUSEMOVE while dragging  → (cx, cy) canvas coords
    "drag_end":      None,   # LBUTTONUP after drag      → (cx1,cy1,cx2,cy2)
}

_DRAG_THRESHOLD = 8   # pixels; smaller motion treated as click, not a draw

# Layout computed each frame in build_canvas so canvas_to_frame() can back-project
_layout: dict = {}


def get_screen_size():
    """Return (width, height) of the primary monitor in physical pixels."""
    try:
        user32 = ctypes.windll.user32
        # Tell Windows this process is DPI-aware so GetSystemMetrics returns
        # physical (not scaled) pixel counts, matching what OpenCV sees.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        try:
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            return 1920, 1080


def on_mouse(event, x, y, flags, param):
    """Route mouse events: timeline seek, sidebar click, video click or draw-drag."""
    screen_w, screen_h, total_frames = param
    timeline_top = int(screen_h * 0.90)

    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= timeline_top:
            if total_frames > 0:
                _state["seek_frame"] = int(max(0.0, min(1.0, x / screen_w)) * total_frames)
        elif x < SIDEBAR_W:
            _state["sidebar_click"] = (x, y)
        else:
            # Video area: start drag (may become a click if motion < threshold)
            _state["drag_start"]   = (x, y)
            _state["drag_current"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if _state["drag_start"] is not None:
            _state["drag_current"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if _state["drag_start"] is not None:
            ox, oy = _state["drag_start"]
            dx, dy = abs(x - ox), abs(y - oy)
            if dx > _DRAG_THRESHOLD or dy > _DRAG_THRESHOLD:
                _state["drag_end"] = (ox, oy, x, y)
            else:
                _state["detect_click"] = (ox, oy)
            _state["drag_start"]   = None
            _state["drag_current"] = None


def canvas_to_frame(cx, cy):
    """Map a canvas pixel (cx, cy) to the corresponding video frame pixel.
    Returns (fx, fy) or None if the point is outside the visible video rect.
    """
    if not _layout:
        return None
    x0, y0 = _layout["x_off"], _layout["y_off"]
    nw, nh  = _layout["new_w"],  _layout["new_h"]
    scale   = _layout["scale"]
    rel_x, rel_y = cx - x0, cy - y0
    if 0 <= rel_x < nw and 0 <= rel_y < nh:
        return int(rel_x / scale), int(rel_y / scale)
    return None


def build_frame_lookup(track_dict):
    """Build frame_num → (x1,y1,x2,y2,label,mode,states) index for a saved track.
    Merges root-level per_track states into every frame's states.
    """
    per_track_root = track_dict.get("track_states", {})
    lu = {}
    for f in track_dict.get("frames", []):
        merged = dict(per_track_root)
        merged.update(f.get("states", {}))
        lu[f["frame_num"]] = (
            f["box"][0], f["box"][1], f["box"][2], f["box"][3],
            f["label"],
            f.get("mode",  DEFAULT_MODE),
            merged,
        )
    return lu


def _state_scope(mode, cat):
    """Return 'per_track' or 'per_frame' for a category in the given mode."""
    for entry in TRACK_STATES.get(mode, []):
        if entry[0] == cat:
            return entry[2] if len(entry) > 2 else 'per_frame'
    return 'per_frame'


def _ordered_state_cats(mode):
    """Return TRACK_STATES entries for mode, per_track first then per_frame.
    Order within each group is preserved from the TRACK_STATES definition.
    """
    cats = TRACK_STATES.get(mode, [])
    per_track = [e for e in cats if (len(e) > 2 and e[2] == 'per_track')]
    per_frame = [e for e in cats if not (len(e) > 2 and e[2] == 'per_track')]
    return per_track + per_frame


def _default_track_states(mode):
    """Return {cat_name: first_option} for the given mode, per_track first."""
    return {e[0]: e[1][0] for e in _ordered_state_cats(mode)}


def draw_popup(canvas, anchor_x, anchor_y, track_mode, track_states, screen_w, screen_h):
    """Compact popup: one toggle row per annotation item (no headers/lists).

    Mode row   – shows current mode; click cycles to next mode.
    State rows – one per category for current mode; shows active value;
                 click cycles to next option in that category.

    Returns zones: [{"rect", "action": "mode"|"state", "cat", "next_val"}]
    """
    all_modes  = list(TRACK_STATES.keys())
    state_cats = _ordered_state_cats(track_mode)
    num_rows   = 1 + len(state_cats)
    total_h    = num_rows * POPUP_ROW_H + (num_rows - 1) * 2

    px = anchor_x
    py = anchor_y - total_h
    if py < 4:
        py = anchor_y + 2
    if px + POPUP_W > screen_w - 4:
        px = screen_w - POPUP_W - 4
    px = max(SIDEBAR_W + 4, px)

    zones = []
    cur_y = py

    # --- Mode row ---
    m_idx     = all_modes.index(track_mode) if track_mode in all_modes else 0
    next_mode = all_modes[(m_idx + 1) % len(all_modes)]
    rx1, ry1  = px, cur_y
    rx2, ry2  = px + POPUP_W, cur_y + POPUP_ROW_H
    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), POPUP_MODE_BG, -1)
    cv2.putText(canvas, track_mode, (rx1 + 8, ry2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, POPUP_FONT, POPUP_MODE_COLOR, 1, cv2.LINE_AA)
    zones.append({"rect": (rx1, ry1, rx2, ry2), "action": "mode",
                  "cat": None, "next_val": next_mode})
    cur_y += POPUP_ROW_H + 2

    # --- One toggle row per state category ---
    for entry in state_cats:
        cat_name = entry[0]
        opts     = entry[1]
        scope    = entry[2] if len(entry) > 2 else 'per_frame'
        is_track = (scope == 'per_track')
        cur_val  = track_states.get(cat_name, opts[0])
        c_idx    = opts.index(cur_val) if cur_val in opts else 0
        next_val = opts[(c_idx + 1) % len(opts)]
        ox1, oy1 = px, cur_y
        ox2, oy2 = px + POPUP_W, cur_y + POPUP_ROW_H
        bg_col   = POPUP_TRACK_BG    if is_track else POPUP_FRAME_BG
        txt_col  = POPUP_TRACK_COLOR if is_track else POPUP_FRAME_COLOR
        cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), bg_col, -1)
        badge = ' [T]' if is_track else ' [F]'
        cv2.putText(canvas, cur_val + badge, (ox1 + 8, oy2 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, POPUP_FONT, txt_col, 1, cv2.LINE_AA)
        zones.append({"rect": (ox1, oy1, ox2, oy2), "action": "state",
                      "cat": cat_name, "next_val": next_val, "scope": scope})
        cur_y += POPUP_ROW_H + 2

    # Outer border
    cv2.rectangle(canvas, (px - 1, py - 1),
                  (px + POPUP_W + 1, cur_y), (120, 120, 120), 1)
    return zones


def build_canvas(frame, screen_w, screen_h, current_frame, total_frames,
                 fps, paused, boxes, preview_boxes=None, canvas_meta=None):
    """
    Full-screen canvas layout:
      [SIDEBAR_W px wide]  |  [video area – top 90%]
      [full-width timeline –  bottom 10%]
    boxes         – current-track boxes: (x1,y1,x2,y2,label,color).
    preview_boxes – saved-track boxes:   (x1,y1,x2,y2,label,color,ann_str).
    canvas_meta   – dict with display flags.
    """
    if canvas_meta is None:
        canvas_meta = {}
    if preview_boxes is None:
        preview_boxes = []
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    video_area_h = int(screen_h * 0.90)
    video_area_w = screen_w - SIDEBAR_W
    timeline_y   = video_area_h

    # ------------------------------------------------------------------ sidebar
    cv2.rectangle(canvas, (0, 0), (SIDEBAR_W, video_area_h), (18, 18, 18), -1)
    cv2.line(canvas, (SIDEBAR_W, 0), (SIDEBAR_W, video_area_h), (55, 55, 55), 1)

    PAD = 10
    sy  = 14     # current y cursor in sidebar (= top of next text region)

    # ---- Sidebar font scales (direct, no multiplier) ----
    F_HDR = 0.68    # section headers: TRACKING / HISTORY / MARKERS
    F_LBL = 0.58    # labels:          track id, mode, video filename
    F_ROW = 0.52    # list rows:       track entries, counts
    F_BTN = 0.50    # buttons:         pagination, show/hide
    F_HNT = 0.46    # hints:           key shortcuts, dim text

    def _lh(scale, gap=4):
        """Pixel advance after one text line — accounts for actual glyph height."""
        (_, h), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        return h + base + gap

    def _cy(y1, row_h, scale):
        """Baseline y to vertically centre glyphs within a row [y1, y1+row_h]."""
        (_, h), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        return y1 + (row_h + h) // 2

    def sb_line(text, scale=F_ROW, color=(140, 140, 140), bold=False, gap=4, x=PAD):
        """Draw text at current sy position and advance sy by the full line height."""
        nonlocal sy
        (_, h), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(canvas, text, (x, sy + h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1, cv2.LINE_AA)
        sy += h + base + gap

    def sb_text(text, y_baseline, scale=F_ROW, color=(140, 140, 140), bold=False):
        """Draw text at an explicit baseline y (used inside fixed-height rows)."""
        cv2.putText(canvas, text, (PAD, y_baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1, cv2.LINE_AA)

    def sb_divider(y):
        cv2.line(canvas, (PAD, y), (SIDEBAR_W - PAD, y), (45, 45, 45), 1)

    # Row / button heights derived from font metrics
    DELETE_W  = 22
    ROW_H     = max(26, _lh(F_ROW, gap=0) + 12)   # track-list rows
    BTN_H     = max(24, _lh(F_BTN, gap=0) + 12)   # pagination / show-hide
    MRK_ROW_H = max(22, _lh(F_ROW, gap=0) + 10)   # marker-list rows

    # ---- Compact current-tracking state ----
    tracking   = canvas_meta.get("tracking", False)
    history    = canvas_meta.get("history",  False)
    t_label    = canvas_meta.get("tracked_label", "")
    annotation = canvas_meta.get("annotation")
    ann_color  = TRACK_COLOR if tracking else HISTORY_COLOR
    track_frames = canvas_meta.get("track_frames", 0)

    if tracking:
        sb_line("TRACKING", F_HDR, TRACK_COLOR, bold=True, gap=4)
        if t_label:
            sb_line(t_label, F_LBL, TRACK_COLOR, gap=3)
    elif history:
        sb_line("HISTORY", F_HDR, HISTORY_COLOR, bold=True, gap=4)
        if t_label:
            sb_line(t_label, F_LBL, HISTORY_COLOR, gap=3)
    else:
        sb_line("click to track", F_LBL, gap=3)
    if annotation and (tracking or history):
        sb_line(annotation["mode"], F_LBL, ann_color, gap=3)
        for s in annotation["states"].values():
            sb_line(str(s), F_ROW, ann_color, gap=2)
    if track_frames > 0:
        sb_line(f"{track_frames} frames", F_HNT, (170, 170, 170), gap=3)
    sb_divider(sy); sy += 8

    # ---- Saved tracks list ----
    saved_tracks  = canvas_meta.get("saved_tracks",  [])
    previewed_set = canvas_meta.get("previewed_idxs", set())
    page          = canvas_meta.get("track_list_page", 0)
    video_name    = canvas_meta.get("video_name", "")
    total_saved   = len(saved_tracks)
    total_pages   = max(1, (total_saved + TRACKS_PER_PAGE - 1) // TRACKS_PER_PAGE)
    page          = max(0, min(page, total_pages - 1))

    sb_line(video_name[:22], F_HNT, (90, 110, 130), gap=2)
    sb_line(f"Saved: {total_saved}", F_LBL,
            color=(90, 200, 90) if total_saved > 0 else (70, 70, 70), gap=6)

    track_list_zones = []

    start_idx = page * TRACKS_PER_PAGE
    end_idx   = min(start_idx + TRACKS_PER_PAGE, total_saved)
    for i in range(start_idx, end_idx):
        t       = saved_tracks[i]
        t_mode  = t.get("mode", "?")
        t_fc    = t.get("frame_count", 0)
        pv      = (i in previewed_set)
        col     = PREVIEW_COLORS[i % len(PREVIEW_COLORS)] if pv else (75, 75, 75)
        prefix  = "[v]" if pv else "[ ]"
        row_txt = f"{prefix} #{i+1} {t_mode[:6]}  {t_fc}f"
        ry1 = sy;  ry2 = sy + ROW_H
        del_x1 = SIDEBAR_W - PAD - DELETE_W
        if pv:
            cv2.rectangle(canvas, (0, ry1), (SIDEBAR_W - 1, ry2), (30, 40, 50), -1)
        # [X] delete button — vertically centred in row
        cv2.rectangle(canvas, (del_x1, ry1 + 2), (SIDEBAR_W - PAD - 1, ry2 - 2),
                      (60, 28, 28), -1)
        cv2.putText(canvas, "X", (del_x1 + 4, _cy(ry1, ROW_H, F_BTN)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (200, 80, 80), 1, cv2.LINE_AA)
        sb_text(row_txt, _cy(ry1, ROW_H, F_ROW), F_ROW, col)
        track_list_zones.append({"y1": ry1, "y2": ry2, "type": "track", "idx": i,
                                  "x2": del_x1 - 1})
        track_list_zones.append({"y1": ry1, "y2": ry2, "type": "delete_track", "idx": i,
                                  "x1": del_x1})
        sy += ROW_H
    # fill blank rows to keep pagination stable
    for _ in range(TRACKS_PER_PAGE - (end_idx - start_idx)):
        sy += ROW_H

    # Pagination row (two separate buttons)
    if total_pages > 1 or total_saved > 0:
        HALF = SIDEBAR_W // 2
        pag_y1 = sy;  pag_y2 = sy + BTN_H
        btn_cy = _cy(pag_y1, BTN_H, F_BTN)
        cv2.rectangle(canvas, (1, pag_y1), (HALF - 1, pag_y2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (1, pag_y1), (HALF - 1, pag_y2), (70, 70, 70), 1)
        cv2.putText(canvas, "< prev", (PAD, btn_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (HALF + 1, pag_y1), (SIDEBAR_W - 1, pag_y2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (HALF + 1, pag_y1), (SIDEBAR_W - 1, pag_y2), (70, 70, 70), 1)
        cv2.putText(canvas, "next >", (HALF + PAD, btn_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (150, 150, 150), 1, cv2.LINE_AA)
        pg_txt = f"{page+1}/{total_pages}"
        (pg_w, _), _ = cv2.getTextSize(pg_txt, cv2.FONT_HERSHEY_SIMPLEX, F_HNT, 1)
        cv2.putText(canvas, pg_txt, (HALF - pg_w // 2, _cy(pag_y1, BTN_H, F_HNT)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_HNT, (110, 110, 110), 1, cv2.LINE_AA)
        track_list_zones.append({"y1": pag_y1, "y2": pag_y2, "type": "prev_page",
                                  "x2": HALF - 1})
        track_list_zones.append({"y1": pag_y1, "y2": pag_y2, "type": "next_page",
                                  "x1": HALF + 1})
        sy += BTN_H + 2

    # Show-all / Hide-all buttons
    if total_saved > 0:
        HALF = SIDEBAR_W // 2
        btn_y1 = sy;  btn_y2 = sy + BTN_H
        btn_cy = _cy(btn_y1, BTN_H, F_BTN)
        cv2.rectangle(canvas, (1, btn_y1), (HALF - 1, btn_y2), (30, 50, 30), -1)
        cv2.rectangle(canvas, (1, btn_y1), (HALF - 1, btn_y2), (60, 100, 60), 1)
        cv2.putText(canvas, "show all", (PAD, btn_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (80, 200, 80), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (HALF + 1, btn_y1), (SIDEBAR_W - 1, btn_y2), (50, 30, 30), -1)
        cv2.rectangle(canvas, (HALF + 1, btn_y1), (SIDEBAR_W - 1, btn_y2), (100, 60, 60), 1)
        cv2.putText(canvas, "hide all", (HALF + PAD, btn_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (200, 80, 80), 1, cv2.LINE_AA)
        track_list_zones.append({"y1": btn_y1, "y2": btn_y2, "type": "show_all",
                                  "x2": HALF - 1})
        track_list_zones.append({"y1": btn_y1, "y2": btn_y2, "type": "hide_all",
                                  "x1": HALF + 1})
        sy += BTN_H + 2

    _layout["track_list_zones"] = track_list_zones
    sb_divider(sy); sy += 8

    # ---- Markers ----
    markers_list = canvas_meta.get("markers", [])

    def _fmt_tc(fn):
        t = fn / fps if fps else 0
        return f"{int(t // 60):02d}:{t % 60:05.2f} f{fn}"

    if markers_list:
        sb_line("MARKERS", F_HDR, (120, 200, 255), bold=True, gap=5)
        for j, m in enumerate(markers_list):
            mc      = MARKER_COLORS[j % len(MARKER_COLORS)]
            row_txt = f"{m['label'][:6]}  {_fmt_tc(m['frame'])}"
            my1 = sy;  my2 = sy + MRK_ROW_H
            del_x1 = SIDEBAR_W - PAD - DELETE_W
            cv2.rectangle(canvas, (del_x1, my1 + 2), (SIDEBAR_W - PAD - 1, my2 - 2),
                          (60, 28, 28), -1)
            cv2.putText(canvas, "X", (del_x1 + 4, _cy(my1, MRK_ROW_H, F_BTN)),
                        cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (200, 80, 80), 1, cv2.LINE_AA)
            sb_text(row_txt, _cy(my1, MRK_ROW_H, F_ROW), F_ROW, mc)
            track_list_zones.append({"y1": my1, "y2": my2, "type": "delete_marker",
                                      "idx": j, "x1": del_x1})
            sy += MRK_ROW_H
    else:
        sb_line("[M] add marker", F_HNT, (80, 80, 80), gap=5)
    sb_divider(sy); sy += 8

    # ---- Key hints (skip any that would overflow into the timeline) ----
    hints = [
        ("[ENTER] save",         (80, 200, 80)),
        ("[BKSP] clear",         (200, 80, 80)),
        ("[A+BKSP] del before",  (200, 140, 60)),
        ("[S+A+BKSP] del after", (200, 140, 60)),
        ("[click] detect",       (140, 140, 140)),
        ("[drag] draw box",      (210, 210, 50)),
        ("[<][>] frame step",    (140, 140, 140)),
        ("[SPACE] play/pause",   (140, 140, 140)),
        ("[Q/ESC] quit",         (140, 140, 140)),
    ]
    for htxt, hcol in hints:
        if sy + _lh(F_HNT) > video_area_h - 6:
            break
        sb_line(htxt, F_HNT, hcol, gap=3)

    # --- video (letterboxed, no stretch) ---
    fh, fw = frame.shape[:2]
    scale  = min(video_area_w / fw, video_area_h / fh)
    new_w  = int(fw * scale)
    new_h  = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_off = SIDEBAR_W + (video_area_w - new_w) // 2
    y_off = (video_area_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Update shared layout for back-projection in canvas_to_frame()
    _layout.update({"x_off": x_off, "y_off": y_off,
                    "new_w": new_w, "new_h": new_h, "scale": scale})

    # --- draw bounding boxes (scaled from frame coords to canvas coords) ---
    show_popup   = canvas_meta.get("show_popup", False)
    popup_zones  = []
    popup_anchor = None   # (sx1, sy1) canvas coords of first live/history box

    def _draw_ann_overlay(ann_lines, bx_left, ann_y0, ann_font, line_h):
        """Draw per-scope translucent coloured background + text for each ann line.
        ann_lines : list of (text, scope) where scope in {'mode','per_frame','per_track'}
        """
        if not ann_lines:
            return
        max_tw = max((cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, ann_font, 1)[0][0]
                      for txt, _ in ann_lines), default=0)
        for k, (txt, scope) in enumerate(ann_lines):
            row_y0  = ann_y0 + k * line_h
            bg_col  = POPUP_MODE_BG    if scope == "mode"      else (
                      POPUP_TRACK_BG   if scope == "per_track" else POPUP_FRAME_BG)
            txt_col = POPUP_MODE_COLOR  if scope == "mode"      else (
                      POPUP_TRACK_COLOR if scope == "per_track" else POPUP_FRAME_COLOR)
            bx1c = max(0, bx_left + 2)
            by1c = max(0, row_y0 - 13)
            bx2c = min(canvas.shape[1] - 1, bx_left + max_tw + 10)
            by2c = min(canvas.shape[0] - 1, row_y0 + 4)
            if bx2c > bx1c and by2c > by1c:
                sub = canvas[by1c:by2c, bx1c:bx2c].copy()
                bg  = np.zeros_like(sub)
                bg[:] = bg_col
                canvas[by1c:by2c, bx1c:bx2c] = cv2.addWeighted(bg, 0.65, sub, 0.35, 0)
            cv2.putText(canvas, txt, (bx_left + 5, row_y0),
                        cv2.FONT_HERSHEY_SIMPLEX, ann_font, txt_col, 1, cv2.LINE_AA)

    for (x1, y1, x2, y2, lbl, col) in boxes:
        sx1 = int(x_off + x1 * scale)
        sy1 = int(y_off + y1 * scale)
        sx2 = int(x_off + x2 * scale)
        sy2 = int(y_off + y2 * scale)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), col, 2)
        label_y = max(sy1 - 6, y_off + 14)
        cv2.putText(canvas, lbl, (sx1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        if annotation and col != DIM_COLOR and not show_popup:
            ann_lines = [(annotation["mode"], "mode")]
            for entry in _ordered_state_cats(annotation["mode"]):
                cat = entry[0]
                if cat in annotation["states"]:
                    ann_lines.append((str(annotation["states"][cat]),
                                      entry[2] if len(entry) > 2 else 'per_frame'))
            _draw_ann_overlay(ann_lines, sx1, label_y + 18, 0.42, 16)
        if popup_anchor is None and col in (TRACK_COLOR, HISTORY_COLOR):
            popup_anchor = (sx1, sy1)

    # --- draw preview boxes from saved tracks ---
    for (x1, y1, x2, y2, lbl, col, ann_lines) in preview_boxes:
        sx1 = int(x_off + x1 * scale)
        sy1 = int(y_off + y1 * scale)
        sx2 = int(x_off + x2 * scale)
        sy2 = int(y_off + y2 * scale)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), col, 2)
        label_y = max(sy1 - 6, y_off + 14)
        cv2.putText(canvas, lbl, (sx1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        _draw_ann_overlay(ann_lines, sx1, label_y + 18, 0.40, 15)

    # --- in-progress drag rectangle (live preview while user is still dragging) ---
    drag_rect = canvas_meta.get("drag_rect")
    if drag_rect is not None:
        dcx1, dcy1, dcx2, dcy2 = drag_rect
        rx1, ry1 = min(dcx1, dcx2), min(dcy1, dcy2)
        rx2, ry2 = max(dcx1, dcx2), max(dcy1, dcy2)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(canvas, "manual box", (rx1 + 4, max(ry1 - 6, y_off + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw popup (shown only on the click frame while paused)
    if show_popup and annotation and popup_anchor is not None:
        popup_zones = draw_popup(
            canvas, popup_anchor[0], popup_anchor[1],
            annotation["mode"], annotation["states"],
            screen_w, screen_h,
        )
    _layout["popup_zones"] = popup_zones

    # --- timeline ---
    cv2.rectangle(canvas, (0, timeline_y), (screen_w, screen_h), (20, 20, 20), -1)
    cv2.line(canvas, (0, timeline_y), (screen_w, timeline_y), (55, 55, 55), 1)

    tl_markers = canvas_meta.get("markers", [])
    tl_mid_y   = timeline_y + (screen_h - timeline_y) // 2

    if total_frames > 1:
        filled_w = int(screen_w * current_frame / total_frames)
        cv2.rectangle(canvas, (0, timeline_y + 1), (filled_w, screen_h), (0, 180, 0), -1)
        cv2.rectangle(canvas,
                      (filled_w - 2, timeline_y),
                      (filled_w + 2, screen_h),
                      (255, 255, 255), -1)
        # Named marker ticks – drawn on top of progress bar
        for j, m in enumerate(tl_markers):
            mc = MARKER_COLORS[j % len(MARKER_COLORS)]
            mx = int(screen_w * m["frame"] / total_frames)
            cv2.rectangle(canvas, (mx - 1, timeline_y), (mx + 2, screen_h), mc, -1)
            cv2.putText(canvas, m["label"], (mx + 4, tl_mid_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, mc, 1, cv2.LINE_AA)

    time_sec  = current_frame / fps if fps else 0
    total_sec = total_frames  / fps if fps else 0
    status    = "PAUSED" if paused else "PLAYING"
    mrk_tag   = (f"   [{len(tl_markers)} marker{'s' if len(tl_markers)!=1 else ''}]"
                 if tl_markers else "")
    label = (
        f"  {status}   "
        f"{int(time_sec // 60):02d}:{time_sec % 60:05.2f} / "
        f"{int(total_sec // 60):02d}:{total_sec % 60:05.2f}   "
        f"Frame {current_frame} / {total_frames}"
        f"{mrk_tag}"
    )
    text_y = timeline_y + (screen_h - timeline_y + 14) // 2
    cv2.putText(canvas, label, (6, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def _markers_path(video_path):
    return os.path.splitext(os.path.abspath(video_path))[0] + "_tracks.json"


def save_markers_to_json(video_path, markers):
    """Persist only the markers list into the JSON file without touching saved tracks."""
    outpath = _markers_path(video_path)
    if os.path.isfile(outpath):
        with open(outpath, "r") as f:
            _raw = json.load(f)
        if isinstance(_raw, dict):
            all_tracks = _raw.get("tracks", [])
        else:
            all_tracks = _raw
    else:
        all_tracks = []
    out_doc = {"tracks": all_tracks, "markers": markers}
    with open(outpath, "w") as f:
        json.dump(out_doc, f, indent=2)


def save_track_to_json(video_path, track_history, fps, markers=None):
    """Append the current track (with per-frame mode+states) to the JSON file.
    Saves markers alongside tracks. Returns the saved track dict.
    """
    outpath = _markers_path(video_path)

    if os.path.isfile(outpath):
        with open(outpath, "r") as f:
            _raw = json.load(f)
        if isinstance(_raw, dict):
            all_tracks = _raw.get("tracks", [])
        else:
            all_tracks = _raw   # migrate old format
    else:
        all_tracks = []

    frames_data = []
    for frame_num in sorted(track_history.keys()):
        x1, y1, x2, y2, label, fmode, fstates = track_history[frame_num]
        # Only store per_frame states per frame
        pf_states = {cat: val for cat, val in fstates.items()
                     if _state_scope(fmode, cat) == 'per_frame'}
        frames_data.append({
            "frame_num": frame_num,
            "time_sec":  round(frame_num / fps, 4) if fps else 0,
            "box":       [x1, y1, x2, y2],
            "label":     label,
            "mode":      fmode,
            "states":    pf_states,
        })

    # Collect per_track states from the first frame (same for all frames)
    if track_history:
        first_fn = min(track_history.keys())
        _, _, _, _, _, first_mode, first_states = track_history[first_fn]
        pt_states = {cat: val for cat, val in first_states.items()
                     if _state_scope(first_mode, cat) == 'per_track'}
    else:
        pt_states = {}

    # Derive a track-level mode from the most common frame-mode
    if frames_data:
        modes    = [d["mode"] for d in frames_data]
        top_mode = max(set(modes), key=modes.count)
    else:
        top_mode = DEFAULT_MODE

    track_dict = {
        "saved_at":     datetime.datetime.now().isoformat(timespec="seconds"),
        "mode":         top_mode,
        "track_states": pt_states,
        "video_file":   os.path.abspath(video_path),
        "frame_count":  len(frames_data),
        "frames":       frames_data,
    }
    all_tracks.append(track_dict)

    out_doc = {"tracks": all_tracks, "markers": markers or []}
    with open(outpath, "w") as f:
        json.dump(out_doc, f, indent=2)

    print(f"[save] {len(frames_data)} frames  →  {outpath}")
    return track_dict


def reset_botsort(model):
    """Reset BotSort's internal state (call on seek/scrub so track IDs restart)."""
    try:
        if hasattr(model, 'predictor') and model.predictor is not None:
            for t in model.predictor.trackers:
                t.reset()
    except Exception:
        # Safest fallback: drop the predictor; it will be re-created on next call
        try:
            model.predictor = None
        except Exception:
            pass


def run_tracking(model, frame, click_pos=None, tracked_id=None):
    """Run YOLO+BotSort on frame – class-agnostic.

    click_pos  – (fx, fy) lock onto whichever box the click lands in.
    tracked_id – int BotSort track ID to follow.

    Returns (boxes, new_tracked_id, tracked_label).
    """
    results = model.track(frame, tracker="botsort.yaml",
                          persist=True, verbose=False)[0]

    all_dets = []   # (tid, x1, y1, x2, y2, class_name)
    if results.boxes.id is not None:
        names = results.names
        for box, tid in zip(results.boxes, results.boxes.id.int().tolist()):
            cls_name = names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_dets.append((tid, x1, y1, x2, y2, cls_name))

    print(f"[botsort] detections={len(all_dets)} tracked_id={tracked_id}")

    # --- user clicked: lock onto whichever box contains the click ---
    if click_pos is not None:
        fx, fy = click_pos
        for (tid, x1, y1, x2, y2, cls_name) in all_dets:
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                print(f"[botsort] locked id={tid} cls={cls_name}")
                return [(x1, y1, x2, y2, cls_name, TRACK_COLOR)], tid, cls_name
        print("[botsort] click outside all boxes – showing all")
        boxes = [(x1, y1, x2, y2, cls_name, DIM_COLOR)
                 for (_, x1, y1, x2, y2, cls_name) in all_dets]
        return boxes, None, ""

    # --- following a specific track ID ---
    if tracked_id is not None:
        for (tid, x1, y1, x2, y2, cls_name) in all_dets:
            if tid == tracked_id:
                return [(x1, y1, x2, y2, cls_name, TRACK_COLOR)], tracked_id, cls_name
        print(f"[botsort] lost id={tracked_id}")
        return [], None, ""

    return [], None, ""


def seek(cap, target_frame, total_frames):
    target_frame = max(0, min(target_frame, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - (1 if ret else 0)
    return ret, frame, actual


def interpolate_track_history(track_history):
    """Fill gaps between keyed frames with linearly-interpolated boxes.
    States, label and mode are carried forward from the earlier bounding frame.
    Does not overwrite frames that already exist in track_history.
    """
    frames = sorted(track_history.keys())
    for i in range(len(frames) - 1):
        fa, fb = frames[i], frames[i + 1]
        if fb - fa <= 1:
            continue
        x1a, y1a, x2a, y2a, lbl_a, mode_a, states_a = track_history[fa]
        x1b, y1b, x2b, y2b, _,     _,      _         = track_history[fb]
        for fi in range(fa + 1, fb):
            if fi in track_history:
                continue   # never overwrite an explicit frame
            t   = (fi - fa) / (fb - fa)
            ix1 = round(x1a + t * (x1b - x1a))
            iy1 = round(y1a + t * (y1b - y1a))
            ix2 = round(x2a + t * (x2b - x2a))
            iy2 = round(y2a + t * (y2b - y2a))
            track_history[fi] = (ix1, iy1, ix2, iy2, lbl_a, mode_a, dict(states_a))


def main():
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        print("Usage: python main.py <path_to_video>")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_FPS, 30.0)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay_ms = max(1, int(1000 / fps))

    screen_w, screen_h = get_screen_size()

    print("Loading YOLO model …")
    model = YOLO("yolo26n.pt")   # auto-downloaded on first run
    print("Model ready.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, (screen_w, screen_h, total_frames))

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read any frames from the video.")
        cap.release()
        sys.exit(1)

    paused         = False
    current_frame  = 0
    boxes          = []    # (x1,y1,x2,y2,label,color)
    tracked_id     = None  # BotSort track ID being followed
    tracked_label  = ""    # class name of locked object
    track_history  = {}    # frame_num -> (x1,y1,x2,y2,label,mode,states)
    # Per-frame annotation
    track_mode   = DEFAULT_MODE
    track_states = _default_track_states(DEFAULT_MODE)
    # Last used states per mode – restored whenever a new track starts or mode switches
    last_states_per_mode = {}   # {mode_str: {cat: val}}
    show_popup   = False
    # Named markers
    markers        = []   # list of {"label": str, "frame": int}
    marker_counter = 0    # auto-increment for default labels
    # Saved-track panel
    saved_tracks    = []    # list of full track dicts (with '_frame_lookup')
    previewed_idxs  = set() # indices of saved_tracks currently being previewed
    track_list_page = 0
    video_name      = os.path.basename(video_path)

    # Load existing tracks from JSON on startup
    json_path = os.path.splitext(os.path.abspath(video_path))[0] + "_tracks.json"
    if os.path.isfile(json_path):
        try:
            with open(json_path) as _f:
                _loaded = json.load(_f)
            # Handle both old format (list of tracks) and new format (dict with tracks+markers)
            if isinstance(_loaded, dict):
                _tracks_raw  = _loaded.get("tracks", [])
                _markers_raw = _loaded.get("markers", [])
            else:
                _tracks_raw  = _loaded
                _markers_raw = []
            for _td in _tracks_raw:
                _td["_frame_lookup"] = build_frame_lookup(_td)
                saved_tracks.append(_td)
            for _m in _markers_raw:
                markers.append(_m)
                marker_counter = max(marker_counter,
                    int(_m["label"][1:]) if _m["label"].startswith("M") and
                    _m["label"][1:].isdigit() else marker_counter)
            print(f"[load] {len(saved_tracks)} tracks, {len(markers)} markers loaded")
            # Auto-seek to first marker if any
            if markers:
                first_m = min(markers, key=lambda mk: mk["frame"])
                paused = True
                ret, frame, current_frame = seek(cap, first_m["frame"], total_frames)
                print(f"[load] auto-seek to marker '{first_m['label']}' at frame {current_frame}")
        except Exception as _e:
            print(f"[load] failed to read {json_path}: {_e}")

    while True:
        in_history  = (tracked_id is None and current_frame in track_history)
        _annotation = {"mode": track_mode, "states": dict(track_states)} \
                      if (tracked_id is not None or in_history) else None

        # Build preview boxes from all actively previewed saved tracks
        preview_boxes = []
        for pidx in sorted(previewed_idxs):
            if pidx < len(saved_tracks):
                lu = saved_tracks[pidx].get("_frame_lookup", {})
                if current_frame in lu:
                    px1, py1, px2, py2, plbl, pmode, pstates = lu[current_frame]
                    pcol      = PREVIEW_COLORS[pidx % len(PREVIEW_COLORS)]
                    ann_lines = [(pmode, "mode")]
                    if pstates:
                        for entry in _ordered_state_cats(pmode):
                            pcat = entry[0]
                            if pcat in pstates:
                                ann_lines.append((str(pstates[pcat]),
                                                  entry[2] if len(entry) > 2 else 'per_frame'))
                    preview_boxes.append((px1, py1, px2, py2, plbl, pcol, ann_lines))

        _dr = _state["drag_start"]
        _dc = _state["drag_current"]
        _drag_rect = (_dr[0], _dr[1], _dc[0], _dc[1]) if _dr and _dc else None
        canvas = build_canvas(frame, screen_w, screen_h,
                              current_frame, total_frames, fps, paused,
                              boxes, preview_boxes,
                              {"tracking":       tracked_id is not None,
                               "history":        in_history,
                               "tracked_label":  tracked_label,
                               "annotation":     _annotation,
                               "show_popup":     show_popup,
                               "track_frames":   len(track_history),
                               "saved_tracks":   saved_tracks,
                               "previewed_idxs": previewed_idxs,
                               "track_list_page":track_list_page,
                               "video_name":     video_name,
                               "markers":        markers,
                               "drag_rect":      _drag_rect})
        cv2.imshow(WINDOW_NAME, canvas)

        wait_ms = 30 if paused else frame_delay_ms
        key = cv2.waitKeyEx(wait_ms)

        # --- timeline click → seek ---
        if _state["seek_frame"] is not None:
            paused     = True
            tracked_id = None
            show_popup = False
            reset_botsort(model)
            ret, frame, current_frame = seek(cap, _state["seek_frame"], total_frames)
            _state["seek_frame"] = None
            if current_frame in track_history:
                hx1, hy1, hx2, hy2, hlbl, hmode, hstates = track_history[current_frame]
                boxes         = [(hx1, hy1, hx2, hy2, hlbl, HISTORY_COLOR)]
                tracked_label = hlbl
                track_mode    = hmode
                track_states  = dict(hstates)
            else:
                boxes         = []
                tracked_label = ""
            continue

        # --- sidebar click → track-list interaction ---
        if _state["sidebar_click"] is not None:
            scx, scy = _state["sidebar_click"]
            _state["sidebar_click"] = None
            total_pages = max(1, (len(saved_tracks) + TRACKS_PER_PAGE - 1) // TRACKS_PER_PAGE)
            for zone in _layout.get("track_list_zones", []):
                if not (zone["y1"] <= scy <= zone["y2"]):
                    continue
                # optional x-range filter
                if scx < zone.get("x1", 0):
                    continue
                if "x2" in zone and scx > zone["x2"]:
                    continue
                ztype = zone["type"]
                if ztype == "track":
                    idx = zone["idx"]
                    if idx in previewed_idxs:
                        previewed_idxs.discard(idx)
                    else:
                        previewed_idxs.add(idx)
                    break
                elif ztype == "delete_track":
                    idx = zone["idx"]
                    if 0 <= idx < len(saved_tracks):
                        del saved_tracks[idx]
                        # shift previewed_idxs: remove deleted, decrement those above
                        previewed_idxs = {(i if i < idx else i - 1)
                                          for i in previewed_idxs if i != idx}
                    break
                elif ztype == "delete_marker":
                    idx = zone["idx"]
                    if 0 <= idx < len(markers):
                        del markers[idx]
                        save_markers_to_json(video_path, markers)
                    break
                elif ztype == "show_all":
                    previewed_idxs = set(range(len(saved_tracks)))
                    break
                elif ztype == "hide_all":
                    previewed_idxs = set()
                    break
                elif ztype == "prev_page":
                    track_list_page = max(0, track_list_page - 1)
                    break
                elif ztype == "next_page":
                    track_list_page = min(total_pages - 1, track_list_page + 1)
                    break
            continue

        # --- drag end → add manual bounding box ---
        if _state["drag_end"] is not None:
            dcx1, dcy1, dcx2, dcy2 = _state["drag_end"]
            _state["drag_end"] = None
            fp1 = canvas_to_frame(dcx1, dcy1)
            fp2 = canvas_to_frame(dcx2, dcy2)
            if fp1 is not None and fp2 is not None:
                fx1, fy1 = fp1
                fx2, fy2 = fp2
                fx1, fx2 = min(fx1, fx2), max(fx1, fx2)
                fy1, fy2 = min(fy1, fy2), max(fy1, fy2)
                if fx2 > fx1 and fy2 > fy1:
                    # Carry mode + states from the nearest previous frame; fall back to current
                    prev_frames = sorted(f for f in track_history if f <= current_frame)
                    if prev_frames:
                        _, _, _, _, _, prev_mode, prev_sts = track_history[prev_frames[-1]]
                        draw_mode   = prev_mode
                        # Filter to only the cats that belong to this mode (no cross-mode leakage)
                        valid_cats  = {e[0] for e in TRACK_STATES.get(draw_mode, [])}
                        draw_states = {k: v for k, v in prev_sts.items() if k in valid_cats}
                        # Fill in any missing cats with defaults (keeps the full set)
                        for e in _ordered_state_cats(draw_mode):
                                draw_states.setdefault(e[0], e[1][0])
                    else:
                        draw_mode   = track_mode
                        draw_states = dict(track_states)
                    draw_label  = draw_mode
                    track_history[current_frame] = (fx1, fy1, fx2, fy2,
                                                    draw_label, draw_mode, draw_states)
                    track_mode    = draw_mode
                    track_states  = dict(draw_states)
                    tracked_id    = None
                    tracked_label = draw_label
                    boxes         = [(fx1, fy1, fx2, fy2, draw_label, TRACK_COLOR)]
                    paused        = True
                    show_popup    = True
                    print(f"[draw] manual box at frame {current_frame}  {fx1},{fy1}-{fx2},{fy2}")
            continue

        # --- video click → popup toggle OR BotSort detect + lock ---
        if _state["detect_click"] is not None:
            cx, cy = _state["detect_click"]
            _state["detect_click"] = None
            # If popup is visible, check popup zones first
            hit_popup = False
            if show_popup:
                for zone in _layout.get("popup_zones", []):
                    zx1, zy1, zx2, zy2 = zone["rect"]
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        if zone["action"] == "mode":
                            last_states_per_mode[track_mode] = dict(track_states)
                            track_mode   = zone["next_val"]
                            track_states = dict(last_states_per_mode.get(
                                track_mode, _default_track_states(track_mode)))
                            # Rewrite current frame with new mode + mode-specific states
                            if current_frame in track_history:
                                x1h,y1h,x2h,y2h,lblh,_,_ = track_history[current_frame]
                                # label follows mode
                                track_history[current_frame] = (
                                    x1h, y1h, x2h, y2h, track_mode,
                                    track_mode, dict(track_states))
                        else:  # "state"
                            cat_name = zone["cat"]
                            new_val  = zone["next_val"]
                            track_states[cat_name] = new_val
                            # per_track: retroactively update ALL existing history frames
                            if zone.get("scope") == 'per_track':
                                for fn in list(track_history):
                                    tx1,ty1,tx2,ty2,tlbl,tmode,tstates = track_history[fn]
                                    tstates = dict(tstates)
                                    tstates[cat_name] = new_val
                                    track_history[fn] = (tx1,ty1,tx2,ty2,tlbl,tmode,tstates)
                            else:  # per_frame: write back only for the current frame
                                if current_frame in track_history:
                                    x1h,y1h,x2h,y2h,lblh,modeh,stsh = track_history[current_frame]
                                    stsh = dict(stsh)
                                    stsh[cat_name] = new_val
                                    track_history[current_frame] = (
                                        x1h, y1h, x2h, y2h, lblh, modeh, stsh)
                        hit_popup = True
                        break
            if not hit_popup:
                frame_pos = canvas_to_frame(cx, cy)
                if frame_pos is not None:
                    # If paused on a history frame, clicking the existing box re-shows popup
                    # (allows editing per_frame states frame-by-frame without re-detecting)
                    in_hist = (tracked_id is None and current_frame in track_history)
                    if in_hist:
                        hx1, hy1, hx2, hy2, *_ = track_history[current_frame]
                        fx, fy = frame_pos
                        if hx1 <= fx <= hx2 and hy1 <= fy <= hy2:
                            show_popup = True
                            hit_popup  = True
                    if not hit_popup:
                        paused     = True
                        boxes, tracked_id, tracked_label = run_tracking(
                            model, frame, click_pos=frame_pos)
                        if tracked_id is not None:
                            last_states_per_mode[track_mode] = dict(track_states)
                            track_mode   = YOLO_CLASS_MAP.get(tracked_label, DEFAULT_MODE)
                            track_states = dict(last_states_per_mode.get(
                                track_mode, _default_track_states(track_mode)))
                            show_popup   = True
                        else:
                            show_popup   = False
            continue

        # --- keyboard ---
        if key != -1:
            key_ch = key & 0xFF

            if key_ch in (ord('q'), ord('Q'), 27):    # Q / Esc
                break

            elif key_ch in (ord('m'), ord('M')):       # M – add named marker
                marker_counter += 1
                mlabel = f"M{marker_counter}"
                markers.append({"label": mlabel, "frame": current_frame})
                markers.sort(key=lambda mk: mk["frame"])
                save_markers_to_json(video_path, markers)
                print(f"[marker] added {mlabel} at frame {current_frame}")

            elif key_ch == 32:                         # Space – play/pause
                paused     = not paused
                show_popup = False  # dismiss popup when playback resumes

            elif key_ch in KEY_ENTER:                  # Enter – save current track
                if track_history:
                    interpolate_track_history(track_history)
                    track_dict = save_track_to_json(video_path, track_history, fps, markers)
                    track_dict["_frame_lookup"] = build_frame_lookup(track_dict)
                    saved_tracks.append(track_dict)
                    last_states_per_mode[track_mode] = dict(track_states)
                    track_history  = {}
                    boxes          = []
                    tracked_id     = None
                    tracked_label  = ""
                    show_popup     = False
                    track_mode     = DEFAULT_MODE
                    track_states   = dict(last_states_per_mode.get(
                        DEFAULT_MODE, _default_track_states(DEFAULT_MODE)))
                    reset_botsort(model)
                    print(f"[save] track saved. total saved this session: {len(saved_tracks)}")
                else:
                    print("[save] nothing to save – no frames tracked yet")

            elif key_ch in KEY_BKSP:
                _user32    = ctypes.windll.user32
                alt_down   = bool(_user32.GetAsyncKeyState(VK_ALT)   & 0x8000)
                shift_down = bool(_user32.GetAsyncKeyState(VK_SHIFT) & 0x8000)

                if alt_down and shift_down:        # Shift+Alt+Backspace – delete AFTER
                    deleted = [f for f in list(track_history) if f > current_frame]
                    for f in deleted:
                        del track_history[f]
                    if tracked_id is not None and not track_history:
                        tracked_id    = None
                        tracked_label = ""
                        reset_botsort(model)
                    if current_frame in track_history:
                        hx1, hy1, hx2, hy2, hlbl, *_ = track_history[current_frame]
                        boxes = [(hx1, hy1, hx2, hy2, hlbl, HISTORY_COLOR)]
                    else:
                        boxes = []
                    print(f"[trim] deleted {len(deleted)} frames after frame {current_frame}")

                elif alt_down:                     # Alt+Backspace – delete BEFORE
                    deleted = [f for f in list(track_history) if f < current_frame]
                    for f in deleted:
                        del track_history[f]
                    if tracked_id is not None and not track_history:
                        tracked_id    = None
                        tracked_label = ""
                        reset_botsort(model)
                    print(f"[trim] deleted {len(deleted)} frames before frame {current_frame}")

                else:                              # plain Backspace – clear current track
                    last_states_per_mode[track_mode] = dict(track_states)
                    track_history = {}
                    boxes         = []
                    tracked_id    = None
                    tracked_label = ""
                    show_popup    = False
                    track_mode    = DEFAULT_MODE
                    track_states  = dict(last_states_per_mode.get(
                        DEFAULT_MODE, _default_track_states(DEFAULT_MODE)))
                    reset_botsort(model)
                    print("[clear] current unsaved track deleted")

            elif key in KEY_RIGHT:                     # → scrub forward
                paused     = True
                tracked_id = None
                show_popup = False
                reset_botsort(model)
                ret, frame, current_frame = seek(cap, current_frame + SCRUB_STEP, total_frames)
                if current_frame in track_history:
                    hx1, hy1, hx2, hy2, hlbl, hmode, hstates = track_history[current_frame]
                    boxes         = [(hx1, hy1, hx2, hy2, hlbl, HISTORY_COLOR)]
                    tracked_label = hlbl
                    track_mode    = hmode
                    track_states  = dict(hstates)
                else:
                    boxes         = []
                    tracked_label = ""

            elif key in KEY_LEFT:                      # ← scrub backward
                paused     = True
                tracked_id = None
                show_popup = False
                reset_botsort(model)
                ret, frame, current_frame = seek(cap, current_frame - SCRUB_STEP, total_frames)
                if current_frame in track_history:
                    hx1, hy1, hx2, hy2, hlbl, hmode, hstates = track_history[current_frame]
                    boxes         = [(hx1, hy1, hx2, hy2, hlbl, HISTORY_COLOR)]
                    tracked_label = hlbl
                    track_mode    = hmode
                    track_states  = dict(hstates)
                else:
                    boxes         = []
                    tracked_label = ""

        # --- advance playback ---
        if not paused:
            ret, next_frame = cap.read()
            if not ret:
                paused = True
            else:
                frame         = next_frame
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                show_popup    = False  # popup is click-frame only

                if tracked_id is not None:
                    # Live BotSort tracking (user clicked to lock a target)
                    boxes, tracked_id, tracked_label = run_tracking(
                        model, frame, tracked_id=tracked_id)
                    # Record to history with per-frame mode and states
                    if boxes:
                        x1, y1, x2, y2, lbl, _ = boxes[0]
                        track_history[current_frame] = (x1, y1, x2, y2, lbl,
                                                        track_mode, dict(track_states))
                elif current_frame in track_history:
                    # Playing through already-tracked frames – show stored box only
                    hx1, hy1, hx2, hy2, hlbl, hmode, hstates = track_history[current_frame]
                    boxes         = [(hx1, hy1, hx2, hy2, hlbl, HISTORY_COLOR)]
                    tracked_label = hlbl
                    track_mode    = hmode
                    track_states  = dict(hstates)
                else:
                    boxes         = []
                    tracked_label = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
