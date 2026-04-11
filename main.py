import cv2
import sys
import os
import ctypes
import json
import datetime
import numpy as np
from ultralytics import YOLO
from scipy.signal import savgol_filter

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
        ('movement',              ['walking', 'waiting'],                                       'per_frame'),
        ('crosswalk position',    ['outside crosswalk', 'inside crosswalk'],                    'per_frame'),
        ('distraction',           ['not distracted', 'distacted'],                              'per_frame'),
        ('interaction',           ['no interaction', 'gesture', 'verbal', 'eye-contact'],       'per_frame'),
        ('wait time',             ['just arrived', 'waited'],                                   'per_track'),
        ('start position',        ['near', 'far'],                                              'per_track'),
        ('carrying object',       ['no object', 'carrying object'],                             'per_track'),
    ],
    'vehicle': [
        ('movement',              ['turning', 'waiting'],                                                       'per_frame'),
        ('position',              ['outside crosswalk', 'inside crosswalk'],                                    'per_frame'),
        ('interaction',           ['no interaction', 'honk', 'gap-find', 'gesture', 'verbal', 'eye-contact'],   'per_frame'),
        ('type',                  ['suv', 'truck', 'bus', 'sedan', 'van', 'pickup'],                            'per_track'),
        ('wait time',             ['just arrived', 'waited'],                                                   'per_track'),
    ]
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

# Performance tuning
DETECT_CLASS_IDS      = [0, 2, 5, 7]  # person, car, bus, truck (COCO IDs)
TRACK_INFER_IMGSZ     = 512            # smaller input => faster inference
TRACK_INFER_EVERY_N   = 2              # run model every N playback frames while locked
PLAYBACK_SKIP_FRAMES  = 1              # extra frames grabbed (not decoded) during playback
FAST_PLAY_WAIT_MS     = 1              # minimum wait while playing

TRACK_COLOR   = (0, 220, 255)   # cyan   – live tracked object
HISTORY_COLOR = (80, 150, 255)  # blue   – historical (seeked-to) box
DIM_COLOR     = (160, 160, 160) # grey   – "showing all" fallback detections
POLY_COLOR    = (0, 255, 255)   # yellow – crosswalk polygon
POLY_TMP_COLOR= (0, 180, 255)   # orange – in-progress polygon
POLY_EDGE_COLORS = [
    (0, 255, 255),   # side 1
    (0, 200, 120),   # side 2
    (255, 120, 0),   # side 3
    (220, 80, 220),  # side 4
]
POLY_DIAG_COLORS = [
    (120, 220, 255),  # diagonal 1
    (255, 180, 120),  # diagonal 2
]

# Shared mutable state (written by mouse callback, read by main loop)
_state = {
    "seek_frame":    None,   # timeline click    → seek to frame
    "detect_click":  None,   # video area click  → (canvas_x, canvas_y)
    "sidebar_click": None,   # sidebar click     → (canvas_x, canvas_y)
    "drag_start":    None,   # LBUTTONDOWN in video area → (cx, cy) canvas coords
    "drag_current":  None,   # MOUSEMOVE while dragging  → (cx, cy) canvas coords
    "drag_end":      None,   # LBUTTONUP after drag      → (cx1,cy1,cx2,cy2)
    "polygon_drag_idx":     None,   # active polygon point index while dragging
    "polygon_drag_current": None,   # current canvas position of dragged polygon point
    "polygon_drag_end":     None,   # (idx, cx, cy) when polygon drag completes
    "hover_pos":      None,   # current mouse canvas position
    "vp_drag_start":  None,   # (cx, cy) start while dragging vehicle-pos line
    "vp_drag_line":   None,   # (line_x, line_y1, line_y2) at drag start (canvas)
    "vp_drag_current": None,  # current mouse canvas position during vehicle-pos drag
    "vp_drag_end":    None,   # (line_x, line_y1, line_y2) after drag completes (canvas)
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

    if event == cv2.EVENT_MOUSEMOVE:
        _state["hover_pos"] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= timeline_top:
            if total_frames > 0:
                _state["seek_frame"] = int(max(0.0, min(1.0, x / screen_w)) * total_frames)
        elif x < SIDEBAR_W:
            _state["sidebar_click"] = (x, y)
        else:
            vp_zone = _layout.get("vehicle_pos_line_zone")
            if vp_zone is not None:
                lx, ly1, ly2 = vp_zone["x"], vp_zone["y1"], vp_zone["y2"]
                tol = vp_zone.get("tol", 10)
                if abs(x - lx) <= tol and (min(ly1, ly2) - tol) <= y <= (max(ly1, ly2) + tol):
                    _state["vp_drag_start"] = (x, y)
                    _state["vp_drag_line"] = (lx, ly1, ly2)
                    _state["vp_drag_current"] = (x, y)
                    return
            for handle in _layout.get("polygon_handle_zones", []):
                hx, hy = handle["center"]
                radius = handle["radius"]
                if (x - hx) * (x - hx) + (y - hy) * (y - hy) <= radius * radius:
                    _state["polygon_drag_idx"] = handle["idx"]
                    _state["polygon_drag_current"] = (x, y)
                    return
            # Video area: start drag (may become a click if motion < threshold)
            _state["drag_start"]   = (x, y)
            _state["drag_current"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if _state["vp_drag_start"] is not None:
            _state["vp_drag_current"] = (x, y)
            return
        if _state["polygon_drag_idx"] is not None:
            _state["polygon_drag_current"] = (x, y)
            return
        if _state["drag_start"] is not None:
            _state["drag_current"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if _state["vp_drag_start"] is not None and _state["vp_drag_line"] is not None:
            sx, sy = _state["vp_drag_start"]
            lx, ly1, ly2 = _state["vp_drag_line"]
            dx = x - sx
            dy = y - sy
            _state["vp_drag_end"] = (lx + dx, ly1 + dy, ly2 + dy)
            _state["vp_drag_start"] = None
            _state["vp_drag_line"] = None
            _state["vp_drag_current"] = None
            return
        if _state["polygon_drag_idx"] is not None:
            _state["polygon_drag_end"] = (_state["polygon_drag_idx"], x, y)
            _state["polygon_drag_idx"] = None
            _state["polygon_drag_current"] = None
            return
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


def build_person_path_points(track_dict):
    """Return sorted [(frame_num, x, bottom_y)] path points for supported modes."""
    pts = []
    for f in track_dict.get("frames", []):
        fmode = f.get("mode", track_dict.get("mode", DEFAULT_MODE))
        if fmode not in ("person", "vehicle-pos"):
            continue
        x1, y1, x2, y2 = f.get("box", [0, 0, 0, 0])
        if fmode == "vehicle-pos":
            px = float(x1)
        else:
            px = float((x1 + x2) / 2.0)
        pts.append((int(f.get("frame_num", 0)), px, float(y2)))
    pts.sort(key=lambda t: t[0])
    return pts


def _smooth_frame_path_points(frame_points):
    """Smooth a sorted list of (frame_num, x, y) points in frame space."""
    if savgol_filter is None or len(frame_points) < 3:
        return list(frame_points)

    sorted_points = sorted(frame_points, key=lambda t: t[0])
    segments = []
    segment = [sorted_points[0]]
    for point in sorted_points[1:]:
        if point[0] == segment[-1][0] + 1:
            segment.append(point)
        else:
            segments.append(segment)
            segment = [point]
    segments.append(segment)

    smoothed_points = []
    for segment in segments:
        if len(segment) < 3:
            smoothed_points.extend(segment)
            continue
        coords = [(point[1], point[2]) for point in segment]
        smoothed_coords = _smooth_xy_points_savgol(coords)
        smoothed_points.extend(
            (point[0], float(smooth[0]), float(smooth[1]))
            for point, smooth in zip(segment, smoothed_coords)
        )
    return smoothed_points


def _build_frame_point_lookup(track_dict, key, fallback_key=None):
    lookup = {}
    for frame_data in track_dict.get("frames", []):
        point = frame_data.get(key)
        if point is None and fallback_key is not None:
            point = frame_data.get(fallback_key)
        if point is None or len(point) < 2:
            continue
        lookup[int(frame_data["frame_num"])] = [float(point[0]), float(point[1])]
    return lookup


def _finalize_saved_track(track_dict):
    """Attach in-memory caches used by the live UI to a saved track dict."""
    track_dict["_frame_lookup"] = build_frame_lookup(track_dict)
    track_dict["_person_path_pts"] = build_person_path_points(track_dict)
    track_dict["_smooth_person_path_pts"] = _smooth_frame_path_points(
        track_dict["_person_path_pts"]
    )
    track_dict["_projected_point_lookup"] = _build_frame_point_lookup(track_dict, "projected_point")
    track_dict["_smooth_projected_point_lookup"] = _build_frame_point_lookup(
        track_dict, "smooth_projected_point", fallback_key="projected_point"
    )
    return track_dict


def _vehicle_line_len_from_box(box, frame_h):
    x1, y1, x2, y2 = box
    span = abs(int(y2) - int(y1))
    if span > 0:
        return span
    return max(18, int(frame_h * 0.12))


def _vehicle_pos_box_from_anchor(anchor, frame_shape, base_box=None):
    """Create a vertical-line box (x1==x2) from anchor bottom point in frame coords."""
    ax, ay = int(anchor[0]), int(anchor[1])
    fh, fw = frame_shape[:2]
    ax = max(0, min(ax, fw - 1))
    ay = max(0, min(ay, fh - 1))
    base_len = _vehicle_line_len_from_box(base_box, fh) if base_box is not None else max(18, int(fh * 0.12))
    y2 = ay
    y1 = max(0, y2 - base_len)
    return (ax, y1, ax, y2)


def _point_hits_track_box(fx, fy, box, mode, tol=6):
    x1, y1, x2, y2 = box
    if mode == "vehicle-pos":
        lx = int(round((x1 + x2) / 2.0))
        ly1, ly2 = (y1, y2) if y1 <= y2 else (y2, y1)
        return (abs(fx - lx) <= tol) and (ly1 - tol <= fy <= ly2 + tol)
    return x1 <= fx <= x2 and y1 <= fy <= y2


def _polygon_state_key(mode):
    if mode == 'person':
        return 'crosswalk position'
    if mode in ('vehicle', 'vehicle-pos'):
        return 'position'
    return None


def _refresh_saved_track_polygon(track_dict, polygon_points):
    poly_key = _polygon_state_key(track_dict.get("mode", DEFAULT_MODE))
    for frame_data in track_dict.get("frames", []):
        frame_mode = frame_data.get("mode", track_dict.get("mode", DEFAULT_MODE))
        poly_key = _polygon_state_key(frame_mode)
        if poly_key is None:
            continue
        merged_states = dict(track_dict.get("track_states", {}))
        merged_states.update(frame_data.get("states", {}))
        x1, y1, x2, y2 = frame_data.get("box", [0, 0, 0, 0])
        updated_states = _apply_polygon_crosswalk_state(
            frame_mode, merged_states, (x1, y1, x2, y2), polygon_points)
        frame_states = dict(frame_data.get("states", {}))
        if poly_key in updated_states:
            frame_states[poly_key] = updated_states[poly_key]
        frame_data["states"] = frame_states
    _finalize_saved_track(track_dict)


def _polygon_edge_lengths_from_points(polygon_points):
    if len(polygon_points) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    out = []
    for i in range(4):
        ax, ay = polygon_points[i]
        bx, by = polygon_points[(i + 1) % 4]
        out.append(float(np.hypot(float(bx - ax), float(by - ay))))
    return out


def _circle_intersections(c0, r0, c1, r1):
    x0, y0 = float(c0[0]), float(c0[1])
    x1, y1 = float(c1[0]), float(c1[1])
    r0 = float(r0)
    r1 = float(r1)
    dx = x1 - x0
    dy = y1 - y0
    d = float(np.hypot(dx, dy))
    if d < 1e-6:
        return []
    if d > r0 + r1 + 1e-6:
        return []
    if d < abs(r0 - r1) - 1e-6:
        return []
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h2 = r0 * r0 - a * a
    if h2 < 0:
        if h2 > -1e-6:
            h2 = 0.0
        else:
            return []
    h = float(np.sqrt(h2))
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry = dx * (h / d)
    p1 = np.array([xm + rx, ym + ry], dtype=np.float32)
    p2 = np.array([xm - rx, ym - ry], dtype=np.float32)
    if np.allclose(p1, p2, atol=1e-6):
        return [p1]
    return [p1, p2]


def _polygon_projection_geometry(edge_lengths, diag_lengths, rot_deg=0.0,
                                 flip=False, reflect_h=False, reflect_v=False):
    """Build quadrilateral from 4 sides + 2 diagonals, centered at origin."""
    if not edge_lengths or len(edge_lengths) != 4:
        return None
    if not diag_lengths or len(diag_lengths) != 2:
        return None
    try:
        s1, s2, s3, s4 = [max(1e-3, float(v)) for v in edge_lengths]
        d1, d2 = [max(1e-3, float(v)) for v in diag_lengths]
    except Exception:
        return None

    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([s1, 0.0], dtype=np.float32)

    cands_c = _circle_intersections(a, d1, b, s2)
    if not cands_c:
        return None

    quads = []
    for c in cands_c:
        cands_d = _circle_intersections(a, s4, c, s3)
        for d in cands_d:
            err = abs(float(np.hypot(*(d - b))) - d2)
            quads.append((err, np.array([a, b, c, d], dtype=np.float32)))
    if not quads:
        return None
    quads.sort(key=lambda t: t[0])
    quad = quads[1][1] if (flip and len(quads) > 1) else quads[0][1]

    quad = quad - quad.mean(axis=0)
    if reflect_h:
        quad[:, 0] *= -1.0
    if reflect_v:
        quad[:, 1] *= -1.0

    rot_rad = float(np.deg2rad(rot_deg))
    ca = float(np.cos(rot_rad))
    sa = float(np.sin(rot_rad))
    rmat = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    quad = (quad @ rmat.T).astype(np.float32)
    return quad


def _project_person_midpoints(track_history, current_frame, saved_tracks, previewed_idxs=None):
    traces = []
    supported_modes = ("person", "vehicle-pos")

    # Only keep projected trace while the corresponding supported track is visible now.
    if current_frame in track_history and track_history[current_frame][5] in supported_modes:
        cur_pts = []
        for fn in sorted(track_history.keys()):
            if fn > current_frame:
                continue
            x1, y1, x2, y2, _, mode, _ = track_history[fn]
            if mode in supported_modes:
                if mode == "vehicle-pos":
                    cur_pts.append((float(x1), float(y2)))
                else:
                    cur_pts.append((float((x1 + x2) / 2.0), float(y2)))
        if cur_pts:
            traces.append(cur_pts)

    idxs = sorted(previewed_idxs) if previewed_idxs is not None else range(len(saved_tracks))
    for i in idxs:
        if i < 0 or i >= len(saved_tracks):
            continue
        tr = saved_tracks[i]
        lu = tr.get("_frame_lookup", {})
        if current_frame not in lu or lu[current_frame][5] not in supported_modes:
            continue
        t_pts = []
        for fn in sorted(lu.keys()):
            if fn > current_frame:
                continue
            x1, y1, x2, y2, _, mode, _ = lu[fn]
            if mode not in supported_modes:
                continue
            if mode == "vehicle-pos":
                t_pts.append((float(x1), float(y2)))
            else:
                t_pts.append((float((x1 + x2) / 2.0), float(y2)))
        if t_pts:
            traces.append(t_pts)
    return traces


def _compute_track_projection(track_dict, polygon_points, edge_lengths, diag_lengths, 
                               rot_deg, flip, reflect_h, reflect_v):
    """Compute projected traces for a saved track in meters.
    
    Returns:
        dict with keys: "polygon_projection" (metadata), "projected_traces" (list of traces)
        or None if projection cannot be computed.
    """
    if not polygon_points or len(polygon_points) < 4:
        return None
    
    geom = _polygon_projection_geometry(
        edge_lengths, diag_lengths,
        rot_deg=rot_deg, flip=flip, reflect_h=reflect_h, reflect_v=reflect_v,
    )
    if geom is None:
        return None
    
    src = np.array(polygon_points[:4], dtype=np.float32)
    dst = geom.astype(np.float32)
    try:
        hmat = cv2.getPerspectiveTransform(src, dst)
    except Exception:
        return None
    
    # Extract supported traces (person + vehicle-pos) from track
    frame_lookup = track_dict.get("_frame_lookup", {})
    traces = []
    trace_pts = []
    supported_modes = ("person", "vehicle-pos")
    for fn in sorted(frame_lookup.keys()):
        x1, y1, x2, y2, _, mode, _ = frame_lookup[fn]
        if mode in supported_modes:
            if mode == "vehicle-pos":
                trace_pts.append((float(x1), float(y2)))
            else:
                trace_pts.append((float((x1 + x2) / 2.0), float(y2)))
        else:
            if trace_pts:
                if len(trace_pts) > 0:
                    arr = np.array(trace_pts, dtype=np.float32).reshape(-1, 1, 2)
                    proj = cv2.perspectiveTransform(arr, hmat).reshape(-1, 2)
                    traces.append([tuple(map(float, p)) for p in proj])
                trace_pts = []
    # Don't forget the last trace
    if trace_pts:
        arr = np.array(trace_pts, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(arr, hmat).reshape(-1, 2)
        traces.append([tuple(map(float, p)) for p in proj])
    
    return {
        "polygon_projection": {
            "polygon_vertices": [tuple(map(float, p)) for p in dst],
            "edge_lengths_meters": edge_lengths,
            "diagonal_lengths_meters": diag_lengths,
            "rotation_degrees": rot_deg,
            "flip": flip,
            "reflect_horizontal": reflect_h,
            "reflect_vertical": reflect_v,
            "origin_note": "center of polygon at (0, 0)",
        },
        "projected_traces": traces,
    }


def _compute_projected_frame_points(frame_lookup, polygon_points, edge_lengths, diag_lengths,
                                    rot_deg, flip, reflect_h, reflect_v,
                                    smooth_trajectories=False):
    """Project raw and smoothed frame-space paths into the normalized polygon space.

    Returns (projected_polygon, projected_points_by_frame, smooth_projected_points_by_frame).
    Both point maps are keyed by frame number and store [x, y] in projected space.
    """
    if not polygon_points or len(polygon_points) < 4:
        return None, {}, {}

    geom = _polygon_projection_geometry(
        edge_lengths,
        diag_lengths,
        rot_deg=rot_deg,
        flip=flip,
        reflect_h=reflect_h,
        reflect_v=reflect_v,
    )
    if geom is None:
        return None, {}, {}

    src = np.array(polygon_points[:4], dtype=np.float32)
    dst = geom.astype(np.float32)
    try:
        hmat = cv2.getPerspectiveTransform(src, dst)
    except Exception:
        return None, {}, {}

    raw_points = []
    for fn in sorted(frame_lookup.keys()):
        x1, y1, x2, y2, _, mode, _ = frame_lookup[fn]
        if mode not in ("person", "vehicle-pos"):
            continue
        px = float(x1) if mode == "vehicle-pos" else float(x1 + x2) / 2.0
        raw_points.append((int(fn), px, float(y2)))

    smooth_points = _smooth_frame_path_points(raw_points)

    def _project_path(path_points):
        projected_points = {}
        for fn, px, py in path_points:
            arr = np.array([[(px), float(py)]], dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(arr, hmat).reshape(-1, 2)[0]
            projected_points[int(fn)] = [float(proj[0]), float(proj[1])]
        return projected_points

    raw_projected_points = _project_path(raw_points)
    smooth_projected_points = _project_path(smooth_points)

    return [tuple(map(float, p)) for p in dst], raw_projected_points, smooth_projected_points


def _compute_projection_overlay_data(polygon_points, edge_lengths, diag_lengths, rot_deg,
                                     flip, reflect_h, reflect_v,
                                     track_history, current_frame, saved_tracks,
                                     previewed_idxs, smooth_trajectories=False):
    geom = _polygon_projection_geometry(
        edge_lengths,
        diag_lengths,
        rot_deg=rot_deg,
        flip=flip,
        reflect_h=reflect_h,
        reflect_v=reflect_v,
    )
    if geom is None:
        return None
    src = np.array(polygon_points, dtype=np.float32)
    dst = geom.astype(np.float32)
    try:
        hmat = cv2.getPerspectiveTransform(src, dst)
    except Exception:
        return None

    out_traces = []
    live_traces = _project_person_midpoints(track_history, current_frame, [], previewed_idxs=None)
    for pts in live_traces:
        if smooth_trajectories:
            pts = _smooth_xy_points_savgol(pts)
        arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(arr, hmat).reshape(-1, 2)
        out_traces.append([tuple(map(float, p)) for p in proj])

    for i in sorted(previewed_idxs):
        if i < 0 or i >= len(saved_tracks):
            continue
        track = saved_tracks[i]
        if current_frame not in track.get("_frame_lookup", {}):
            continue
        frame_lookup = track.get("_frame_lookup", {})
        _, _, _, _, _, mode, _ = frame_lookup[current_frame]
        if mode not in ("person", "vehicle-pos"):
            continue
        path_key = "_smooth_person_path_pts" if smooth_trajectories else "_person_path_pts"
        frame_pts = [
            (float(px), float(py))
            for (fn, px, py) in track.get(path_key, [])
            if fn <= current_frame
        ]
        pts = []
        if frame_pts:
            arr = np.array(frame_pts, dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(arr, hmat).reshape(-1, 2)
            pts = [tuple(map(float, p)) for p in proj]
        if pts:
            out_traces.append(pts)

    return {
        "poly": [tuple(map(float, p)) for p in dst],
        "traces": out_traces,
    }


def _serialize_track_dict(track_dict):
    out = {}
    for key, value in track_dict.items():
        if key.startswith("_"):
            continue
        if key == "projected_track":
            continue
        if key == "frames":
            out[key] = [dict(frame) for frame in value]
        elif isinstance(value, dict):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def _read_projection_settings(raw_doc):
    """Read projection transform settings from JSON with backward-compatible fallbacks."""
    settings = {}
    if isinstance(raw_doc, dict):
        settings = raw_doc.get("projection_settings", {}) or {}

    def _pick_bool(new_key, legacy_key, default):
        if isinstance(settings, dict) and new_key in settings:
            return bool(settings.get(new_key))
        if isinstance(raw_doc, dict) and legacy_key in raw_doc:
            return bool(raw_doc.get(legacy_key))
        return bool(default)

    def _pick_float(new_key, legacy_key, default):
        if isinstance(settings, dict) and new_key in settings:
            try:
                return float(settings.get(new_key))
            except Exception:
                pass
        if isinstance(raw_doc, dict) and legacy_key in raw_doc:
            try:
                return float(raw_doc.get(legacy_key))
            except Exception:
                pass
        return float(default)

    return {
        "rotation_degrees": _pick_float("rotation_degrees", "projection_rotation_deg", 0.0),
        "flip": _pick_bool("flip", "projection_flip", False),
        "reflect_horizontal": _pick_bool("reflect_horizontal", "projection_reflect_h", False),
        "reflect_vertical": _pick_bool("reflect_vertical", "projection_reflect_v", False),
    }


def save_all_annotations_to_json(video_path, saved_tracks, markers, polygon=None,
                                 polygon_edge_lengths=None,
                                 polygon_diagonal_lengths=None,
                                 projected_polygon=None,
                                 projection_rot_deg=None,
                                 projection_flip=None,
                                 projection_reflect_h=None,
                                 projection_reflect_v=None):
    outpath = _markers_path(video_path)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    old_projected_polygon = []
    old_projection_settings = _read_projection_settings({})
    if os.path.isfile(outpath):
        try:
            with open(outpath, "r") as f:
                _raw = json.load(f)
            if isinstance(_raw, dict):
                old_projected_polygon = _raw.get("projected_polygon", [])
                old_projection_settings = _read_projection_settings(_raw)
        except Exception:
            old_projected_polygon = []
            old_projection_settings = _read_projection_settings({})
    projection_settings = {
        "rotation_degrees": (
            float(projection_rot_deg)
            if projection_rot_deg is not None else float(old_projection_settings.get("rotation_degrees", 0.0))
        ),
        "flip": (
            bool(projection_flip)
            if projection_flip is not None else bool(old_projection_settings.get("flip", False))
        ),
        "reflect_horizontal": (
            bool(projection_reflect_h)
            if projection_reflect_h is not None else bool(old_projection_settings.get("reflect_horizontal", False))
        ),
        "reflect_vertical": (
            bool(projection_reflect_v)
            if projection_reflect_v is not None else bool(old_projection_settings.get("reflect_vertical", False))
        ),
    }
    out_doc = {
        "tracks": [_serialize_track_dict(track) for track in saved_tracks],
        "markers": markers,
        "polygon": polygon or [],
        "polygon_edge_lengths": polygon_edge_lengths or [],
        "polygon_diagonal_lengths": polygon_diagonal_lengths or [],
        "projected_polygon": projected_polygon if projected_polygon is not None else old_projected_polygon,
        "projection_settings": projection_settings,
    }
    with open(outpath, "w") as f:
        json.dump(out_doc, f, indent=2)


def draw_smooth_polyline(canvas, frame_pts, x_off, y_off, scale, color, thickness=2):
    """Draw anti-aliased smoothed polyline through frame-space points."""
    if len(frame_pts) < 2:
        return
    arr = np.array([
        [x_off + p[0] * scale, y_off + p[1] * scale]
        for p in frame_pts
    ], dtype=np.float32)

    # Moving-average smoothing to reduce jitter from bbox edges.
    if len(arr) >= 3:
        sm = arr.copy()
        rad = 2
        for i in range(len(arr)):
            a = max(0, i - rad)
            b = min(len(arr), i + rad + 1)
            sm[i] = arr[a:b].mean(axis=0)
        arr = sm

    cv2.polylines(canvas, [arr.astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def _savgol_window(base_window, n_points):
    """Return valid odd window length for savgol_filter or 0 when unavailable."""
    if n_points <= 2:
        return 0
    w = min(int(base_window), int(n_points))
    if w < 3:
        return 0
    if w % 2 == 0:
        w -= 1
    return w if w >= 3 else 0


def _smooth_xy_points_savgol(points):
    """Smooth list of (x, y) points using Savitzky-Golay on each axis."""
    if savgol_filter is None or len(points) < 3:
        return list(points)
    arr = np.array(points, dtype=np.float32)
    wx = _savgol_window(50, len(arr))
    wy = _savgol_window(50, len(arr))
    if wx == 0 or wy == 0:
        return list(points)
    try:
        sx = savgol_filter(arr[:, 0], window_length=wx, polyorder=1, delta=10)
        sy = savgol_filter(arr[:, 1], window_length=wy, polyorder=1, delta=10)
        out = np.stack([sx, sy], axis=1)
        return [tuple(map(float, p)) for p in out]
    except Exception:
        return list(points)


def _draw_trajectory_polyline(canvas, frame_pts, x_off, y_off, scale, color,
                              smooth_enabled=True, thickness=2):
    """Draw trajectory line in frame space from precomputed points."""
    if len(frame_pts) < 2:
        return
    draw_pts = list(frame_pts)
    arr = np.array([
        [x_off + p[0] * scale, y_off + p[1] * scale]
        for p in draw_pts
    ], dtype=np.float32)
    cv2.polylines(canvas, [arr.astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def _draw_projection_overlay(canvas, overlay_data, edge_colors, alpha=0.75,
                             smooth_trajectories=False):
    if overlay_data is None:
        return
    poly = overlay_data.get("poly", [])
    traces = overlay_data.get("traces", [])
    if len(poly) != 4:
        return

    h, w = canvas.shape[:2]
    x1 = max(SIDEBAR_W + 12, int(w * 0.58))
    x2 = w - 12
    y1 = 12
    y2 = int(h * 0.55)
    if x2 - x1 < 120 or y2 - y1 < 120:
        return

    panel = canvas.copy()
    cv2.rectangle(panel, (x1, y1), (x2, y2), (12, 12, 12), -1)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (70, 70, 70), 1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.line(panel, (x1 + 10, cy), (x2 - 10, cy), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(panel, (cx, y1 + 10), (cx, y2 - 10), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(panel, "X", (x2 - 20, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, "Y", (cx + 6, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)

    arr = np.array(poly, dtype=np.float32)
    max_x = max(float(np.max(np.abs(arr[:, 0]))), 1e-3)
    max_y = max(float(np.max(np.abs(arr[:, 1]))), 1e-3)
    fit_x = (x2 - x1 - 30) / (2.0 * max_x)
    fit_y = (y2 - y1 - 30) / (2.0 * max_y)
    sf = max(1e-6, min(fit_x, fit_y))

    def _to_panel(pt):
        px = int(round(cx + float(pt[0]) * sf))
        py = int(round(cy - float(pt[1]) * sf))
        return px, py

    poly_canvas = [_to_panel(p) for p in poly]
    for i in range(4):
        a = poly_canvas[i]
        b = poly_canvas[(i + 1) % 4]
        col = edge_colors[i % len(edge_colors)]
        cv2.line(panel, a, b, col, 2, cv2.LINE_AA)
    cv2.line(panel, poly_canvas[0], poly_canvas[2], POLY_DIAG_COLORS[0], 1, cv2.LINE_AA)
    cv2.line(panel, poly_canvas[1], poly_canvas[3], POLY_DIAG_COLORS[1], 1, cv2.LINE_AA)

    for idx, tr in enumerate(traces):
        if len(tr) < 2:
            continue
        tr_arr = np.array([_to_panel(p) for p in tr], dtype=np.int32)
        col = PREVIEW_COLORS[idx % len(PREVIEW_COLORS)]
        cv2.polylines(panel, [tr_arr], False, col, 2, cv2.LINE_AA)

    cv2.addWeighted(panel, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)


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
    preview_boxes – saved-track boxes:   (x1,y1,x2,y2,label,color,ann_lines,traj_points).
    canvas_meta   – dict with display flags.
    """
    if canvas_meta is None:
        canvas_meta = {}
    if preview_boxes is None:
        preview_boxes = []
    smooth_trajectories = bool(canvas_meta.get("smooth_trajectories", False))
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

    # ---- Polygon edge lengths ----
    polygon_edge_lengths = canvas_meta.get("polygon_edge_lengths", [0.0, 0.0, 0.0, 0.0])
    polygon_diag_lengths = canvas_meta.get("polygon_diag_lengths", [0.0, 0.0])
    polygon_len_edit_idx = canvas_meta.get("polygon_len_edit_idx")
    polygon_len_edit_text = canvas_meta.get("polygon_len_edit_text", "")
    polygon_length_zones = []

    sb_line("POLY LENGTHS", F_HDR, (160, 210, 255), bold=True, gap=5)
    poly_fields = [
        ("L1", 0, "edge", POLY_EDGE_COLORS[0]),
        ("L2", 1, "edge", POLY_EDGE_COLORS[1]),
        ("L3", 2, "edge", POLY_EDGE_COLORS[2]),
        ("L4", 3, "edge", POLY_EDGE_COLORS[3]),
        ("D1", 0, "diag", POLY_DIAG_COLORS[0]),
        ("D2", 1, "diag", POLY_DIAG_COLORS[1]),
    ]
    for i, (tag, val_idx, kind, col) in enumerate(poly_fields):
        ly1 = sy
        ly2 = sy + MRK_ROW_H
        cv2.rectangle(canvas, (PAD, ly1 + 4), (PAD + 10, ly2 - 4), col, -1)
        field_x1 = PAD + 16
        field_x2 = SIDEBAR_W - PAD - 2
        bg = (50, 60, 75) if i == polygon_len_edit_idx else (34, 34, 34)
        bd = (95, 125, 170) if i == polygon_len_edit_idx else (70, 70, 70)
        cv2.rectangle(canvas, (field_x1, ly1 + 2), (field_x2, ly2 - 2), bg, -1)
        cv2.rectangle(canvas, (field_x1, ly1 + 2), (field_x2, ly2 - 2), bd, 1)
        if i == polygon_len_edit_idx:
            val_txt = polygon_len_edit_text if polygon_len_edit_text else "0"
        else:
            src = polygon_edge_lengths[val_idx] if kind == "edge" else polygon_diag_lengths[val_idx]
            val_txt = f"{float(src):.2f}"
        cv2.putText(canvas, f"{tag}: {val_txt}",
                    (field_x1 + 4, _cy(ly1, MRK_ROW_H, F_ROW)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_ROW, (210, 210, 210), 1, cv2.LINE_AA)
        polygon_length_zones.append({
            "type": "poly_len_field",
            "idx": i,
            "kind": kind,
            "val_idx": val_idx,
            "x1": field_x1,
            "x2": field_x2,
            "y1": ly1,
            "y2": ly2,
        })
        sy += MRK_ROW_H

    save_y1 = sy
    save_y2 = sy + BTN_H
    cv2.rectangle(canvas, (1, save_y1), (SIDEBAR_W - 1, save_y2), (35, 60, 35), -1)
    cv2.rectangle(canvas, (1, save_y1), (SIDEBAR_W - 1, save_y2), (80, 130, 80), 1)
    cv2.putText(canvas, "save lengths", (PAD, _cy(save_y1, BTN_H, F_BTN)),
                cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (130, 220, 130), 1, cv2.LINE_AA)
    polygon_length_zones.append({
        "type": "poly_len_save",
        "x1": 1,
        "x2": SIDEBAR_W - 1,
        "y1": save_y1,
        "y2": save_y2,
    })
    sy += BTN_H + 4
    
    # Save all projections button
    proj_y1 = sy
    proj_y2 = sy + BTN_H
    cv2.rectangle(canvas, (1, proj_y1), (SIDEBAR_W - 1, proj_y2), (40, 50, 70), -1)
    cv2.rectangle(canvas, (1, proj_y1), (SIDEBAR_W - 1, proj_y2), (100, 130, 180), 1)
    cv2.putText(canvas, "save all proj", (PAD, _cy(proj_y1, BTN_H, F_BTN)),
                cv2.FONT_HERSHEY_SIMPLEX, F_BTN, (150, 190, 255), 1, cv2.LINE_AA)
    polygon_length_zones.append({
        "type": "poly_save_all_proj",
        "x1": 1,
        "x2": SIDEBAR_W - 1,
        "y1": proj_y1,
        "y2": proj_y2,
    })
    sy += BTN_H + 4
    
    _layout["polygon_length_zones"] = polygon_length_zones
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
        ("[S] smooth traj",      (120, 220, 190)),
        ("[P] projection",       (160, 210, 255)),
        ("[O] rotate projection",(160, 210, 255)),
        ("[F] flip solution",    (160, 210, 255)),
        ("[H] reflect H",        (160, 210, 255)),
        ("[v] reflect V",        (160, 210, 255)),
        ("[V] vehicle-pos",      (140, 180, 220)),
        ("[click] detect",       (140, 140, 140)),
        ("[drag] draw box",      (210, 210, 50)),
        ("[<][>] frame step",    (140, 140, 140)),
        ("[SPACE] play/pause",   (140, 140, 140)),
        ("[Z] 4x speed",         (140, 140, 140)),
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

    # --- crosswalk polygon (always shown when present) ---
    polygon_points = canvas_meta.get("polygon_points", [])
    polygon_drag_idx = canvas_meta.get("polygon_drag_idx")
    polygon_handle_zones = []
    if polygon_points:
        p_canvas = [(int(x_off + px * scale), int(y_off + py * scale))
                    for (px, py) in polygon_points]
        if len(p_canvas) >= 2:
            for i in range(len(p_canvas) - 1):
                seg_col = POLY_EDGE_COLORS[i % len(POLY_EDGE_COLORS)] if len(p_canvas) == 4 else POLY_TMP_COLOR
                cv2.line(canvas, p_canvas[i], p_canvas[i + 1], seg_col, 2, cv2.LINE_AA)
            if len(p_canvas) == 4:
                cv2.line(canvas, p_canvas[-1], p_canvas[0], POLY_EDGE_COLORS[3], 2, cv2.LINE_AA)
                cv2.line(canvas, p_canvas[0], p_canvas[2], POLY_DIAG_COLORS[0], 1, cv2.LINE_AA)
                cv2.line(canvas, p_canvas[1], p_canvas[3], POLY_DIAG_COLORS[1], 1, cv2.LINE_AA)
        for idx, p in enumerate(p_canvas):
            radius = 8 if idx == polygon_drag_idx else 6
            fill = (0, 200, 255) if idx == polygon_drag_idx else POLY_COLOR
            cv2.circle(canvas, p, radius, fill, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, radius + 2, (20, 20, 20), 1, cv2.LINE_AA)
            polygon_handle_zones.append({"idx": idx, "center": p, "radius": 12})
    _layout["polygon_handle_zones"] = polygon_handle_zones

    if canvas_meta.get("polygon_setup", False):
        hint = "Crosswalk setup: click 4 points, drag points to adjust, Enter=confirm, Backspace=clear"
        cv2.putText(canvas, hint, (SIDEBAR_W + 10, max(22, y_off + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)
    elif polygon_points:
        hint = "Drag polygon points to adjust crosswalk"
        cv2.putText(canvas, hint, (SIDEBAR_W + 10, max(22, y_off + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

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

    def _draw_vehicle_pos_line(sx, sy1, sy2, col, lbl, ann_lines=None):
        ly1, ly2 = (sy1, sy2) if sy1 <= sy2 else (sy2, sy1)
        cv2.line(canvas, (sx, ly1), (sx, ly2), col, 2, cv2.LINE_AA)
        cv2.circle(canvas, (sx, ly2), 4, col, -1, cv2.LINE_AA)
        label_y = max(ly1 - 6, y_off + 14)
        cv2.putText(canvas, lbl, (sx + 6, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
        if ann_lines:
            _draw_ann_overlay(ann_lines, sx + 2, label_y + 18, 0.40, 15)
        return (sx, ly1, ly2)

    vehicle_line_zone = None

    for (x1, y1, x2, y2, lbl, col) in boxes:
        sx1 = int(x_off + x1 * scale)
        sy1 = int(y_off + y1 * scale)
        sx2 = int(x_off + x2 * scale)
        sy2 = int(y_off + y2 * scale)
        is_vehicle_pos = (annotation is not None and annotation.get("mode") == "vehicle-pos") or (lbl == "vehicle-pos")
        if is_vehicle_pos:
            ann_lines = None
            if annotation and col != DIM_COLOR and not show_popup:
                ann_lines = [(annotation["mode"], "mode")]
                for entry in _ordered_state_cats(annotation["mode"]):
                    cat = entry[0]
                    if cat in annotation["states"]:
                        ann_lines.append((str(annotation["states"][cat]),
                                          entry[2] if len(entry) > 2 else 'per_frame'))
            zone = _draw_vehicle_pos_line(sx1, sy1, sy2, col, lbl, ann_lines)
            if col in (TRACK_COLOR, HISTORY_COLOR):
                vehicle_line_zone = {"x": zone[0], "y1": zone[1], "y2": zone[2], "tol": 10}
        else:
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
    for (x1, y1, x2, y2, lbl, col, ann_lines, traj_points, pmode) in preview_boxes:
        sx1 = int(x_off + x1 * scale)
        sy1 = int(y_off + y1 * scale)
        sx2 = int(x_off + x2 * scale)
        sy2 = int(y_off + y2 * scale)
        if traj_points:
            _draw_trajectory_polyline(
                canvas, traj_points, x_off, y_off, scale, col,
                smooth_enabled=smooth_trajectories, thickness=2)
        if pmode == "vehicle-pos":
            _draw_vehicle_pos_line(sx1, sy1, sy2, col, lbl, ann_lines)
        else:
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

    vp_cursor_line = canvas_meta.get("vehicle_cursor_line")
    if vp_cursor_line is not None:
        vx, vy1, vy2 = vp_cursor_line
        cv2.line(canvas, (vx, vy1), (vx, vy2), (0, 235, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (vx, vy2), 3, (0, 235, 255), -1, cv2.LINE_AA)

    vp_drag_line = canvas_meta.get("vehicle_drag_line")
    if vp_drag_line is not None:
        vx, vy1, vy2 = vp_drag_line
        cv2.line(canvas, (vx, vy1), (vx, vy2), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (vx, vy2), 4, (255, 255, 0), -1, cv2.LINE_AA)

    # Draw popup (shown only on the click frame while paused)
    if show_popup and annotation and popup_anchor is not None:
        popup_zones = draw_popup(
            canvas, popup_anchor[0], popup_anchor[1],
            annotation["mode"], annotation["states"],
            screen_w, screen_h,
        )
    _layout["popup_zones"] = popup_zones
    _layout["vehicle_pos_line_zone"] = vehicle_line_zone

    if canvas_meta.get("show_projection_overlay", False):
        _draw_projection_overlay(
            canvas,
            canvas_meta.get("projection_overlay_data"),
            POLY_EDGE_COLORS,
            alpha=0.78,
            smooth_trajectories=smooth_trajectories,
        )

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
    video_abs = os.path.abspath(video_path)
    video_dir = os.path.dirname(video_abs)
    base_name = os.path.splitext(os.path.basename(video_abs))[0]
    # Preferred location: next to video file, same base name.
    primary = os.path.join(video_dir, f"{base_name}.json")
    # Backward-compatible legacy location used by older versions.
    legacy = os.path.join(video_dir, base_name, f"{base_name}.json")
    if os.path.isfile(primary):
        return primary
    if os.path.isfile(legacy):
        return legacy
    return primary


def _point_in_polygon(pt, polygon_points):
    if len(polygon_points) < 3:
        return False
    poly_np = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0


def _apply_polygon_crosswalk_state(mode, states, box, polygon_points):
    """Set crosswalk per-frame states from bbox bottom corners and polygon."""
    out = dict(states)
    if len(polygon_points) < 3:
        return out
    x1, y1, x2, y2 = box
    left_in = _point_in_polygon((x1, y2), polygon_points)
    right_in = _point_in_polygon((x2, y2), polygon_points)
    if mode == 'person' and 'crosswalk position' in out:
        if left_in and right_in:
            out['crosswalk position'] = 'inside crosswalk'
        elif (not left_in) and (not right_in):
            out['crosswalk position'] = 'outside crosswalk'
    elif mode == 'vehicle' and 'position' in out:
        if left_in and right_in:
            out['position'] = 'inside crosswalk'
        elif (not left_in) and (not right_in):
            out['position'] = 'outside crosswalk'
    return out


def _apply_polygon_to_history(track_history, polygon_points):
    if len(polygon_points) < 3:
        return
    for fn in list(track_history.keys()):
        x1, y1, x2, y2, lbl, mode, states = track_history[fn]
        states = _apply_polygon_crosswalk_state(mode, states, (x1, y1, x2, y2), polygon_points)
        track_history[fn] = (x1, y1, x2, y2, lbl, mode, states)


def save_markers_to_json(video_path, markers, polygon=None):
    """Persist only the markers list into the JSON file without touching saved tracks."""
    outpath = _markers_path(video_path)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if os.path.isfile(outpath):
        with open(outpath, "r") as f:
            _raw = json.load(f)
        if isinstance(_raw, dict):
            all_tracks = _raw.get("tracks", [])
            old_polygon = _raw.get("polygon", [])
            old_lengths = _raw.get("polygon_edge_lengths", [])
            old_diags = _raw.get("polygon_diagonal_lengths", [])
            old_projection_settings = _read_projection_settings(_raw)
        else:
            all_tracks = _raw
            old_polygon = []
            old_lengths = []
            old_diags = []
            old_projection_settings = _read_projection_settings({})
    else:
        all_tracks = []
        old_polygon = []
        old_lengths = []
        old_diags = []
        old_projection_settings = _read_projection_settings({})
    out_doc = {
        "tracks": all_tracks,
        "markers": markers,
        "polygon": polygon if polygon is not None else old_polygon,
        "polygon_edge_lengths": old_lengths,
        "polygon_diagonal_lengths": old_diags,
        "projection_settings": old_projection_settings,
    }
    with open(outpath, "w") as f:
        json.dump(out_doc, f, indent=2)


def save_track_to_json(video_path, track_history, fps, markers=None, polygon=None,
                       polygon_edge_lengths=None, polygon_diagonal_lengths=None,
                       projection_rot_deg=None, projection_flip=None, 
                       projection_reflect_h=None, projection_reflect_v=None,
                       smooth_trajectories=False):
    """Append the current track (with per-frame mode+states) to the JSON file.
    Saves markers alongside tracks. Optionally includes projected track data in meters.
    Returns the saved track dict.
    """
    outpath = _markers_path(video_path)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    if os.path.isfile(outpath):
        with open(outpath, "r") as f:
            _raw = json.load(f)
        if isinstance(_raw, dict):
            all_tracks = _raw.get("tracks", [])
            old_polygon = _raw.get("polygon", [])
            old_lengths = _raw.get("polygon_edge_lengths", [])
            old_diags = _raw.get("polygon_diagonal_lengths", [])
            old_projected_polygon = _raw.get("projected_polygon", [])
            old_projection_settings = _read_projection_settings(_raw)
        else:
            all_tracks = _raw   # migrate old format
            old_polygon = []
            old_lengths = []
            old_diags = []
            old_projected_polygon = []
            old_projection_settings = _read_projection_settings({})
    else:
        all_tracks = []
        old_polygon = []
        old_lengths = []
        old_diags = []
        old_projected_polygon = []
        old_projection_settings = _read_projection_settings({})

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
    
    projected_polygon = None
    use_proj = (polygon is not None and len(polygon) >= 4 and 
                polygon_edge_lengths and len(polygon_edge_lengths) >= 4 and
                polygon_diagonal_lengths and len(polygon_diagonal_lengths) >= 2 and
                projection_rot_deg is not None and projection_flip is not None and
                projection_reflect_h is not None and projection_reflect_v is not None)
    
    if use_proj:
        frame_lookup = {}
        for frame_data in frames_data:
            fn = frame_data["frame_num"]
            x1, y1, x2, y2, label, fmode, fstates = track_history[fn]
            frame_lookup[fn] = (x1, y1, x2, y2, label, fmode, fstates)

        projected_polygon, projected_points, smooth_projected_points = _compute_projected_frame_points(
            frame_lookup,
            polygon,
            polygon_edge_lengths[:4],
            polygon_diagonal_lengths[:2],
            projection_rot_deg,
            projection_flip,
            projection_reflect_h,
            projection_reflect_v,
        )
        if projected_points or smooth_projected_points:
            for frame_data in frames_data:
                fn = frame_data["frame_num"]
                proj_point = projected_points.get(fn)
                if proj_point is not None:
                    frame_data["projected_point"] = proj_point
                smooth_proj_point = smooth_projected_points.get(fn)
                if smooth_proj_point is not None:
                    frame_data["smooth_projected_point"] = smooth_proj_point

    track_dict = _finalize_saved_track(track_dict)
    
    all_tracks.append(track_dict)

    out_doc = {
        "tracks": all_tracks,
        "markers": markers or [],
        "polygon": polygon if polygon is not None else old_polygon,
        "polygon_edge_lengths": (
            polygon_edge_lengths
            if polygon_edge_lengths is not None else old_lengths
        ),
        "polygon_diagonal_lengths": (
            polygon_diagonal_lengths
            if polygon_diagonal_lengths is not None else old_diags
        ),
        "projected_polygon": projected_polygon if projected_polygon is not None else old_projected_polygon,
        "projection_settings": {
            "rotation_degrees": (
                float(projection_rot_deg)
                if projection_rot_deg is not None else float(old_projection_settings.get("rotation_degrees", 0.0))
            ),
            "flip": (
                bool(projection_flip)
                if projection_flip is not None else bool(old_projection_settings.get("flip", False))
            ),
            "reflect_horizontal": (
                bool(projection_reflect_h)
                if projection_reflect_h is not None else bool(old_projection_settings.get("reflect_horizontal", False))
            ),
            "reflect_vertical": (
                bool(projection_reflect_v)
                if projection_reflect_v is not None else bool(old_projection_settings.get("reflect_vertical", False))
            ),
        },
    }
    with open(outpath, "w") as f:
        json.dump(out_doc, f, indent=2)

    print(f"[save] {len(frames_data)} frames  →  {outpath}")
    return track_dict


def apply_projections_to_all_tracks(video_path, saved_tracks, polygon, polygon_edge_lengths,
                                    polygon_diagonal_lengths, rot_deg, flip, reflect_h, reflect_v,
                                    smooth_trajectories=False):
    """Apply/update projections for all saved tracks and persist to JSON.
    
    Retroactively computes projected track data for all saved tracks using current projection
    parameters, and saves them to the JSON file.
    
    Returns: True if successful, False otherwise.
    """
    if not polygon or len(polygon) < 4:
        print("[proj] Invalid polygon, cannot apply projections")
        return False
    
    if not (polygon_edge_lengths and len(polygon_edge_lengths) >= 4 and
            polygon_diagonal_lengths and len(polygon_diagonal_lengths) >= 2):
        print("[proj] Missing or incomplete edge/diagonal lengths")
        return False
    
    outpath = _markers_path(video_path)
    try:
        with open(outpath) as f:
            doc = json.load(f)
    except Exception as e:
        print(f"[proj] Failed to read JSON: {e}")
        return False
    
    all_tracks = doc.get("tracks", [])
    projected_polygon = None
    updated_count = 0
    
    for i, track in enumerate(all_tracks):
        frame_lookup = {}
        for fd in track.get("frames", []):
            fn = fd["frame_num"]
            x1, y1, x2, y2 = fd["box"]
            label = fd.get("label", "")
            mode = fd.get("mode", "")
            states = fd.get("states", {})
            frame_lookup[fn] = (x1, y1, x2, y2, label, mode, states)

        projected_polygon, projected_points, smooth_projected_points = _compute_projected_frame_points(
            frame_lookup,
            polygon,
            polygon_edge_lengths[:4],
            polygon_diagonal_lengths[:2],
            rot_deg,
            flip,
            reflect_h,
            reflect_v,
        )

        for fd in track.get("frames", []):
            if "projected_point" in fd:
                del fd["projected_point"]
            if "smooth_projected_point" in fd:
                del fd["smooth_projected_point"]
        if projected_points or smooth_projected_points:
            for fd in track.get("frames", []):
                fn = fd["frame_num"]
                proj_point = projected_points.get(fn)
                if proj_point is not None:
                    fd["projected_point"] = proj_point
                smooth_proj_point = smooth_projected_points.get(fn)
                if smooth_proj_point is not None:
                    fd["smooth_projected_point"] = smooth_proj_point
            updated_count += 1

        if "projected_track" in track:
            del track["projected_track"]
        _finalize_saved_track(track)
    
    # Save back to JSON
    try:
        doc["projected_polygon"] = projected_polygon if projected_polygon is not None else doc.get("projected_polygon", [])
        doc["projection_settings"] = {
            "rotation_degrees": float(rot_deg),
            "flip": bool(flip),
            "reflect_horizontal": bool(reflect_h),
            "reflect_vertical": bool(reflect_v),
        }
        with open(outpath, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"[proj] Applied projections to {updated_count}/{len(all_tracks)} tracks → {outpath}")
        return True
    except Exception as e:
        print(f"[proj] Failed to save JSON: {e}")
        return False


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


def run_tracking(model, frame, click_pos=None, tracked_id=None, do_log=False):
    """Run YOLO+BotSort on frame – class-agnostic.

    click_pos  – (fx, fy) lock onto whichever box the click lands in.
    tracked_id – int BotSort track ID to follow.

    Returns (boxes, new_tracked_id, tracked_label).
    """
    results = model.track(
        frame,
        tracker="botsort.yaml",
        persist=True,
        verbose=False,
        imgsz=TRACK_INFER_IMGSZ,
        conf=0.25,
        iou=0.6,
        classes=DETECT_CLASS_IDS,
    )[0]

    all_dets = []   # (tid, x1, y1, x2, y2, class_name)
    if results.boxes.id is not None:
        names = results.names
        for box, tid in zip(results.boxes, results.boxes.id.int().tolist()):
            cls_name = names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_dets.append((tid, x1, y1, x2, y2, cls_name))

    if do_log:
        print(f"[botsort] detections={len(all_dets)} tracked_id={tracked_id}")

    # --- user clicked: lock onto whichever box contains the click ---
    if click_pos is not None:
        fx, fy = click_pos
        for (tid, x1, y1, x2, y2, cls_name) in all_dets:
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                if do_log:
                    print(f"[botsort] locked id={tid} cls={cls_name}")
                return [(x1, y1, x2, y2, cls_name, TRACK_COLOR)], tid, cls_name
        if do_log:
            print("[botsort] click outside all boxes – showing all")
        boxes = [(x1, y1, x2, y2, cls_name, DIM_COLOR)
                 for (_, x1, y1, x2, y2, cls_name) in all_dets]
        return boxes, None, ""

    # --- following a specific track ID ---
    if tracked_id is not None:
        for (tid, x1, y1, x2, y2, cls_name) in all_dets:
            if tid == tracked_id:
                return [(x1, y1, x2, y2, cls_name, TRACK_COLOR)], tracked_id, cls_name
        if do_log:
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

    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, os.cpu_count() or 1))
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
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
    speed_4x        = False  # Z toggles 4x playback speed
    track_infer_tick = 0      # increments during live playback while locked
    cached_track_box = None   # (x1,y1,x2,y2,label) reused between infer ticks
    polygon_points  = []     # list of 4 (x, y) points in frame coordinates
    polygon_edge_lengths = [0.0, 0.0, 0.0, 0.0]
    polygon_diag_lengths = [0.0, 0.0]
    polygon_len_edit_idx = None
    polygon_len_edit_text = ""
    polygon_confirmed = False
    show_projection_overlay = False
    projection_rotation_deg = 0.0
    projection_flip = False
    projection_reflect_h = False
    projection_reflect_v = False
    smooth_trajectories = False
    video_name      = os.path.basename(video_path)

    # Load existing tracks from JSON on startup
    json_path = _markers_path(video_path)
    if os.path.isfile(json_path):
        try:
            with open(json_path) as _f:
                _loaded = json.load(_f)
            # Handle both old format (list of tracks) and new format (dict with tracks+markers)
            if isinstance(_loaded, dict):
                _tracks_raw  = _loaded.get("tracks", [])
                _markers_raw = _loaded.get("markers", [])
                _polygon_raw = _loaded.get("polygon", [])
                _edge_raw    = _loaded.get("polygon_edge_lengths", [])
                _diag_raw    = _loaded.get("polygon_diagonal_lengths", [])
                _proj_settings = _read_projection_settings(_loaded)
            else:
                _tracks_raw  = _loaded
                _markers_raw = []
                _polygon_raw = []
                _edge_raw    = []
                _diag_raw    = []
                _proj_settings = _read_projection_settings({})
            for _td in _tracks_raw:
                saved_tracks.append(_finalize_saved_track(_td))
            for _m in _markers_raw:
                markers.append(_m)
                marker_counter = max(marker_counter,
                    int(_m["label"][1:]) if _m["label"].startswith("M") and
                    _m["label"][1:].isdigit() else marker_counter)
            if isinstance(_polygon_raw, list):
                polygon_points = [(int(p[0]), int(p[1]))
                                  for p in _polygon_raw
                                  if isinstance(p, (list, tuple)) and len(p) == 2]
            if len(polygon_points) > 4:
                polygon_points = polygon_points[:4]
            polygon_confirmed = (len(polygon_points) == 4)
            if isinstance(_edge_raw, list) and len(_edge_raw) == 4:
                try:
                    polygon_edge_lengths = [max(0.0, float(v)) for v in _edge_raw]
                except Exception:
                    polygon_edge_lengths = [0.0, 0.0, 0.0, 0.0]
            elif polygon_confirmed:
                polygon_edge_lengths = _polygon_edge_lengths_from_points(polygon_points)
            if isinstance(_diag_raw, list) and len(_diag_raw) == 2:
                try:
                    polygon_diag_lengths = [max(0.0, float(v)) for v in _diag_raw]
                except Exception:
                    polygon_diag_lengths = [0.0, 0.0]
            elif polygon_confirmed and all(v <= 0 for v in polygon_diag_lengths):
                p = np.array(polygon_points, dtype=np.float32)
                polygon_diag_lengths = [
                    float(np.hypot(*(p[2] - p[0]))),
                    float(np.hypot(*(p[3] - p[1]))),
                ]
            projection_rotation_deg = float(_proj_settings.get("rotation_degrees", 0.0))
            projection_flip = bool(_proj_settings.get("flip", False))
            projection_reflect_h = bool(_proj_settings.get("reflect_horizontal", False))
            projection_reflect_v = bool(_proj_settings.get("reflect_vertical", False))
            print(f"[load] {len(saved_tracks)} tracks, {len(markers)} markers loaded")
            if polygon_confirmed:
                print("[load] crosswalk polygon loaded")
            # Auto-seek to first marker if any
            if markers:
                first_m = min(markers, key=lambda mk: mk["frame"])
                paused = True
                ret, frame, current_frame = seek(cap, first_m["frame"], total_frames)
                print(f"[load] auto-seek to marker '{first_m['label']}' at frame {current_frame}")
        except Exception as _e:
            print(f"[load] failed to read {json_path}: {_e}")

    if not polygon_confirmed:
        paused = True
        print("[setup] draw crosswalk polygon with 4 clicks, Enter to confirm")

    while True:
        display_polygon_points = list(polygon_points)
        drag_idx = _state["polygon_drag_idx"]
        drag_cur = _state["polygon_drag_current"]
        if drag_idx is not None and drag_cur is not None and drag_idx < len(display_polygon_points):
            drag_frame_pos = canvas_to_frame(drag_cur[0], drag_cur[1])
            if drag_frame_pos is not None:
                display_polygon_points[drag_idx] = (int(drag_frame_pos[0]), int(drag_frame_pos[1]))

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
                    traj_points = []
                    if pstates:
                        for entry in _ordered_state_cats(pmode):
                            pcat = entry[0]
                            if pcat in pstates:
                                ann_lines.append((str(pstates[pcat]),
                                                  entry[2] if len(entry) > 2 else 'per_frame'))
                    if pmode in ("person", "vehicle-pos"):
                        path_key = "_smooth_person_path_pts" if smooth_trajectories else "_person_path_pts"
                        path_all = saved_tracks[pidx].get(path_key, [])
                        traj_points = [(mx, my) for (fn, mx, my) in path_all if fn <= current_frame]
                    preview_boxes.append((px1, py1, px2, py2, plbl, pcol, ann_lines, traj_points, pmode))

        _dr = _state["drag_start"]
        _dc = _state["drag_current"]
        _drag_rect = (_dr[0], _dr[1], _dc[0], _dc[1]) if _dr and _dc else None

        vp_cursor_line = None
        hover_pos = _state.get("hover_pos")
        if paused and polygon_confirmed and track_mode == "vehicle-pos" and hover_pos is not None:
            hp = canvas_to_frame(hover_pos[0], hover_pos[1])
            if hp is not None:
                base_box = None
                if current_frame in track_history:
                    bx1h, by1h, bx2h, by2h, _, _, _ = track_history[current_frame]
                    base_box = (bx1h, by1h, bx2h, by2h)
                line_box = _vehicle_pos_box_from_anchor(hp, frame.shape, base_box=base_box)
                vx = int(_layout["x_off"] + line_box[0] * _layout["scale"])
                vy1 = int(_layout["y_off"] + line_box[1] * _layout["scale"])
                vy2 = int(_layout["y_off"] + line_box[3] * _layout["scale"])
                vp_cursor_line = (vx, vy1, vy2)

        vp_drag_line = None
        if _state.get("vp_drag_start") is not None and _state.get("vp_drag_line") is not None and _state.get("vp_drag_current") is not None:
            sx, sy = _state["vp_drag_start"]
            lx, ly1, ly2 = _state["vp_drag_line"]
            cx, cy = _state["vp_drag_current"]
            vp_drag_line = (lx + (cx - sx), ly1 + (cy - sy), ly2 + (cy - sy))

        projection_overlay_data = None
        if show_projection_overlay and polygon_confirmed:
            projection_overlay_data = _compute_projection_overlay_data(
                polygon_points,
                polygon_edge_lengths,
                polygon_diag_lengths,
                projection_rotation_deg,
                projection_flip,
                projection_reflect_h,
                projection_reflect_v,
                track_history,
                current_frame,
                saved_tracks,
                previewed_idxs,
                smooth_trajectories=smooth_trajectories,
            )

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
                               "drag_rect":      _drag_rect,
                               "vehicle_cursor_line": vp_cursor_line,
                               "vehicle_drag_line": vp_drag_line,
                               "polygon_points": display_polygon_points,
                               "polygon_edge_lengths": polygon_edge_lengths,
                               "polygon_diag_lengths": polygon_diag_lengths,
                               "polygon_len_edit_idx": polygon_len_edit_idx,
                               "polygon_len_edit_text": polygon_len_edit_text,
                               "polygon_drag_idx": drag_idx,
                               "polygon_setup":  not polygon_confirmed,
                               "smooth_trajectories": smooth_trajectories,
                               "show_projection_overlay": show_projection_overlay,
                               "projection_overlay_data": projection_overlay_data})
        cv2.imshow(WINDOW_NAME, canvas)

        wait_ms = 30 if paused else max(
            FAST_PLAY_WAIT_MS,
            (frame_delay_ms // 4) if speed_4x else frame_delay_ms
        )
        key = cv2.waitKeyEx(wait_ms)

        # --- timeline click → seek ---
        if _state["seek_frame"] is not None:
            if not polygon_confirmed:
                _state["seek_frame"] = None
                continue
            paused     = True
            tracked_id = None
            cached_track_box = None
            track_infer_tick = 0
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

        # --- polygon point drag end → move crosswalk point and refresh states ---
        if _state["polygon_drag_end"] is not None:
            drag_idx, drag_cx, drag_cy = _state["polygon_drag_end"]
            _state["polygon_drag_end"] = None
            frame_pos = canvas_to_frame(drag_cx, drag_cy)
            if frame_pos is not None and 0 <= drag_idx < len(polygon_points):
                polygon_points[drag_idx] = (int(frame_pos[0]), int(frame_pos[1]))
                _apply_polygon_to_history(track_history, polygon_points)
                for saved_track in saved_tracks:
                    _refresh_saved_track_polygon(saved_track, polygon_points)
                if current_frame in track_history:
                    hx1, hy1, hx2, hy2, hlbl, hmode, hstates = track_history[current_frame]
                    if tracked_id is None:
                        boxes = [(hx1, hy1, hx2, hy2, hlbl, HISTORY_COLOR)]
                        tracked_label = hlbl
                    track_mode = hmode
                    track_states = dict(hstates)
                elif tracked_id is not None and boxes:
                    x1b, y1b, x2b, y2b, _, _ = boxes[0]
                    track_states = _apply_polygon_crosswalk_state(
                        track_mode, track_states, (x1b, y1b, x2b, y2b), polygon_points)
                if polygon_confirmed:
                    save_all_annotations_to_json(
                        video_path, saved_tracks, markers, polygon_points,
                        polygon_edge_lengths=polygon_edge_lengths,
                        polygon_diagonal_lengths=polygon_diag_lengths,
                        projection_rot_deg=projection_rotation_deg,
                        projection_flip=projection_flip,
                        projection_reflect_h=projection_reflect_h,
                        projection_reflect_v=projection_reflect_v)
                print(f"[polygon] moved point {drag_idx + 1}/{len(polygon_points)}")
            continue

        # --- vehicle-pos line drag end ---
        if _state["vp_drag_end"] is not None:
            lx, ly1, ly2 = _state["vp_drag_end"]
            _state["vp_drag_end"] = None
            fp_top = canvas_to_frame(lx, ly1)
            fp_bot = canvas_to_frame(lx, ly2)
            if fp_top is not None and fp_bot is not None and polygon_confirmed:
                x1, y1 = fp_top
                x2, y2 = fp_bot
                line_x = int(round((x1 + x2) / 2.0))
                by1, by2 = (y1, y2) if y1 <= y2 else (y2, y1)
                if current_frame in track_history:
                    _, _, _, _, _, _, cur_states = track_history[current_frame]
                    use_states = dict(cur_states)
                else:
                    use_states = dict(track_states)
                use_states = _apply_polygon_crosswalk_state(
                    "vehicle-pos", use_states, (line_x, by1, line_x, by2), polygon_points)
                track_mode = "vehicle-pos"
                track_states = dict(use_states)
                track_history[current_frame] = (line_x, by1, line_x, by2,
                                                "vehicle-pos", "vehicle-pos", dict(track_states))
                boxes = [(line_x, by1, line_x, by2, "vehicle-pos", TRACK_COLOR)]
                tracked_id = None
                cached_track_box = None
                track_infer_tick = 0
                tracked_label = "vehicle-pos"
                paused = True
                show_popup = True
            continue

        # --- sidebar click → track-list interaction ---
        if _state["sidebar_click"] is not None:
            if not polygon_confirmed:
                _state["sidebar_click"] = None
                continue
            scx, scy = _state["sidebar_click"]
            _state["sidebar_click"] = None

            sidebar_handled = False
            for zone in _layout.get("polygon_length_zones", []):
                if not (zone["y1"] <= scy <= zone["y2"] and zone["x1"] <= scx <= zone["x2"]):
                    continue
                if zone["type"] == "poly_len_field":
                    polygon_len_edit_idx = zone["idx"]
                    if zone.get("kind") == "diag":
                        polygon_len_edit_text = f"{float(polygon_diag_lengths[zone.get('val_idx', 0)]):.2f}"
                    else:
                        polygon_len_edit_text = f"{float(polygon_edge_lengths[zone.get('val_idx', 0)]):.2f}"
                    sidebar_handled = True
                    break
                if zone["type"] == "poly_len_save":
                    save_all_annotations_to_json(
                        video_path, saved_tracks, markers, polygon_points,
                        polygon_edge_lengths=polygon_edge_lengths,
                        polygon_diagonal_lengths=polygon_diag_lengths,
                        projection_rot_deg=projection_rotation_deg,
                        projection_flip=projection_flip,
                        projection_reflect_h=projection_reflect_h,
                        projection_reflect_v=projection_reflect_v)
                    print("[polygon] edge lengths saved")
                    sidebar_handled = True
                    break
                if zone["type"] == "poly_save_all_proj":
                    if polygon_confirmed and polygon_points and len(polygon_points) >= 4:
                        polygon_edge_lengths_valid = (polygon_edge_lengths and 
                                                      all(x > 0 for x in polygon_edge_lengths[:4]))
                        polygon_diag_lengths_valid = (polygon_diag_lengths and 
                                                      all(x > 0 for x in polygon_diag_lengths[:2]))
                        if polygon_edge_lengths_valid and polygon_diag_lengths_valid:
                            success = apply_projections_to_all_tracks(
                                video_path, saved_tracks, polygon_points,
                                polygon_edge_lengths,
                                polygon_diag_lengths,
                                projection_rotation_deg,
                                projection_flip,
                                projection_reflect_h,
                                projection_reflect_v,
                                smooth_trajectories=smooth_trajectories)
                            if success:
                                mode_txt = "smoothed" if smooth_trajectories else "raw"
                                print(f"[polygon] all track projections saved ({mode_txt})")
                            sidebar_handled = True
                        else:
                            print("[polygon] invalid edge/diagonal lengths")
                            sidebar_handled = True
                    else:
                        print("[polygon] polygon not confirmed or incomplete")
                        sidebar_handled = True
                    break
            if sidebar_handled:
                continue

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
                        total_pages_after = max(1, (len(saved_tracks) + TRACKS_PER_PAGE - 1)
                                                // TRACKS_PER_PAGE)
                        track_list_page = min(track_list_page, total_pages_after - 1)
                        save_all_annotations_to_json(
                            video_path, saved_tracks, markers, polygon_points,
                            polygon_edge_lengths=polygon_edge_lengths,
                            polygon_diagonal_lengths=polygon_diag_lengths,
                            projection_rot_deg=projection_rotation_deg,
                            projection_flip=projection_flip,
                            projection_reflect_h=projection_reflect_h,
                            projection_reflect_v=projection_reflect_v)
                        print(f"[delete] removed saved track #{idx + 1}")
                    break
                elif ztype == "delete_marker":
                    idx = zone["idx"]
                    if 0 <= idx < len(markers):
                        del markers[idx]
                        save_all_annotations_to_json(
                            video_path, saved_tracks, markers, polygon_points,
                            polygon_edge_lengths=polygon_edge_lengths,
                            polygon_diagonal_lengths=polygon_diag_lengths,
                            projection_rot_deg=projection_rotation_deg,
                            projection_flip=projection_flip,
                            projection_reflect_h=projection_reflect_h,
                            projection_reflect_v=projection_reflect_v)
                        print(f"[delete] removed marker #{idx + 1}")
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
            if not polygon_confirmed:
                _state["drag_end"] = None
                continue
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
                    draw_states = _apply_polygon_crosswalk_state(
                        draw_mode, draw_states, (fx1, fy1, fx2, fy2), polygon_points)
                    if draw_mode == "vehicle-pos":
                        line_x = int(round((fx1 + fx2) / 2.0))
                        by1 = min(fy1, fy2)
                        by2 = max(fy1, fy2)
                        draw_states = _apply_polygon_crosswalk_state(
                            draw_mode, draw_states, (line_x, by1, line_x, by2), polygon_points)
                        draw_label = "vehicle-pos"
                        track_history[current_frame] = (line_x, by1, line_x, by2,
                                                        draw_label, draw_mode, draw_states)
                        boxes = [(line_x, by1, line_x, by2, draw_label, TRACK_COLOR)]
                    else:
                        draw_label  = draw_mode
                        track_history[current_frame] = (fx1, fy1, fx2, fy2,
                                                        draw_label, draw_mode, draw_states)
                        boxes         = [(fx1, fy1, fx2, fy2, draw_label, TRACK_COLOR)]
                    track_mode    = draw_mode
                    track_states  = dict(draw_states)
                    tracked_id    = None
                    cached_track_box = None
                    track_infer_tick = 0
                    tracked_label = draw_label
                    paused        = True
                    show_popup    = True
                    print(f"[draw] manual box at frame {current_frame}  {fx1},{fy1}-{fx2},{fy2}")
            continue

        # --- video click → popup toggle OR BotSort detect + lock ---
        if _state["detect_click"] is not None:
            cx, cy = _state["detect_click"]
            _state["detect_click"] = None

            if not polygon_confirmed:
                frame_pos = canvas_to_frame(cx, cy)
                if frame_pos is not None and len(polygon_points) < 4:
                    polygon_points.append((int(frame_pos[0]), int(frame_pos[1])))
                    print(f"[setup] polygon point {len(polygon_points)}/4 added")
                continue

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
                                if track_mode == "vehicle-pos":
                                    line_x = int(round((x1h + x2h) / 2.0))
                                    x1h, x2h = line_x, line_x
                                # label follows mode
                                adj_states = _apply_polygon_crosswalk_state(
                                    track_mode, track_states, (x1h, y1h, x2h, y2h), polygon_points)
                                track_states = dict(adj_states)
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
                        hx1, hy1, hx2, hy2, _, hmode, _ = track_history[current_frame]
                        fx, fy = frame_pos
                        if _point_hits_track_box(fx, fy, (hx1, hy1, hx2, hy2), hmode):
                            show_popup = True
                            hit_popup  = True
                    if not hit_popup:
                        if track_mode == "vehicle-pos":
                            prev_box = None
                            prev_frames = sorted(f for f in track_history if f <= current_frame)
                            if prev_frames:
                                px1, py1, px2, py2, _, pmode, _ = track_history[prev_frames[-1]]
                                if pmode == "vehicle-pos":
                                    prev_box = (px1, py1, px2, py2)
                            vx1, vy1, vx2, vy2 = _vehicle_pos_box_from_anchor(
                                frame_pos, frame.shape, base_box=prev_box)
                            manual_states = _apply_polygon_crosswalk_state(
                                "vehicle-pos", dict(track_states),
                                (vx1, vy1, vx2, vy2), polygon_points)
                            track_mode = "vehicle-pos"
                            track_states = dict(manual_states)
                            track_history[current_frame] = (vx1, vy1, vx2, vy2,
                                                            "vehicle-pos", "vehicle-pos",
                                                            dict(track_states))
                            boxes = [(vx1, vy1, vx2, vy2, "vehicle-pos", TRACK_COLOR)]
                            tracked_id = None
                            cached_track_box = None
                            track_infer_tick = 0
                            tracked_label = "vehicle-pos"
                            paused = True
                            show_popup = True
                            continue
                        paused     = True
                        boxes, tracked_id, tracked_label = run_tracking(
                            model, frame, click_pos=frame_pos, do_log=True)
                        if tracked_id is not None:
                            if boxes:
                                bx1, by1, bx2, by2, blbl, _ = boxes[0]
                                cached_track_box = (bx1, by1, bx2, by2, blbl)
                            else:
                                cached_track_box = None
                            track_infer_tick = 0
                            last_states_per_mode[track_mode] = dict(track_states)
                            track_mode   = YOLO_CLASS_MAP.get(tracked_label, DEFAULT_MODE)
                            track_states = dict(last_states_per_mode.get(
                                track_mode, _default_track_states(track_mode)))
                            show_popup   = True
                        else:
                            cached_track_box = None
                            track_infer_tick = 0
                            show_popup   = False
            continue

        # --- keyboard ---
        if key != -1:
            key_ch = key & 0xFF

            if polygon_len_edit_idx is not None:
                if key_ch in KEY_ENTER:
                    try:
                        v = float(polygon_len_edit_text)
                        if polygon_len_edit_idx < 4:
                            polygon_edge_lengths[polygon_len_edit_idx] = max(0.0, v)
                        else:
                            polygon_diag_lengths[polygon_len_edit_idx - 4] = max(0.0, v)
                    except Exception:
                        pass
                    polygon_len_edit_idx = None
                    polygon_len_edit_text = ""
                elif key_ch in KEY_BKSP:
                    polygon_len_edit_text = polygon_len_edit_text[:-1]
                elif key_ch == 27:
                    polygon_len_edit_idx = None
                    polygon_len_edit_text = ""
                elif ord('0') <= key_ch <= ord('9'):
                    polygon_len_edit_text += chr(key_ch)
                elif key_ch == ord('.') and '.' not in polygon_len_edit_text:
                    polygon_len_edit_text += "."
                continue

            if key_ch in (ord('q'), ord('Q'), 27):    # Q / Esc
                break

            if not polygon_confirmed:
                if key_ch in KEY_BKSP:
                    polygon_points = []
                    print("[setup] polygon points cleared")
                elif key_ch in KEY_ENTER:
                    if len(polygon_points) == 4:
                        polygon_confirmed = True
                        if not any(v > 0 for v in polygon_edge_lengths):
                            polygon_edge_lengths = _polygon_edge_lengths_from_points(polygon_points)
                        if not any(v > 0 for v in polygon_diag_lengths):
                            p = np.array(polygon_points, dtype=np.float32)
                            polygon_diag_lengths = [
                                float(np.hypot(*(p[2] - p[0]))),
                                float(np.hypot(*(p[3] - p[1]))),
                            ]
                        save_all_annotations_to_json(
                            video_path, saved_tracks, markers, polygon_points,
                            polygon_edge_lengths=polygon_edge_lengths,
                            polygon_diagonal_lengths=polygon_diag_lengths,
                            projection_rot_deg=projection_rotation_deg,
                            projection_flip=projection_flip,
                            projection_reflect_h=projection_reflect_h,
                            projection_reflect_v=projection_reflect_v)
                        print("[setup] crosswalk polygon confirmed")
                    else:
                        print("[setup] need exactly 4 points before confirm")
                continue

            elif key_ch in (ord('m'), ord('M')):       # M – add named marker
                marker_counter += 1
                mlabel = f"M{marker_counter}"
                markers.append({"label": mlabel, "frame": current_frame})
                markers.sort(key=lambda mk: mk["frame"])
                save_all_annotations_to_json(
                    video_path, saved_tracks, markers, polygon_points,
                    polygon_edge_lengths=polygon_edge_lengths,
                    polygon_diagonal_lengths=polygon_diag_lengths,
                    projection_rot_deg=projection_rotation_deg,
                    projection_flip=projection_flip,
                    projection_reflect_h=projection_reflect_h,
                    projection_reflect_v=projection_reflect_v)
                print(f"[marker] added {mlabel} at frame {current_frame}")

            elif key_ch in (ord('p'), ord('P')):
                show_projection_overlay = not show_projection_overlay

            elif key_ch in (ord('s'), ord('S')):
                smooth_trajectories = not smooth_trajectories
                mode_txt = "ON" if smooth_trajectories else "OFF"
                if smooth_trajectories and savgol_filter is None:
                    print("[smooth] scipy not available, using raw trajectories")
                print(f"[smooth] trajectory smoothing {mode_txt}")

            elif key_ch in (ord('o'), ord('O')):
                if show_projection_overlay:
                    projection_rotation_deg = (projection_rotation_deg + 7.5) % 360.0
                    save_all_annotations_to_json(
                        video_path, saved_tracks, markers, polygon_points,
                        polygon_edge_lengths=polygon_edge_lengths,
                        polygon_diagonal_lengths=polygon_diag_lengths,
                        projection_rot_deg=projection_rotation_deg,
                        projection_flip=projection_flip,
                        projection_reflect_h=projection_reflect_h,
                        projection_reflect_v=projection_reflect_v)

            elif key_ch in (ord('f'), ord('F')):
                if show_projection_overlay:
                    projection_flip = not projection_flip
                    save_all_annotations_to_json(
                        video_path, saved_tracks, markers, polygon_points,
                        polygon_edge_lengths=polygon_edge_lengths,
                        polygon_diagonal_lengths=polygon_diag_lengths,
                        projection_rot_deg=projection_rotation_deg,
                        projection_flip=projection_flip,
                        projection_reflect_h=projection_reflect_h,
                        projection_reflect_v=projection_reflect_v)

            elif key_ch in (ord('h'), ord('H')):
                if show_projection_overlay:
                    projection_reflect_h = not projection_reflect_h
                    save_all_annotations_to_json(
                        video_path, saved_tracks, markers, polygon_points,
                        polygon_edge_lengths=polygon_edge_lengths,
                        polygon_diagonal_lengths=polygon_diag_lengths,
                        projection_rot_deg=projection_rotation_deg,
                        projection_flip=projection_flip,
                        projection_reflect_h=projection_reflect_h,
                        projection_reflect_v=projection_reflect_v)

            elif key_ch == ord('v'):
                if show_projection_overlay:
                    projection_reflect_v = not projection_reflect_v
                    save_all_annotations_to_json(
                        video_path, saved_tracks, markers, polygon_points,
                        polygon_edge_lengths=polygon_edge_lengths,
                        polygon_diagonal_lengths=polygon_diag_lengths,
                        projection_rot_deg=projection_rotation_deg,
                        projection_flip=projection_flip,
                        projection_reflect_h=projection_reflect_h,
                        projection_reflect_v=projection_reflect_v)

            elif key_ch == ord('V'):       # V – manual vehicle-pos mode
                last_states_per_mode[track_mode] = dict(track_states)
                track_mode = "vehicle-pos"
                track_states = dict(last_states_per_mode.get(
                    track_mode, _default_track_states(track_mode)))
                tracked_id = None
                cached_track_box = None
                track_infer_tick = 0
                tracked_label = "vehicle-pos"
                show_popup = False
                paused = True
                if current_frame in track_history:
                    hx1, hy1, hx2, hy2, hlbl, hmode, hstates = track_history[current_frame]
                    if hmode == "vehicle-pos":
                        boxes = [(hx1, hy1, hx2, hy2, hlbl, HISTORY_COLOR)]
                        track_states = dict(hstates)
                print("[mode] vehicle-pos manual mode (click to place vertical line)")

            elif key_ch == 32:                         # Space – play/pause
                paused     = not paused
                show_popup = False  # dismiss popup when playback resumes

            elif key_ch == ord('z'):                    # Z - toggle 4x speed
                speed_4x = not speed_4x

            elif key_ch in KEY_ENTER:                  # Enter – save current track
                if track_history:
                    interpolate_track_history(track_history)
                    _apply_polygon_to_history(track_history, polygon_points)
                    track_dict = save_track_to_json(
                        video_path, track_history, fps, markers, polygon_points,
                        polygon_edge_lengths=polygon_edge_lengths,
                        polygon_diagonal_lengths=polygon_diag_lengths,
                        projection_rot_deg=projection_rotation_deg,
                        projection_flip=projection_flip,
                        projection_reflect_h=projection_reflect_h,
                        projection_reflect_v=projection_reflect_v,
                        smooth_trajectories=smooth_trajectories)
                    track_dict["_frame_lookup"] = build_frame_lookup(track_dict)
                    track_dict["_person_path_pts"] = build_person_path_points(track_dict)
                    saved_tracks.append(track_dict)
                    last_states_per_mode[track_mode] = dict(track_states)
                    track_history  = {}
                    boxes          = []
                    tracked_id     = None
                    cached_track_box = None
                    track_infer_tick = 0
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
                        cached_track_box = None
                        track_infer_tick = 0
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
                        cached_track_box = None
                        track_infer_tick = 0
                        tracked_label = ""
                        reset_botsort(model)
                    print(f"[trim] deleted {len(deleted)} frames before frame {current_frame}")

                else:                              # plain Backspace – clear current track
                    last_states_per_mode[track_mode] = dict(track_states)
                    track_history = {}
                    boxes         = []
                    tracked_id    = None
                    cached_track_box = None
                    track_infer_tick = 0
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
                cached_track_box = None
                track_infer_tick = 0
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
                cached_track_box = None
                track_infer_tick = 0
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
            # Fast path: grab (cheap), skip extra frames, then decode once.
            got = cap.grab()
            if not got:
                paused = True
            else:
                for _ in range(PLAYBACK_SKIP_FRAMES):
                    if not cap.grab():
                        break
                ret, next_frame = cap.retrieve()
                if not ret:
                    paused = True
                    continue

                frame         = next_frame
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                show_popup    = False  # popup is click-frame only

                if tracked_id is not None:
                    # Live tracking with throttled inference and bbox reuse.
                    track_infer_tick += 1
                    should_infer = (
                        track_infer_tick % TRACK_INFER_EVERY_N == 0
                        or cached_track_box is None
                    )
                    if should_infer:
                        boxes, tracked_id, tracked_label = run_tracking(
                            model, frame, tracked_id=tracked_id)
                        if boxes:
                            x1c, y1c, x2c, y2c, lblc, _ = boxes[0]
                            cached_track_box = (x1c, y1c, x2c, y2c, lblc)
                        else:
                            cached_track_box = None
                    elif cached_track_box is not None:
                        x1c, y1c, x2c, y2c, lblc = cached_track_box
                        boxes = [(x1c, y1c, x2c, y2c, lblc, TRACK_COLOR)]

                    # Record to history with per-frame mode and states
                    if boxes:
                        x1, y1, x2, y2, lbl, _ = boxes[0]
                        auto_states = _apply_polygon_crosswalk_state(
                            track_mode, track_states, (x1, y1, x2, y2), polygon_points)
                        track_states = dict(auto_states)
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
