import os
import shutil
import time
import threading
import cv2
import numpy as np
import torch
import chess
import chess.engine
from ultralytics import YOLO

torch.backends.cudnn.benchmark = True

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
DROIDCAM_SOURCE = "http://192.168.31.119:4747/video"
CALIB_FILE = "board_corners.npz"

DEFAULT_CONF_THRESH = 0.40
NMS_IOU_THRESH = 0.45
EMA_ALPHA = 0.28
HOLD_SECONDS = 1.0

STOCKFISH_CANDIDATES = [
    "stockfish.exe",
    "stockfish",
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    shutil.which("stockfish") or ""
]
STOCKFISH_PATH = next((p for p in STOCKFISH_CANDIDATES if p and os.path.exists(p)), None)

for candidate in ["models/best.pt", "best.pt"]:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break
else:
    raise FileNotFoundError("Could not find model weights inside models/best.pt")

YOLO_TO_CHESS = {
    "white-pawn":   (chess.PAWN, chess.WHITE),
    "white-knight": (chess.KNIGHT, chess.WHITE),
    "white-bishop": (chess.BISHOP, chess.WHITE),
    "white-rook":   (chess.ROOK, chess.WHITE),
    "white-queen":  (chess.QUEEN, chess.WHITE),
    "white-king":   (chess.KING, chess.WHITE),
    "black-pawn":   (chess.PAWN, chess.BLACK),
    "black-knight": (chess.KNIGHT, chess.BLACK),
    "black-bishop": (chess.BISHOP, chess.BLACK),
    "black-rook":   (chess.ROOK, chess.BLACK),
    "black-queen":  (chess.QUEEN, chess.BLACK),
    "black-king":   (chess.KING, chess.BLACK),
}


# =====================================================================
# 2. THREADED CAMERA
# =====================================================================
class ThreadedCamera:
    def __init__(self, src: str):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {src}. Check IP or run: adb forward tcp:4747 tcp:4747")

        self.ret, self.frame = self.cap.read()
        self.seq = 0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    self.seq += 1
            else:
                time.sleep(0.002)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None, self.seq

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# =====================================================================
# 3. AUTOMATIC CORNER DETECTION ALGORITHMS
# =====================================================================
def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points strictly as: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]     # TL
    rect[2] = pts[np.argmax(s)]     # BR
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def auto_detect_corners_from_pieces(raw_detections: list, frame_shape: tuple):
    """
    Auto-calibrates the 8x8 board corners using the detected White and Black pieces
    at the starting position. Also automatically determines if Black is at the bottom.
    raw_detections: list of (base_x, base_y, is_white, conf)
    Returns: (corners_list, black_at_bottom) or (None, None)
    """
    if frame_shape is None or len(raw_detections) < 12:
        return None, None

    h, w = frame_shape[:2]
    pts = np.array([[d[0], d[1]] for d in raw_detections], dtype=np.float32)
    colors = np.array([1 if d[2] else 0 for d in raw_detections])

    # Filter spatial outliers (e.g., background desk detections far from the board cluster)
    median_pt = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - median_pt, axis=1)
    inliers = dists < np.percentile(dists, 88)
    pts = pts[inliers]
    colors = colors[inliers]

    if len(pts) < 10:
        return None, None

    # Determine orientation: Are Black pieces lower in the image (higher Y) than White pieces?
    white_pts = pts[colors == 1]
    black_pts = pts[colors == 0]
    if len(white_pts) >= 4 and len(black_pts) >= 4:
        black_at_bottom = float(np.median(black_pts[:, 1])) > float(np.median(white_pts[:, 1]))
    else:
        black_at_bottom = True

    # Fit convex hull & minimum area quadrilateral around piece centers
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    quad = order_quad_points(box)

    # Expand piece-center quad outward from 7x7 center span to full 8x8 outer border
    center = np.mean(quad, axis=0)
    scale_x = 8.15 / 7.0
    scale_y = 8.20 / 7.0
    expanded = np.zeros_like(quad)
    for i in range(4):
        vec = quad[i] - center
        expanded[i, 0] = np.clip(center[0] + vec[0] * scale_x, 5, w - 5)
        expanded[i, 1] = np.clip(center[1] + vec[1] * scale_y, 5, h - 5)

    corners = [(int(p[0]), int(p[1])) for p in expanded]
    return corners, black_at_bottom


def auto_detect_corners_from_edges(frame: np.ndarray):
    """
    Finds the largest quadrilateral chessboard contour using Canny edges + PolyDP.
    """
    if frame is None:
        return None

    h, w = frame.shape[:2]
    min_area = (w * h) * 0.18
    max_area = (w * h) * 0.92

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    for low, high in [(40, 120), (25, 90), (60, 180)]:
        edges = cv2.Canny(blur, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:8]:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    ordered = order_quad_points(approx.reshape(4, 2))
                    return [(int(p[0]), int(p[1])) for p in ordered]
    return None


# =====================================================================
# 4. EMA STATE TRACKER + VISUAL MOTION DETECTOR
# =====================================================================
class BoardStateTracker:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.color_ema = np.zeros((64, 3), dtype=np.float32)
        self.color_ema[:, 0] = 1.0
        self.type_ema = np.zeros((64, 7), dtype=np.float32)
        self.ref_warped_gray = None

    def capture_reference(self, warped_bgr: np.ndarray):
        if warped_bgr is not None:
            gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
            self.ref_warped_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    def update(self, frame_obs: dict):
        for sq in chess.SQUARES:
            obs = frame_obs.get(sq)
            target_color = np.zeros(3, dtype=np.float32)
            if obs is None:
                target_color[0] = 1.0
                self.type_ema[sq] *= (1.0 - self.alpha * 0.5)
            else:
                p_type, p_color, conf = obs
                weight = min(1.0, conf * 1.35)
                if p_color == chess.WHITE:
                    target_color[1] = weight
                    target_color[0] = 1.0 - weight
                else:
                    target_color[2] = weight
                    target_color[0] = 1.0 - weight

                self.type_ema[sq] *= (1.0 - self.alpha)
                self.type_ema[sq, p_type] += self.alpha * conf

            self.color_ema[sq] = (1.0 - self.alpha) * self.color_ema[sq] + self.alpha * target_color

    def stable_squares(self):
        out = {}
        for sq in chess.SQUARES:
            empty_p, white_p, black_p = self.color_ema[sq]
            occ_p = white_p + black_p
            if occ_p > 0.45 and occ_p > empty_p:
                p_color = chess.WHITE if white_p >= black_p else chess.BLACK
                best_type = int(np.argmax(self.type_ema[sq, 1:])) + 1
                if self.type_ema[sq, best_type] <= 0.01:
                    best_type = chess.PAWN
                out[sq] = (best_type, p_color, float(max(white_p, black_p)))
        return out

    def compute_square_diff(self, warped_bgr: np.ndarray, black_at_bottom: bool):
        diffs = np.zeros(64, dtype=np.float32)
        if self.ref_warped_gray is None or warped_bgr is None:
            return diffs

        curr_gray = cv2.GaussianBlur(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        abs_diff = cv2.absdiff(curr_gray, self.ref_warped_gray)

        for sq in chess.SQUARES:
            f_idx = chess.square_file(sq)
            r_idx = chess.square_rank(sq)
            col, row = (7 - f_idx, r_idx) if black_at_bottom else (f_idx, 7 - r_idx)
            x1, y1 = col * 100 + 20, row * 100 + 20
            patch = abs_diff[y1:y1 + 60, x1:x1 + 60]
            diffs[sq] = float(np.mean(patch))
        return diffs


# =====================================================================
# 5. PERSPECTIVE GRID MAPPER (WITH DRAGGABLE CORNERS & AUTO-CALIB)
# =====================================================================
class BoardMapper:
    def __init__(self, black_at_bottom=True):
        self.corners = []
        self.M = None
        self.M_inv = None
        self.black_at_bottom = black_at_bottom
        self.dragging_idx = None
        self.load_calibration()

    def save_calibration(self):
        if self.is_calibrated:
            np.savez(CALIB_FILE, corners=np.array(self.corners), black_at_bottom=self.black_at_bottom)

    def load_calibration(self):
        if os.path.exists(CALIB_FILE):
            try:
                data = np.load(CALIB_FILE)
                pts = data["corners"]
                if len(pts) == 4:
                    self.corners = [(int(p[0]), int(p[1])) for p in pts]
                    self.black_at_bottom = bool(data["black_at_bottom"])
                    self._compute_homography()
                    print(f"[CALIBRATION LOADED] from {CALIB_FILE}")
            except Exception:
                pass

    def set_corners(self, corners_list, black_at_bottom=None):
        self.corners = corners_list
        if black_at_bottom is not None:
            self.black_at_bottom = black_at_bottom
        self._compute_homography()
        self.save_calibration()

    def toggle_orientation(self):
        self.black_at_bottom = not self.black_at_bottom
        self.save_calibration()

    def handle_mouse(self, event, x: int, y: int):
        """Supports clicking 4 corners initially AND dragging existing yellow corners to fine-tune."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_calibrated:
                for idx, pt in enumerate(self.corners):
                    if np.hypot(x - pt[0], y - pt[1]) < 22:
                        self.dragging_idx = idx
                        return True
            elif len(self.corners) < 4:
                self.corners.append((x, y))
                if len(self.corners) == 4:
                    self._compute_homography()
                    self.save_calibration()
                return True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_idx is not None:
            self.corners[self.dragging_idx] = (x, y)
            self._compute_homography()
            return True

        elif event == cv2.EVENT_LBUTTONUP and self.dragging_idx is not None:
            self.corners[self.dragging_idx] = (x, y)
            self.dragging_idx = None
            self._compute_homography()
            self.save_calibration()
            return True
        return False

    def reset(self):
        self.corners = []
        self.M = None
        self.M_inv = None
        self.dragging_idx = None
        if os.path.exists(CALIB_FILE):
            os.remove(CALIB_FILE)

    def _compute_homography(self):
        src = np.array(self.corners, dtype=np.float32)
        dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    @property
    def is_calibrated(self):
        return self.M is not None

    def warp_board(self, frame: np.ndarray):
        if not self.is_calibrated or frame is None:
            return None
        return cv2.warpPerspective(frame, self.M, (800, 800))

    def pixel_to_square(self, x: float, y: float):
        if not self.is_calibrated:
            return None
        pt = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.M)[0][0]
        wx, wy = warped[0], warped[1]
        if 0 <= wx < 800 and 0 <= wy < 800:
            col = int(wx // 100)
            row = int(wy // 100)
            if self.black_at_bottom:
                file_idx, rank_idx = 7 - col, row
            else:
                file_idx, rank_idx = col, 7 - row
            return chess.square(file_idx, rank_idx)
        return None

    def square_to_pixel(self, sq: chess.Square):
        if not self.is_calibrated:
            return None
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        col, row = (7 - file_idx, rank_idx) if self.black_at_bottom else (file_idx, 7 - rank_idx)
        wx = (col + 0.5) * 100.0
        wy = (row + 0.5) * 100.0
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        orig = cv2.perspectiveTransform(pt, self.M_inv)[0][0]
        return int(orig[0]), int(orig[1])

    def draw_grid(self, frame):
        corner_labels = (
            ["1:h1 (TL)", "2:a1 (TR)", "3:a8 (BR)", "4:h8 (BL)"]
            if self.black_at_bottom
            else ["1:a8 (TL)", "2:h8 (TR)", "3:h1 (BR)", "4:a1 (BL)"]
        )

        if self.is_calibrated:
            for i in range(9):
                color = (255, 255, 0) if i in (0, 8) else (150, 150, 150)
                thick = 2 if i in (0, 8) else 1
                p1 = cv2.perspectiveTransform(np.array([[[i * 100.0, 0.0]]], dtype=np.float32), self.M_inv)[0][0]
                p2 = cv2.perspectiveTransform(np.array([[[i * 100.0, 800.0]]], dtype=np.float32), self.M_inv)[0][0]
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick)

                p3 = cv2.perspectiveTransform(np.array([[[0.0, i * 100.0]]], dtype=np.float32), self.M_inv)[0][0]
                p4 = cv2.perspectiveTransform(np.array([[[800.0, i * 100.0]]], dtype=np.float32), self.M_inv)[0][0]
                cv2.line(frame, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), color, thick)

        for idx, pt in enumerate(self.corners):
            cv2.circle(frame, pt, 8, (0, 0, 0), -1)
            cv2.circle(frame, pt, 6, (0, 255, 255), -1)
            cv2.putText(frame, corner_labels[idx], (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# =====================================================================
# 6. HYBRID MOVE DETECTOR
# =====================================================================
def match_best_legal_move(board: chess.Board, detected_squares: dict, sq_diffs: np.ndarray):
    def score_board(b: chess.Board):
        score = 0.0
        for sq in chess.SQUARES:
            expected = b.piece_at(sq)
            obs = detected_squares.get(sq)
            if expected is not None and obs is not None:
                if expected.color == obs[1]:
                    score += 2.5 * obs[2]
                    if expected.piece_type == obs[0]:
                        score += 1.0 * obs[2]
                else:
                    score -= 2.0
            elif expected is None and obs is None:
                score += 2.0
            elif expected is not None and obs is None:
                score -= 1.5
        return score

    base_score = score_board(board)
    best_move = None
    best_gain = 0.0

    for move in board.legal_moves:
        from_obs = detected_squares.get(move.from_square)
        to_obs = detected_squares.get(move.to_square)

        from_vacated = (from_obs is None) or (sq_diffs[move.from_square] > 14.0)
        if not from_vacated:
            continue
        if to_obs is None or to_obs[1] != board.turn:
            continue

        board.push(move)
        new_score = score_board(board)
        board.pop()

        diff_bonus = 1.5 if (sq_diffs[move.from_square] > 12.0 and sq_diffs[move.to_square] > 12.0) else 0.0
        gain = (new_score - base_score) + diff_bonus
        if gain > best_gain:
            best_gain = gain
            best_move = move

    if best_move is not None and best_gain >= 2.8:
        return best_move
    return None


# =====================================================================
# 7. MAIN APPLICATION LOOP
# =====================================================================
def main():
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    use_half = use_cuda and MODEL_PATH.endswith(".pt")
    quantize = 16 if use_half else None

    print(f"Loading model: {MODEL_PATH} on {device} (FP16={use_half})")
    model = YOLO(MODEL_PATH, task="detect")

    engine = None
    if STOCKFISH_PATH:
        print(f"Starting Stockfish engine from: {STOCKFISH_PATH}")
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    else:
        print("WARNING: Stockfish binary not found! Set STOCKFISH_PATH in script.")

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, device=device, quantize=quantize, verbose=False)

    cam = ThreadedCamera(DROIDCAM_SOURCE)
    conf_thresh = DEFAULT_CONF_THRESH
    tracker = BoardStateTracker()
    mapper = BoardMapper(black_at_bottom=True)
    board = chess.Board()

    candidate_move = None
    candidate_since = 0.0
    pending_bot_move = None
    show_topdown = True
    auto_calibrated_once = mapper.is_calibrated

    # Cached state across idle camera ticks so key handlers never receive None
    last_frame = None
    last_warped = None
    last_raw_detections = []

    banner_msg = (
        "YOUR TURN (WHITE): Play your move!"
        if mapper.is_calibrated
        else "Press [A] Auto-Detect from Pieces | [G] Edge Detect | Or Click 4 Corners"
    )
    win_name = "Chess Vision (Main Camera)"
    cv2.namedWindow(win_name)

    def on_mouse(event, x, y, flags, param):
        nonlocal banner_msg
        if mapper.handle_mouse(event, x, y):
            if mapper.is_calibrated:
                banner_msg = "Calibrated! (Drag any yellow corner to fine-tune, or press [R])"
            else:
                banner_msg = f"Click corner {len(mapper.corners)+1}/4 (TL -> TR -> BR -> BL)"

    cv2.setMouseCallback(win_name, on_mouse)

    def handle_keys(frame, warped_frame, raw_detections):
        nonlocal conf_thresh, banner_msg, pending_bot_move, candidate_move, show_topdown
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key in (ord('+'), ord('=')):
            conf_thresh = min(0.90, conf_thresh + 0.05)
        elif key == ord('-'):
            conf_thresh = max(0.15, conf_thresh - 0.05)
        elif key == ord('a'):
            # Method 1: Auto-detect corners from YOLO pieces
            if frame is not None:
                corners, b_bottom = auto_detect_corners_from_pieces(raw_detections, frame.shape)
                if corners is not None:
                    mapper.set_corners(corners, b_bottom)
                    tracker.reset()
                    w_img = mapper.warp_board(frame)
                    tracker.capture_reference(w_img)
                    banner_msg = "AUTO-CALIBRATED from pieces! Drag yellow dots to tweak."
                    print("[AUTO-CALIB] Success using YOLO piece layout!")
                else:
                    banner_msg = f"Auto-Calib failed: Only {len(raw_detections)}/12 pieces visible."
        elif key == ord('g'):
            # Method 2: Auto-detect corners from OpenCV board edges
            if frame is not None:
                corners = auto_detect_corners_from_edges(frame)
                if corners is not None:
                    mapper.set_corners(corners)
                    tracker.reset()
                    w_img = mapper.warp_board(frame)
                    tracker.capture_reference(w_img)
                    banner_msg = "AUTO-CALIBRATED from board edges! Drag yellow dots to tweak."
                    print("[AUTO-CALIB] Success using OpenCV quad contour!")
                else:
                    banner_msg = "Edge Auto-Calib couldn't find a clear 4-corner frame. Try [A]!"
        elif key == ord('f'):
            mapper.toggle_orientation()
            tracker.reset()
            banner_msg = "Orientation Flipped!"
        elif key == ord('t'):
            show_topdown = not show_topdown
            if not show_topdown:
                cv2.destroyWindow("Top-Down 8x8 Inspector")
        elif key == ord('c'):
            mapper.reset()
            tracker.reset()
            banner_msg = "Cleared! Press [A] for Auto-Corners or click 4 corners."
        elif key == ord('r'):
            board.reset()
            tracker.reset()
            if warped_frame is not None:
                tracker.capture_reference(warped_frame)
            pending_bot_move = None
            candidate_move = None
            banner_msg = "Board Reset & Reference Captured! YOUR TURN (WHITE)"
        return True

    fps = 0.0
    prev_time = time.perf_counter()
    last_seq = -1

    print("\n--- CONTROLS ---")
    print("[A]         : AUTO-DETECT corners from pieces (Works with all 32 pieces on board!)")
    print("[G]         : AUTO-DETECT corners from board edges/contour")
    print("Mouse Drag  : Click & drag any of the 4 yellow corners to fine-tune")
    print("[F]         : Flip board orientation (Black at bottom <-> White at bottom)")
    print("[T]         : Toggle 800x800 Top-Down Inspector window")
    print("[C]         : Clear calibration")
    print("[R]         : Reset game & capture clean reference board")
    print("[Q]         : Quit\n")

    try:
        while True:
            ret, frame, seq = cam.read()
            if not ret or frame is None or seq == last_seq:
                if not handle_keys(last_frame, last_warped, last_raw_detections):
                    break
                time.sleep(0.001)
                continue
            last_seq = seq
            last_frame = frame.copy()

            h, w, _ = frame.shape

            results = model.predict(
                frame,
                imgsz=640,
                conf=conf_thresh,
                iou=NMS_IOU_THRESH,
                agnostic_nms=True,
                device=device,
                quantize=quantize,
                verbose=False
            )

            boxes = results[0].boxes
            piece_count = 0
            frame_obs = {}
            raw_detections = []

            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                conf_arr = boxes.conf.cpu().numpy()

                for i in range(len(xyxy_arr)):
                    conf = float(conf_arr[i])
                    cls_name = model.names[cls_arr[i]]
                    x1, y1, x2, y2 = xyxy_arr[i].astype(int)

                    box_w = x2 - x1
                    base_x = (x1 + x2) >> 1
                    base_y = int(y2 - max(4, box_w * 0.28))

                    if cls_name in YOLO_TO_CHESS:
                        raw_detections.append((base_x, base_y, "white" in cls_name, conf))

                # Auto-calibrate on the very first startup frame if enough pieces are visible
                if not mapper.is_calibrated and not auto_calibrated_once and len(raw_detections) >= 16:
                    corners, b_bottom = auto_detect_corners_from_pieces(raw_detections, frame.shape)
                    if corners is not None:
                        mapper.set_corners(corners, b_bottom)
                        auto_calibrated_once = True
                        banner_msg = "AUTO-CALIBRATED! Drag yellow corners if needed, then press [R]"

                for i in range(len(xyxy_arr)):
                    conf = float(conf_arr[i])
                    cls_name = model.names[cls_arr[i]]
                    x1, y1, x2, y2 = xyxy_arr[i].astype(int)
                    box_w = x2 - x1
                    base_x = (x1 + x2) >> 1
                    base_y = int(y2 - max(4, box_w * 0.28))

                    sq = mapper.pixel_to_square(base_x, base_y)
                    if mapper.is_calibrated and sq is None:
                        continue

                    piece_count += 1
                    color = (255, 255, 0) if "white" in cls_name else (0, 140, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (base_x, base_y), 4, (0, 0, 255), -1)

                    if sq is not None and cls_name in YOLO_TO_CHESS:
                        p_type, p_color = YOLO_TO_CHESS[cls_name]
                        if sq not in frame_obs or conf > frame_obs[sq][2]:
                            frame_obs[sq] = (p_type, p_color, conf)

                        short_sym = ("w" if p_color == chess.WHITE else "b") + chess.Piece(p_type, p_color).symbol().upper()
                        cv2.putText(
                            frame,
                            f"{short_sym}[{chess.square_name(sq)}]",
                            (x1, max(y1 - 6, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            color,
                            2
                        )

            warped = mapper.warp_board(last_frame)
            last_warped = warped
            last_raw_detections = raw_detections

            if mapper.is_calibrated and tracker.ref_warped_gray is None and warped is not None:
                tracker.capture_reference(warped)

            now = time.perf_counter()
            tracker.update(frame_obs)
            detected_squares = tracker.stable_squares()
            sq_diffs = tracker.compute_square_diff(warped, mapper.black_at_bottom)

            mapper.draw_grid(frame)

            if show_topdown and warped is not None:
                td_vis = warped.copy()
                for i in range(1, 8):
                    cv2.line(td_vis, (i * 100, 0), (i * 100, 800), (100, 100, 100), 1)
                    cv2.line(td_vis, (0, i * 100), (800, i * 100), (100, 100, 100), 1)

                for sq in chess.SQUARES:
                    f_idx, r_idx = chess.square_file(sq), chess.square_rank(sq)
                    col, row = (7 - f_idx, r_idx) if mapper.black_at_bottom else (f_idx, 7 - r_idx)
                    cx, cy = col * 100 + 50, row * 100 + 50

                    cv2.putText(td_vis, chess.square_name(sq), (col * 100 + 6, row * 100 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                    if sq in detected_squares:
                        p_type, p_color, _ = detected_squares[sq]
                        sym = chess.Piece(p_type, p_color).symbol().upper()
                        c_bgr = (255, 255, 0) if p_color == chess.WHITE else (0, 140, 255)
                        cv2.putText(td_vis, sym, (cx - 12, cy + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, c_bgr, 3)

                cv2.imshow("Top-Down 8x8 Inspector", cv2.resize(td_vis, (520, 520)))

            # =================================================================
            # GAME LOGIC: WHITE (YOU) vs BLACK (STOCKFISH)
            # =================================================================
            if mapper.is_calibrated and not board.is_game_over():
                if board.turn == chess.WHITE:
                    detected_move = match_best_legal_move(board, detected_squares, sq_diffs)
                    if detected_move is not None:
                        if detected_move == candidate_move:
                            elapsed = now - candidate_since
                            banner_msg = f"Confirming White: {board.san(detected_move)} ({elapsed:.1f}s/{HOLD_SECONDS}s)"
                            if elapsed >= HOLD_SECONDS:
                                san_str = board.san(detected_move)
                                board.push(detected_move)
                                tracker.capture_reference(warped)
                                candidate_move = None
                                print(f"[WHITE PLAYED]: {san_str} ({detected_move.uci()})")

                                if not board.is_game_over() and engine is not None:
                                    res = engine.play(board, chess.engine.Limit(time=0.4))
                                    pending_bot_move = res.move
                                    bot_san = board.san(pending_bot_move)
                                    banner_msg = f"STOCKFISH (BLACK): {pending_bot_move.uci()[:2].upper()} -> {pending_bot_move.uci()[2:4].upper()} ({bot_san})"
                                    print(f"[STOCKFISH SAYS]: {banner_msg}")
                        else:
                            candidate_move = detected_move
                            candidate_since = now
                    else:
                        candidate_move = None
                        banner_msg = "YOUR TURN (WHITE): Play your move on the board"

                elif board.turn == chess.BLACK and pending_bot_move is not None:
                    p_from = mapper.square_to_pixel(pending_bot_move.from_square)
                    p_to = mapper.square_to_pixel(pending_bot_move.to_square)
                    if p_from and p_to:
                        cv2.arrowedLine(frame, p_from, p_to, (0, 255, 0), 5, tipLength=0.25)
                        cv2.circle(frame, p_from, 10, (0, 255, 0), 2)

                    from_obs = detected_squares.get(pending_bot_move.from_square)
                    to_obs = detected_squares.get(pending_bot_move.to_square)
                    from_empty = (from_obs is None) or (sq_diffs[pending_bot_move.from_square] > 14.0)
                    to_has_black = to_obs is not None and to_obs[1] == chess.BLACK

                    if from_empty and to_has_black:
                        if candidate_move == pending_bot_move:
                            elapsed = now - candidate_since
                            if elapsed >= HOLD_SECONDS:
                                board.push(pending_bot_move)
                                tracker.capture_reference(warped)
                                print(f"[BLACK MOVE COMPLETED]: {pending_bot_move.uci()}")
                                pending_bot_move = None
                                candidate_move = None
                                banner_msg = "YOUR TURN (WHITE): Play your move on the board"
                        else:
                            candidate_move = pending_bot_move
                            candidate_since = now
                    else:
                        candidate_move = None

            elif board.is_game_over():
                banner_msg = f"GAME OVER! Result: {board.result()}"

            # =================================================================
            # HUD & ON-SCREEN DISPLAY
            # =================================================================
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = 0.85 * fps + 0.15 * instant_fps if fps > 0 else instant_fps
            prev_time = now

            orient_str = "Black@Bottom" if mapper.black_at_bottom else "White@Bottom"
            status = (
                f"FPS: {fps:.1f} | Boxes: {piece_count} | Stable: {len(detected_squares)} "
                f"| [A]:Auto-Pieces [G]:Auto-Edge | {orient_str}"
            )
            cv2.rectangle(frame, (0, 0), (w, 72), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

            banner_color = (0, 255, 255) if board.turn == chess.WHITE else (0, 255, 0)
            cv2.putText(frame, banner_msg, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, banner_color, 2)

            cv2.imshow(win_name, frame)

            if not handle_keys(last_frame, last_warped, last_raw_detections):
                break

    finally:
        if engine is not None:
            engine.quit()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
