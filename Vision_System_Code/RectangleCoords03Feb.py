import socket
import time
import sys
import cv2
import numpy as np
import threading
import queue
import select

from picamera2 import Picamera2

# ----------------------------
# Robot connection settings
# ----------------------------
ROBOT_IP = "192.168.1.5"
ROBOT_PORT = 2000

# Fixed Z and U (edit for your setup)
ROBOT_Z_FIXED = -20.0
ROBOT_U_FIXED = 180.0

# ----------------------------
# Camera "world" size (mm) used during homography entry
# ----------------------------
WORLD_W = 1200.0
WORLD_H = 900.0

# Separate calibration window so live window doesn't disappear
CALIB_WINDOW_NAME = "Pi Camera Calibration"
WINDOW_NAME = "Pi Camera Calibration & Detection"

# If camera "world" coords are not the same origin as robot coords,
# apply offsets here (mm) AFTER mapping:
ROBOT_X_OFFSET = 0.0
ROBOT_Y_OFFSET = 0.0

RECTANGULARITY_MIN = 0.80 # 0.40–0.60
ASPECT_MAX = 25.0
MAX_AREA_FRAC = 0.35        # reject blobs >35% of frame (prevents "whole screen")
BORDER_MARGIN_FRAC = 0.02   # reject blobs touching within 2% of border
V_PERCENTILE = 20           # adaptive cutoff percentile (15–30)
V_PAD = 15                  # extra pad above percentile (10–25)


# World coords (mm) corresponding to click order 1..4
AUTO_WORLD_PTS = [
    (250.0, 374.0),  # 1
    (250.0, 60.0),   # 2
    (416.0, 60.0),   # 3
    (333.0, 217.0),  # 4
]


# If your Y axis is flipped between camera-world and robot-world, set True
FLIP_Y = False

# Rate limiting / filtering
SEND_PERIOD = 0.5     # seconds between robot commands
MIN_MOVE_MM = 5.0     # only send if target changed by at least this much

# ----------------------------
# Camera settings
# ----------------------------
MAX_RESOLUTION = (4056, 3040)
PROC_WIDTH = 1280

# ----------------------------
# Sending control (SAFE DEFAULT)
# ----------------------------
SEND_ENABLED = False   # SAFE DEFAULT: start paused; toggle with 'p'

# --- Black-rectangle detection tuning ---
BLACK_V_MAX = 110
BLACK_S_MAX = 200
MIN_AREA_FRAC = 0.00008
APPROX_EPS_FRAC = 0.02
MORPH_K = 7

RECONNECT_DELAY = 1.0  # seconds between reconnect attempts


def recv_line(sock: socket.socket, timeout=120) -> str:
    sock.settimeout(timeout)
    data = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("Controller closed the connection")
        if b in (b"\n", b"\r"):
            sock.settimeout(0.0)
            try:
                nxt = sock.recv(1)
                if nxt not in (b"\n", b"\r", b""):
                    data.extend(nxt)
            except Exception:
                pass
            finally:
                sock.settimeout(timeout)
            break
        data.extend(b)
    return data.decode("ascii", errors="replace").strip()


class RobotWorker(threading.Thread):
    """
    Background robot comms thread.
    - Holds the socket connection
    - Takes latest target or raw commands from a queue
    - Sends one command, waits for reply, then takes next
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.sock = None
        self.running = True
        self.last_sent = None  # (x,y,z,u)

    def disconnect(self):
        try:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
        finally:
            self.sock = None

    def connect(self):
        self.disconnect()
        self.sock = socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=5)
        ready = recv_line(self.sock, timeout=10)
        print("Robot:", ready)

    def stop(self):
        self.running = False
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

    def flush_queue(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def send_target(self, x, y, z, u):
        self.flush_queue()
        self.q.put(("target", (x, y, z, u)))

    def send_raw(self, cmd: str, flush: bool = False):
        if flush:
            self.flush_queue()
        self.q.put(("raw", cmd.strip()))

    def run(self):
        while self.running:
            if self.sock is None:
                try:
                    print("[RobotWorker] Connecting...")
                    self.connect()
                    print("[RobotWorker] Connected.")
                except Exception as e:
                    print(f"[RobotWorker] Connect failed: {e} (retrying)")
                    self.disconnect()
                    time.sleep(RECONNECT_DELAY)
                    continue

            item = self.q.get()
            if item is None:
                break

            kind, payload = item

            if kind == "target":
                x, y, z, u = payload
                cmd = f"GOP;{x:.3f};{y:.3f};{z:.3f};{u:.3f}"
            else:
                cmd = str(payload)

            try:
                print(f"Sending: {cmd}")
                self.sock.sendall((cmd + "\r\n").encode("ascii"))
                reply = recv_line(self.sock, timeout=120)
                print("Robot:", reply)

                if reply.upper().startswith("ERROR"):
                    print("[RobotWorker] Robot returned ERROR. Flushing queued commands.")
                    self.flush_queue()

                if kind == "target":
                    self.last_sent = payload

            except socket.timeout:
                print("[RobotWorker] Reply timeout. Dropping connection and retrying...")
                self.disconnect()

            except Exception as e:
                print(f"[RobotWorker] Comm error: {e}. Dropping connection and retrying...")
                self.disconnect()

        try:
            if self.sock:
                try:
                    self.sock.sendall(b"EXIT\r\n")
                except Exception:
                    pass
                self.disconnect()
        except Exception:
            pass


class ConsoleInputWorker(threading.Thread):
    """
    Sends whatever you type into the terminal to the robot *only when enabled*.
    Uses select() so it never blocks during calibration.
    """
    def __init__(self, robot: RobotWorker, allow_event: threading.Event):
        super().__init__(daemon=True)
        self.robot = robot
        self.allow_event = allow_event
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if not self.allow_event.is_set():
                time.sleep(0.05)
                continue

            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue

                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.05)
                    continue

                line = line.strip()
                if not line:
                    continue

                self.robot.send_raw(line, flush=False)

            except Exception as e:
                print(f"[ConsoleInputWorker] Error: {e}")
                time.sleep(0.2)


def capture_frame_bgr(picam2):
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def calibration_loop(picam2):
    image_points = []
    homography = None
    calib_img = capture_frame_bgr(picam2)
    msg = "Click 4 corners, 'n'=new image, 'c'=calibrate, 'r'=reset, 'q'=quit"

    def mouse_cb(event, x, y, flags, param):
        nonlocal image_points
        if event == cv2.EVENT_LBUTTONDOWN and len(image_points) < 4:
            image_points.append([float(x), float(y)])
            print(f"Clicked image point {len(image_points)}: ({x}, {y})")

    cv2.namedWindow(CALIB_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(CALIB_WINDOW_NAME, mouse_cb)

    while True:
        disp = calib_img.copy()

        for idx, (px, py) in enumerate(image_points):
            cv2.circle(disp, (int(px), int(py)), 8, (0, 255, 255), -1)
            cv2.putText(disp, str(idx + 1), (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(disp, msg, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(CALIB_WINDOW_NAME, disp)
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q'):
            sys.exit(0)
        elif key == ord('r'):
            image_points = []
            print("Calibration reset")
        elif key == ord('n'):
            calib_img = capture_frame_bgr(picam2)
            image_points = []
            print("New calibration image captured")
        elif key == ord('c'):
            if len(image_points) != 4:
                print("Need 4 image points. Click 4 points first.")
                continue

            # Auto-fill world points (must match click order 1..4)
            user_world_pts = [[x, y] for (x, y) in AUTO_WORLD_PTS]
            print("Using AUTO_WORLD_PTS for world coordinates:")
            for i, (x, y) in enumerate(AUTO_WORLD_PTS, 1):
                print(f"  {i}: {x:.1f} {y:.1f}")


            img_pts_np = np.array(image_points, dtype=np.float32)
            world_np = np.array(user_world_pts, dtype=np.float32)
            H, _ = cv2.findHomography(img_pts_np, world_np)

            if H is None:
                print("Homography failed, try again.")
                image_points = []
                continue

            homography = H.astype(np.float32)
            print("Homography computed.")
            break

    cv2.destroyWindow(CALIB_WINDOW_NAME)
    return homography


def detect_rectangle(frame_bgr, homography):
    h_full, w_full = frame_bgr.shape[:2]

    # Downscale
    proc_w = PROC_WIDTH
    proc_h = int((proc_w * h_full) / w_full)
    proc = cv2.resize(frame_bgr, (proc_w, proc_h))

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    Hc, S, V = cv2.split(hsv)

    # --- Adaptive black threshold to avoid "whole screen is black" under bad lighting ---
    v_cut = int(np.percentile(V, V_PERCENTILE) + V_PAD)
    v_cut = min(v_cut, BLACK_V_MAX)  # never exceed your configured max
    black_mask = cv2.inRange(hsv, (0, 0, 0), (179, BLACK_S_MAX, v_cut))

    black_mask = cv2.GaussianBlur(black_mask, (5, 5), 0)

    k = MORPH_K
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proc_area = float(proc_w * proc_h)
    area_min = max(800.0, MIN_AREA_FRAC * proc_area)
    area_max = MAX_AREA_FRAC * proc_area

    # Border margin in pixels (proc space)
    mx = int(BORDER_MARGIN_FRAC * proc_w)
    my = int(BORDER_MARGIN_FRAC * proc_h)

    best_rect_full = None
    best_score = -1.0

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)

        # Reject tiny and "whole screen" blobs
        if area < area_min or area > area_max:
            continue

        # Reject anything touching the border (often background/lighting)
        x, y, w, h = cv2.boundingRect(hull)
        if x <= mx or y <= my or (x + w) >= (proc_w - mx) or (y + h) >= (proc_h - my):
            continue

        rect = cv2.minAreaRect(hull)  # PROC coords
        (cx, cy), (rw, rh), ang = rect
        if rw <= 2 or rh <= 2:
            continue

        rect_area = float(rw * rh)
        rectangularity = float(area / (rect_area + 1e-6))
        if rectangularity < RECTANGULARITY_MIN:
            continue

        aspect = (rw / rh) if rw >= rh else (rh / rw)
        if aspect > ASPECT_MAX:
            continue

        # Darkness sanity check inside hull
        mask_hull = np.zeros(black_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_hull, [hull], -1, 255, thickness=-1)
        mean_v = cv2.mean(V, mask=mask_hull)[0]
        if mean_v > (v_cut + 25):
            continue

        # Score: prefer big + rectangular
        score = area * (0.5 + 0.5 * rectangularity)
        if score > best_score:
            best_score = score

            # Convert rect center/size from PROC -> FULL
            scale_x = w_full / proc_w
            scale_y = h_full / proc_h
            cx_full = cx * scale_x
            cy_full = cy * scale_y
            rw_full = rw * scale_x
            rh_full = rh * scale_y

            best_rect_full = ((cx_full, cy_full), (rw_full, rh_full), ang)

    if best_rect_full is None or homography is None:
        return {}

    box = cv2.boxPoints(best_rect_full).astype(int)
    (cx, cy), (_, _), angle = best_rect_full

    center_pixel = np.array([[[cx, cy]]], dtype=np.float32)
    real_pt = cv2.perspectiveTransform(center_pixel, homography)
    rx, ry = real_pt[0][0]

    if FLIP_Y:
        ry = WORLD_H - ry

    return {
        "box": box,
        "center_px": (cx, cy),
        "center_mm": (float(rx), float(ry)),
        "angle": float(angle),
    }



def main():
    global SEND_ENABLED

    print("Starting robot worker...")
    robot = RobotWorker()
    robot.start()

    # Console passthrough gate (OFF until calibration finishes)
    console_allowed = threading.Event()
    console_allowed.clear()

    console = ConsoleInputWorker(robot, console_allowed)
    console.start()
    print("[INFO] Console passthrough will enable AFTER calibration completes.")

    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_still_configuration({"size": MAX_RESOLUTION})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("[INFO] Moving robot out of camera view for calibration: GOJ;90;0;0;0")
    robot.send_raw("GOJ;90;0;0;0", flush=True)

    print("Calibration...")
    homography = calibration_loop(picam2)

    console_allowed.set()
    print("[INFO] Calibration done. Console passthrough is now ENABLED.")

    # Create the LIVE window and keep it for the rest of the program
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_sent_t = 0.0
    last_xy = None
    last_cmd = None  # (x, y)

    try:
        while True:
            frame_bgr = capture_frame_bgr(picam2)
            result = detect_rectangle(frame_bgr, homography)

            disp = frame_bgr.copy()

            status = "SEND: ARMED" if SEND_ENABLED else "SEND: SAFE (PAUSED)  [press 'p' to ARM]"
            color = (0, 255, 0) if SEND_ENABLED else (0, 0, 255)
            cv2.putText(disp, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            if result:
                box = result["box"]
                cx, cy = result["center_px"]
                rx, ry = result["center_mm"]
                angle = result["angle"]

                cv2.drawContours(disp, [box], 0, (0, 255, 0), 3)
                cv2.circle(disp, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(disp, f"X={rx:.1f}mm Y={ry:.1f}mm a={angle:.1f}",
                            (int(cx) - 160, int(cy) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                x_cmd = rx + ROBOT_X_OFFSET
                y_cmd = ry + ROBOT_Y_OFFSET
                last_cmd = (x_cmd, y_cmd)

                now = time.time()
                moved_enough = True
                if last_xy is not None:
                    dx = x_cmd - last_xy[0]
                    dy = y_cmd - last_xy[1]
                    moved_enough = (dx * dx + dy * dy) ** 0.5 >= MIN_MOVE_MM

                if SEND_ENABLED and (now - last_sent_t) >= SEND_PERIOD and moved_enough:
                    print(f"Queueing target: X={x_cmd:.3f} Y={y_cmd:.3f} Z={ROBOT_Z_FIXED:.3f} U={ROBOT_U_FIXED:.3f}")
                    robot.send_target(x_cmd, y_cmd, ROBOT_Z_FIXED, ROBOT_U_FIXED)
                    last_sent_t = now
                    last_xy = (x_cmd, y_cmd)
            else:
                cv2.putText(disp, "No rectangle detected", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("c"):
                # Disable console passthrough so calibration input() is safe
                console_allowed.clear()
                print("[INFO] Console passthrough DISABLED during calibration.")

                print("[INFO] Recalibrating: moving robot out of view: GOJ;90;0;0;0")
                robot.send_raw("GOJ;90;0;0;0", flush=True)

                print("Recalibrating...")
                homography = calibration_loop(picam2)

                console_allowed.set()
                print("[INFO] Calibration done. Console passthrough is now ENABLED.")

                last_cmd = None
                last_xy = None
                last_sent_t = 0.0

            elif key == ord("p"):
                SEND_ENABLED = not SEND_ENABLED
                state = "ARMED" if SEND_ENABLED else "SAFE (PAUSED)"
                print(f"[INFO] Robot sending state: {state}")
                if not SEND_ENABLED:
                    robot.flush_queue()

            elif key == ord("s"):
                if SEND_ENABLED:
                    print("[INFO] Step ignored (ARMED). Press 'p' to disarm first.")
                else:
                    if last_cmd is None:
                        print("[INFO] No target available to step-send yet.")
                    else:
                        x_cmd, y_cmd = last_cmd
                        print(f"[STEP] Sending one target: X={x_cmd:.3f} Y={y_cmd:.3f} Z={ROBOT_Z_FIXED:.3f} U={ROBOT_U_FIXED:.3f}")
                        robot.send_target(x_cmd, y_cmd, ROBOT_Z_FIXED, ROBOT_U_FIXED)
                        last_xy = (x_cmd, y_cmd)
                        last_sent_t = time.time()

    finally:
        cv2.destroyAllWindows()
        try:
            picam2.stop()
        except Exception:
            pass
        console.stop()
        robot.stop()


if __name__ == "__main__":
    main()
