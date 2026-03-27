import socket
import time
import sys
import cv2
import numpy as np
import threading
import queue
import RPi.GPIO as GPIO

from picamera2 import Picamera2

# ----------------------------
# Robot connection settings
# ----------------------------
ROBOT_IP = "192.168.1.5"
ROBOT_PORT = 2000

# Fixed Z and U (edit for your setup)
ROBOT_Z_FIXED = -100
ROBOT_U_OFFSET = 0.0  # deg (set if robot zero doesn't match camera zero)

# ----------------------------
# Camera "world" size (mm) used during homography entry
# IMPORTANT: set these to match the coordinates you enter at calibration time.
WORLD_W = 1200.0
WORLD_H = 900.0

# Use these values instead of typing calibration numbers
AUTO_WORLD_PTS = [
    (250.0, 374.0),  # 1
    (250.0, 60.0),   # 2
    (416.0, 60.0),   # 3
    (160.0, 225.0),  # 4
]

# If camera "world" coords are not the same origin as robot coords,
# apply offsets here (mm) AFTER mapping:
ROBOT_X_OFFSET = 0.0
ROBOT_Y_OFFSET = 0.0

# If your Y axis is flipped between camera-world and robot-world, set True
FLIP_Y = False

# Rate limiting / filtering (used only when continuous send is ON)
SEND_PERIOD = 0.5     # seconds between robot commands
MIN_MOVE_MM = 5.0     # only send if target changed by at least this much

# ----------------------------
# Camera settings
# ----------------------------
MAX_RESOLUTION = (4056, 3040)
PROC_WIDTH = 1280

WINDOW_NAME = "Pi Camera View"
MASK_WINDOW = "Mask"
CONTROL_WINDOW = "Controls"

# ----------------------------
# Safer default: start PAUSED (no auto sending)
# ----------------------------
SEND_ENABLED = False  # toggle continuous sending with 'p'

# ----------------------------
# Simplified detection sliders
# ----------------------------
DARK_THRESHOLD = 78      # pixels below this are considered object
BLUR_K = 6               # Gaussian blur kernel (odd)
MORPH_K = 5              # morphology kernel size (odd)
MORPH_ITERS = 0          # closing iterations
MIN_AREA_FRAC = 0.0081   # minimum contour area fraction of processed image
MIN_FILL_RATIO = 0.45    # contour area / minAreaRect area
MAX_AREA_FRAC = 0.035  

RECONNECT_DELAY = 1.0  # seconds between reconnect attempts

# ----------------------------
# Solenoid setup
# ----------------------------
SOLENOID_PIN = 23

GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setup(SOLENOID_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(SOLENOID_PIN, GPIO.LOW)


def activate_vac():
    print("[VAC] ON")
    GPIO.output(SOLENOID_PIN, GPIO.HIGH)


def deactivate_vac():
    print("[VAC] OFF")
    GPIO.output(SOLENOID_PIN, GPIO.LOW)


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


def rect_angle_to_u_from_world(box_full: np.ndarray, homography) -> float:
    """
    Compute object orientation from the rectangle corners AFTER homography.

    box_full: 4x2 rectangle corner points in full-image pixel coordinates
    homography: image->world homography

    Returns:
        u_deg in range [0, 180)
    """
    world_pts = cv2.perspectiveTransform(
        box_full.reshape(-1, 1, 2).astype(np.float32),
        homography
    ).reshape(4, 2)

    if FLIP_Y:
        world_pts[:, 1] = WORLD_H - world_pts[:, 1]

    edges = [
        world_pts[1] - world_pts[0],
        world_pts[2] - world_pts[1],
        world_pts[3] - world_pts[2],
        world_pts[0] - world_pts[3],
    ]

    lengths = [np.linalg.norm(e) for e in edges]
    long_edge = edges[int(np.argmax(lengths))]

    u = np.degrees(np.arctan2(long_edge[1], long_edge[0]))

    if u < 0:
        u += 180.0

    if u >= 180.0:
        u -= 180.0

    return float(u)


class RobotWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.sock = None
        self.running = True
        self.last_sent = None

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

    def send_cmd(self, cmd: str, wait: bool = False):
        cmd = cmd.strip()
        if not cmd:
            return
        done = threading.Event() if wait else None
        self.flush_queue()
        self.q.put((cmd, done))
        if done is not None:
            done.wait(timeout=180)

    def send_gop(self, x, y, z, u, wait: bool = False):
        cmd = f"GOP;{x:.3f};{y:.3f};{z:.3f};{u:.3f}"
        self.send_cmd(cmd, wait=wait)

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

            cmd, done_evt = item

            try:
                print(f"Sending: {cmd}")
                self.sock.sendall((cmd + "\r\n").encode("ascii"))

                while True:
                    reply = recv_line(self.sock, timeout=120)
                    reply_u = reply.strip().upper()
                    print("Robot:", reply_u)

                    if reply_u == "ACTIVATE_VAC":
                        activate_vac()
                        continue

                    if reply_u == "DEACTIVATE_VAC":
                        deactivate_vac()
                        continue

                    if reply_u == "READY":
                        break

                self.last_sent = cmd

            except socket.timeout:
                print("[RobotWorker] Reply timeout. Dropping connection and retrying...")
                self.disconnect()

            except Exception as e:
                print(f"[RobotWorker] Comm error: {e}. Dropping connection and retrying...")
                self.disconnect()

            finally:
                if done_evt is not None:
                    done_evt.set()

        try:
            if self.sock:
                try:
                    self.sock.sendall(b"EXIT\r\n")
                except Exception:
                    pass
                self.disconnect()
        except Exception:
            pass


class ConsoleCommandWorker(threading.Thread):
    def __init__(self, robot: RobotWorker, calibrated_event: threading.Event):
        super().__init__(daemon=True)
        self.robot = robot
        self.calibrated_event = calibrated_event
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        self.calibrated_event.wait()
        print("\n[Console] Calibration complete. You can now type robot commands here.")
        print("[Console] Examples: GOJ;90;0;0;0   |   GOP;100;200;-20;180   |   EXIT\n")
        while self.running:
            try:
                line = input().strip()
            except EOFError:
                break
            except Exception:
                continue
            if not line:
                continue
            self.robot.send_cmd(line, wait=False)


def capture_frame_bgr(picam2):
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def setup_tuning_sliders(window_name: str):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 420, 350)

    cv2.createTrackbar("Dark threshold", window_name, DARK_THRESHOLD, 255, lambda v: None)
    cv2.createTrackbar("Blur k (odd)", window_name, BLUR_K, 31, lambda v: None)
    cv2.createTrackbar("Morph k (odd)", window_name, MORPH_K, 51, lambda v: None)
    cv2.createTrackbar("Morph iters", window_name, MORPH_ITERS, 10, lambda v: None)
    cv2.createTrackbar("Min area x1e6", window_name, int(MIN_AREA_FRAC * 1_000_000), 200000, lambda v: None)
    cv2.createTrackbar("Min fill x100", window_name, int(MIN_FILL_RATIO * 100), 100, lambda v: None)
    cv2.createTrackbar("Max area x1e6", window_name, int(MAX_AREA_FRAC * 1_000_000), 1000000, lambda v: None)


def read_tuning_sliders(window_name: str):
    global DARK_THRESHOLD, BLUR_K, MORPH_K, MORPH_ITERS, MIN_AREA_FRAC, MAX_AREA_FRAC, MIN_FILL_RATIO

    DARK_THRESHOLD = cv2.getTrackbarPos("Dark threshold", window_name)

    BLUR_K = cv2.getTrackbarPos("Blur k (odd)", window_name)
    if BLUR_K < 1:
        BLUR_K = 1
    if BLUR_K % 2 == 0:
        BLUR_K += 1

    MORPH_K = cv2.getTrackbarPos("Morph k (odd)", window_name)
    if MORPH_K < 1:
        MORPH_K = 1
    if MORPH_K % 2 == 0:
        MORPH_K += 1

    MORPH_ITERS = cv2.getTrackbarPos("Morph iters", window_name)
    MAX_AREA_FRAC = cv2.getTrackbarPos("Max area x1e6", window_name) / 1_000_000.0
    MIN_AREA_FRAC = cv2.getTrackbarPos("Min area x1e6", window_name) / 1_000_000.0
    MIN_FILL_RATIO = cv2.getTrackbarPos("Min fill x100", window_name) / 100.0
    
    if MAX_AREA_FRAC < MIN_AREA_FRAC:
        MAX_AREA_FRAC = MIN_AREA_FRAC

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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    while True:
        disp = calib_img.copy()
        for idx, (px, py) in enumerate(image_points):
            cv2.circle(disp, (int(px), int(py)), 8, (0, 255, 255), -1)
            cv2.putText(disp, str(idx + 1), (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(disp, msg, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, disp)
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

            print("Using AUTO_WORLD_PTS (no typing).")
            user_world_pts = [list(p) for p in AUTO_WORLD_PTS]

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

    cv2.destroyWindow(WINDOW_NAME)
    return homography


def detect_rectangles(frame_bgr, homography):
    h_full, w_full = frame_bgr.shape[:2]

    proc_w = PROC_WIDTH
    proc_h = int((proc_w * h_full) / w_full)
    proc = cv2.resize(frame_bgr, (proc_w, proc_h))

    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (int(BLUR_K), int(BLUR_K)), 0)

    # Dark objects become white in the mask
    _, mask = cv2.threshold(gray, int(DARK_THRESHOLD), 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(MORPH_K), int(MORPH_K)))
    if MORPH_ITERS > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(MORPH_ITERS))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proc_area = float(proc_w * proc_h)
    area_min = max(400.0, float(MIN_AREA_FRAC) * proc_area)
    area_max = float(MAX_AREA_FRAC) * proc_area

    scale_x = w_full / proc_w
    scale_y = h_full / proc_h

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min or area > area_max:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), raw_angle = rect
        if rw <= 1 or rh <= 1:
            continue

        rect_area = float(rw * rh)
        if rect_area <= 1:
            continue

        fill_ratio = float(area) / rect_area
        if fill_ratio < float(MIN_FILL_RATIO):
            continue

        box_proc = cv2.boxPoints(rect).astype(np.float32)
        box_full = np.array([[pt[0] * scale_x, pt[1] * scale_y] for pt in box_proc], dtype=np.float32)
        (cxp, cyp), _, _ = rect
        cx = cxp * scale_x
        cy = cyp * scale_y

        if homography is None:
            continue

        center_pixel = np.array([[[cx, cy]]], dtype=np.float32)
        real_pt = cv2.perspectiveTransform(center_pixel, homography)
        rx, ry = real_pt[0][0]
        if FLIP_Y:
            ry = WORLD_H - ry

        u_deg = rect_angle_to_u_from_world(box_full, homography)

        # Score: prefer larger, better-filled rectangles
        score = area * fill_ratio

        detections.append({
            "score": float(score),
            "area": float(area),
            "fill_ratio": float(fill_ratio),
            "box": box_full,
            "center_px": (float(cx), float(cy)),
            "center_mm": (float(rx), float(ry)),
            "angle": float(raw_angle),
            "u_deg": float(u_deg),
            "rect_wh": (float(rw), float(rh)),
        })

    # Deterministic target selection
    detections.sort(key=lambda d: (-d["score"], -d["area"], d["center_px"][0], d["center_px"][1]))

    return {
        "mask": mask,
        "detections": detections
    }


def main():
    global SEND_ENABLED

    print("Starting robot worker...")
    robot = RobotWorker()
    robot.start()

    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_still_configuration({"size": MAX_RESOLUTION})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    calibrated_event = threading.Event()
    console_thread = ConsoleCommandWorker(robot, calibrated_event)
    console_thread.start()

    print("[INFO] Moving robot out of camera view: GOJ;90;0;0;0")
    robot.send_cmd("GOJ;90;0;0;0", wait=True)

    print("Calibration...")
    homography = calibration_loop(picam2)
    calibrated_event.set()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    setup_tuning_sliders(CONTROL_WINDOW)

    last_sent_t = 0.0
    last_xy = None
    last_cmd = None

    try:
        while True:
            frame_bgr = capture_frame_bgr(picam2)
            read_tuning_sliders(CONTROL_WINDOW)

            result = detect_rectangles(frame_bgr, homography)

            cv2.imshow(MASK_WINDOW, result.get("mask", np.zeros((10, 10), dtype=np.uint8)))

            disp = frame_bgr.copy()

            status = "SEND: ON (continuous)" if SEND_ENABLED else "SEND: PAUSED (press 's' to step, 'p' to enable)"
            color = (0, 255, 0) if SEND_ENABLED else (0, 0, 255)
            cv2.putText(disp, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            cv2.putText(
                disp,
                f"thr={DARK_THRESHOLD} blur={BLUR_K} morph={MORPH_K} iters={MORPH_ITERS}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                disp,
                f"minArea={MIN_AREA_FRAC:.6f} minFill={MIN_FILL_RATIO:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            detections = result.get("detections", [])

            if detections:
                cv2.putText(
                    disp,
                    f"Detected boxes: {len(detections)}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                for i, det in enumerate(detections):
                    box = det["box"]
                    cx, cy = det["center_px"]
                    rx, ry = det["center_mm"]
                    u_deg = det["u_deg"]

                    is_selected = (i == 0)

                    box_color = (0, 255, 0) if is_selected else (255, 200, 0)
                    center_color = (0, 0, 255) if is_selected else (255, 0, 0)
                    thickness = 4 if is_selected else 2

                    cv2.drawContours(disp, [box.astype(np.int32)], 0, box_color, thickness)
                    cv2.circle(disp, (int(cx), int(cy)), 8, center_color, -1)

                    label = f"#{i+1} X={rx:.1f} Y={ry:.1f} U={u_deg:.1f}"
                    cv2.putText(
                        disp,
                        label,
                        (int(cx) - 140, int(cy) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2
                    )

                target = detections[0]
                rx, ry = target["center_mm"]
                u_deg = target["u_deg"]

                x_cmd = rx + ROBOT_X_OFFSET
                y_cmd = ry + ROBOT_Y_OFFSET
                u_cmd = u_deg + ROBOT_U_OFFSET

                last_cmd = (x_cmd, y_cmd, u_cmd)

                cv2.putText(
                    disp,
                    f"SELECTED -> X={x_cmd:.1f} Y={y_cmd:.1f} U={u_cmd:.1f}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                now = time.time()
                moved_enough = True
                if last_xy is not None:
                    dx = x_cmd - last_xy[0]
                    dy = y_cmd - last_xy[1]
                    moved_enough = (dx * dx + dy * dy) ** 0.5 >= MIN_MOVE_MM

                if SEND_ENABLED and (now - last_sent_t) >= SEND_PERIOD and moved_enough:
                    print(f"[AUTO] Queueing GOP: X={x_cmd:.3f} Y={y_cmd:.3f} Z={ROBOT_Z_FIXED:.3f} U={u_cmd:.3f}")
                    robot.send_gop(x_cmd, y_cmd, ROBOT_Z_FIXED, u_cmd, wait=False)
                    last_sent_t = now
                    last_xy = (x_cmd, y_cmd)

            else:
                cv2.putText(
                    disp, "No target detected", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
                )

            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("c"):
                print("[INFO] Recalibrating: pausing auto-send, moving robot out of view, then calibrating...")
                SEND_ENABLED = False
                robot.flush_queue()
                robot.send_cmd("GOJ;90;0;0;0", wait=True)

                homography = calibration_loop(picam2)
                last_cmd = None
                last_xy = None
                last_sent_t = 0.0

                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
                cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)

            elif key == ord("p"):
                SEND_ENABLED = not SEND_ENABLED
                state = "ENABLED (continuous)" if SEND_ENABLED else "PAUSED"
                print(f"[INFO] Robot sending: {state}")
                if not SEND_ENABLED:
                    robot.flush_queue()

            elif key == ord("s"):
                if SEND_ENABLED:
                    print("[INFO] Step ignored (continuous sending is ON). Press 'p' to pause first.")
                else:
                    if last_cmd is None:
                        print("[INFO] No target available to step-send yet.")
                    else:
                        x_cmd, y_cmd, u_cmd = last_cmd
                        print(f"[STEP] Sending GOP: X={x_cmd:.3f} Y={y_cmd:.3f} Z={ROBOT_Z_FIXED:.3f} U={u_cmd:.3f}")
                        robot.send_gop(x_cmd, y_cmd, ROBOT_Z_FIXED, u_cmd, wait=False)
                        last_xy = (x_cmd, y_cmd)
                        last_sent_t = time.time()

    finally:
        cv2.destroyAllWindows()
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            console_thread.stop()
        except Exception:
            pass
        robot.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()