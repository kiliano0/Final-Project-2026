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
ROBOT_Z_FIXED = -159.0
ROBOT_U_FIXED = 180.0

# ----------------------------
# Camera "world" size (mm) used during homography entry
# ----------------------------
WORLD_W = 1200.0
WORLD_H = 900.0

CALIB_WINDOW_NAME = "Pi Camera Calibration"
WINDOW_NAME = "Pi Camera Live (Click to Move | P=Arm | M=Move | C=Recal | Q=Quit)"

# Offsets applied AFTER mapping (mm)
ROBOT_X_OFFSET = 0.0
ROBOT_Y_OFFSET = 0.0

# World coords (mm) corresponding to click order 1..4
AUTO_WORLD_PTS = [
    (250.0, 374.0),  # 1
    (250.0, 60.0),   # 2
    (416.0, 60.0),   # 3
    (160.0, 225.0),  # 4
]

# If your Y axis is flipped between camera-world and robot-world, set True
FLIP_Y = False

# Rate limiting / filtering
SEND_PERIOD = 0.5     # seconds between robot commands
MIN_MOVE_MM = 5.0     # only send if target changed by at least this much

# Camera settings
MAX_RESOLUTION = (4056, 3040)

# Safety default
SEND_ENABLED = False   # start disarmed

RECONNECT_DELAY = 1.0

def pick_max_mode_4by3(picam2, aspect=4/3, tol=0.06):
    """
    Pick the largest sensor mode close to 4:3 (prevents centre crop / punch-in).
    tol is relative aspect error tolerance.
    """
    best = None
    best_area = -1

    for m in picam2.sensor_modes:
        w, h = m["size"]
        a = w / h
        if abs(a - aspect) / aspect > tol:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = m

    # Fallback: just pick largest mode if no close 4:3 found
    if best is None:
        for m in picam2.sensor_modes:
            w, h = m["size"]
            area = w * h
            if area > best_area:
                best_area = area
                best = m

    return best


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
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.sock = None
        self.running = True

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
        # latest-only
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
    frame = picam2.capture_array("main")  # always the same stream
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def calibration_loop(picam2):
    """
    Click 4 image points in order, press 'c' to compute homography
    using AUTO_WORLD_PTS.
    """
    image_points = []
    homography = None
    calib_img = capture_frame_bgr(picam2)
    msg = "Click 4 points in order, 'n'=new image, 'c'=calibrate, 'r'=reset, 'q'=quit"

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

    # --- One fixed stream for BOTH calibration + live, at max 4:3 resolution ---
    mode = pick_max_mode_4by3(picam2)  # largest 4:3-ish mode
    STREAM_SIZE = mode["size"]
    print("[CAM] Selected sensor mode size:", STREAM_SIZE)

    config = picam2.create_preview_configuration(
        main={"size": STREAM_SIZE, "format": "RGB888"},
        controls={"FrameRate": 15}  # adjust if you want faster/slower
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    # Lock crop to the full sensor area (prevents unexpected centre-cropping)
    pa = picam2.camera_properties.get("PixelArraySize", None)
    if pa:
        picam2.set_controls({"ScalerCrop": (0, 0, pa[0], pa[1])})
        print("[CAM] Locked ScalerCrop to full PixelArraySize:", pa)

    # Sanity print: calibration + live MUST see the same shape
    test = picam2.capture_array("main")
    print("[CAM] main stream frame shape:", test.shape)


    # Move robot out of camera view before calibration
    print("[INFO] Moving robot out of camera view for calibration: GOJ;90;0;0;0")
    robot.send_raw("GOJ;90;0;0;0", flush=True)

    print("Calibration...")
    homography = calibration_loop(picam2)

    console_allowed.set()
    print("[INFO] Calibration done. Console passthrough is now ENABLED.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Click-to-move state
    clicked_px = None   # (x_px, y_px)
    clicked_mm = None   # (x_mm, y_mm)
    clicked_cmd = None  # (x_cmd, y_cmd)

    last_sent_t = 0.0
    last_xy = None

    def on_mouse_live(event, x, y, flags, param):
        nonlocal clicked_px, clicked_mm, clicked_cmd, last_sent_t, last_xy, homography

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if homography is None:
            print("[CLICK] Not calibrated yet.")
            return

        clicked_px = (float(x), float(y))

        pt = np.array([[[clicked_px[0], clicked_px[1]]]], dtype=np.float32)
        XY = cv2.perspectiveTransform(pt, homography)[0][0]
        rx, ry = float(XY[0]), float(XY[1])

        if FLIP_Y:
            ry = WORLD_H - ry

        clicked_mm = (rx, ry)

        x_cmd = rx + ROBOT_X_OFFSET
        y_cmd = ry + ROBOT_Y_OFFSET
        clicked_cmd = (x_cmd, y_cmd)

        print(f"[CLICK] px=({x},{y}) -> world=({rx:.1f},{ry:.1f}) -> cmd=({x_cmd:.1f},{y_cmd:.1f})")

        # If ARMED: send immediately (rate + min-move protected)
        if SEND_ENABLED:
            now = time.time()

            moved_enough = True
            if last_xy is not None:
                dx = x_cmd - last_xy[0]
                dy = y_cmd - last_xy[1]
                moved_enough = (dx * dx + dy * dy) ** 0.5 >= MIN_MOVE_MM

            if (now - last_sent_t) >= SEND_PERIOD and moved_enough:
                print(f"[CLICK->MOVE] Queueing: X={x_cmd:.3f} Y={y_cmd:.3f} Z={ROBOT_Z_FIXED:.3f} U={ROBOT_U_FIXED:.3f}")
                robot.send_target(x_cmd, y_cmd, ROBOT_Z_FIXED, ROBOT_U_FIXED)
                last_sent_t = now
                last_xy = (x_cmd, y_cmd)
            else:
                print("[CLICK->MOVE] Ignored (rate limit or too small move).")
        else:
            print("[CLICK] Stored target. Press 'm' to move while paused.")

    cv2.setMouseCallback(WINDOW_NAME, on_mouse_live)

    try:
        while True:
            frame_bgr = capture_frame_bgr(picam2)
            disp = frame_bgr.copy()

            status = "SEND: ARMED (click moves)" if SEND_ENABLED else "SEND: SAFE (PAUSED)  [P=Arm, Click stores, M=Move]"
            color = (0, 255, 0) if SEND_ENABLED else (0, 0, 255)
            cv2.putText(disp, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # Overlay last clicked target
            if clicked_px is not None and clicked_cmd is not None:
                cx, cy = int(clicked_px[0]), int(clicked_px[1])
                cv2.circle(disp, (cx, cy), 10, (255, 0, 255), 3)
                cv2.putText(disp, f"TARGET X={clicked_cmd[0]:.1f} Y={clicked_cmd[1]:.1f}",
                            (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break

            elif key == ord("p"):
                SEND_ENABLED = not SEND_ENABLED
                state = "ARMED" if SEND_ENABLED else "SAFE (PAUSED)"
                print(f"[INFO] Robot sending state: {state}")
                if not SEND_ENABLED:
                    robot.flush_queue()

            elif key == ord("m"):
                if clicked_cmd is None:
                    print("[M] No clicked target yet.")
                else:
                    x_cmd, y_cmd = clicked_cmd
                    print(f"[M] Sending clicked target: X={x_cmd:.3f} Y={y_cmd:.3f} Z={ROBOT_Z_FIXED:.3f} U={ROBOT_U_FIXED:.3f}")
                    robot.send_target(x_cmd, y_cmd, ROBOT_Z_FIXED, ROBOT_U_FIXED)
                    last_xy = (x_cmd, y_cmd)
                    last_sent_t = time.time()

            elif key == ord("c"):
                # Recalibrate
                console_allowed.clear()
                print("[INFO] Console passthrough DISABLED during calibration.")

                print("[INFO] Recalibrating: moving robot out of view: GOJ;90;0;0;0")
                robot.send_raw("GOJ;90;0;0;0", flush=True)

                print("Recalibrating...")
                homography = calibration_loop(picam2)

                console_allowed.set()
                print("[INFO] Calibration done. Console passthrough is now ENABLED.")

                clicked_px = None
                clicked_mm = None
                clicked_cmd = None
                last_xy = None
                last_sent_t = 0.0

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
