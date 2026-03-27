import socket
import time
import sys
import cv2
import numpy as np
import threading
import queue

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
# IMPORTANT: set these to match the coordinates you enter at calibration time.
# Example: if you enter (0,0), (1200,0), (1200,900), (0,900)
WORLD_W = 1200.0
WORLD_H = 900.0

# If camera "world" coords are not the same origin as robot coords,
# apply offsets here (mm) AFTER mapping:
ROBOT_X_OFFSET = 0.0
ROBOT_Y_OFFSET = 0.0

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
WINDOW_NAME = "Pi Camera Calibration & Detection"

# ----------------------------
# Sending control
# ----------------------------
SEND_ENABLED = True   # toggle with 'p'

# --- Black-rectangle detection tuning ---
BLACK_V_MAX = 80           # lower = stricter "black"; higher = more sensitive
BLACK_S_MAX = 120          # allow low/medium saturation (helps with shadows)
MIN_AREA_FRAC = 0.00008    # smaller = more sensitive to small rectangles
APPROX_EPS_FRAC = 0.02     # smaller = more corners/noisy; larger = smoother
MORPH_K = 7                # larger = closes gaps more; too large can blob stuff

RECONNECT_DELAY = 1.0  # seconds between reconnect attempts


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def recv_line(sock: socket.socket, timeout=120) -> str:
    sock.settimeout(timeout)
    data = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("Controller closed the connection")
        if b in (b"\n", b"\r"):
            # Consume a following \n after \r (handles \r\n)
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
    - Takes latest target from a queue
    - Sends one command, waits for READY, then takes next
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
        """Drop any queued (stale) targets."""
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def send_target(self, x, y, z, u):
        """Enqueue the latest target only."""
        self.flush_queue()
        self.q.put((x, y, z, u))

    def run(self):
        while self.running:
            # Ensure connected (keep trying forever)
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

            # Wait for a target
            item = self.q.get()
            if item is None:
                break

            x, y, z, u = item
            cmd = f"GOP;{x:.3f};{y:.3f};{z:.3f};{u:.3f}"

            try:
                print(f"Sending: {cmd}")
                self.sock.sendall((cmd + "\r\n").encode("ascii"))
                reply = recv_line(self.sock, timeout=120)
                print("Robot:", reply)

                # Treat ANY reply as the end of the transaction.
                # If the robot reports an error, keep running, but flush queued targets
                # so we don't chase stale points while the cell is in a bad state.
                if reply.upper().startswith("ERROR"):
                    print("[RobotWorker] Robot returned ERROR (continuing). Flushing queued targets.")
                    self.flush_queue()

                self.last_sent = (x, y, z, u)

            except socket.timeout:
                print("[RobotWorker] Reply timeout (no READY). Dropping connection and retrying...")
                self.disconnect()

            except Exception as e:
                print(f"[RobotWorker] Comm error: {e}. Dropping connection and retrying...")
                self.disconnect()

        # Clean shutdown
        try:
            if self.sock:
                try:
                    self.sock.sendall(b"EXIT\r\n")
                except Exception:
                    pass
                self.disconnect()
        except Exception:
            pass



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

            print("Enter world coords (mm) for each clicked image point in same order.")
            user_world_pts = []
            for i in range(4):
                while True:
                    s = input(f"Point {i+1} -> enter world X Y (mm): ").strip()
                    s = s.replace(",", " ")
                    parts = s.split()
                    if len(parts) != 2:
                        print("Enter: x y   (comma or space is ok)")
                        continue
                    try:
                        x = float(parts[0]); y = float(parts[1])
                    except ValueError:
                        print("Invalid numbers, try again.")
                        continue
                    user_world_pts.append([x, y])
                    break

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


def detect_rectangle(frame_bgr, homography):
    h_full, w_full = frame_bgr.shape[:2]

    proc_w = PROC_WIDTH
    proc_h = int((proc_w * h_full) / w_full)
    proc = cv2.resize(frame_bgr, (proc_w, proc_h))

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    Hc, S, V = cv2.split(hsv)

    black_mask = cv2.inRange(hsv, (0, 0, 0), (179, BLACK_S_MAX, BLACK_V_MAX))
    black_mask = cv2.GaussianBlur(black_mask, (5, 5), 0)

    k = MORPH_K
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    proc_area = float(proc_w * proc_h)
    area_min = max(800.0, MIN_AREA_FRAC * proc_area)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_FRAC * peri, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        aspect = (w / h) if w >= h else (h / w)
        if aspect > 12.0:
            continue

        mask_cnt = np.zeros(black_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_cnt, [approx], -1, 255, thickness=-1)
        mean_v = cv2.mean(V, mask=mask_cnt)[0]
        if mean_v > (BLACK_V_MAX + 15):
            continue

        if area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None or homography is None:
        return {}

    scale_x = w_full / proc_w
    scale_y = h_full / proc_h
    quad_proc = best_quad.reshape(4, 2).astype(np.float32)
    quad_full = np.array([[pt[0] * scale_x, pt[1] * scale_y] for pt in quad_proc], dtype=np.float32)

    rect = cv2.minAreaRect(quad_full)
    box = cv2.boxPoints(rect).astype(int)
    (cx, cy), (_, _), angle = rect

    center_pixel = np.array([[[cx, cy]]], dtype=np.float32)
    real_pt = cv2.perspectiveTransform(center_pixel, homography)
    rx, ry = real_pt[0][0]

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

    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_still_configuration({"size": MAX_RESOLUTION})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("Calibration...")
    homography = calibration_loop(picam2)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_sent_t = 0.0
    last_xy = None

    # Remember the latest computed command so 's' can send it while paused
    last_cmd = None  # (x, y)

    try:
        while True:
            frame_bgr = capture_frame_bgr(picam2)
            result = detect_rectangle(frame_bgr, homography)

            disp = frame_bgr.copy()

            status = "SEND: ON" if SEND_ENABLED else "SEND: PAUSED (press 's' to step)"
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
                cv2.putText(
                    disp, f"X={rx:.1f}mm Y={ry:.1f}mm a={angle:.1f}",
                    (int(cx) - 160, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3
                )

                print(f"Rectangle: center=({rx:.1f}mm, {ry:.1f}mm), angle={angle:.1f}")

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
                cv2.putText(
                    disp, "No rectangle detected", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
                )

            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                print("Recalibrating...")
                homography = calibration_loop(picam2)
                last_cmd = None
                last_xy = None
                last_sent_t = 0.0
            elif key == ord("p"):
                SEND_ENABLED = not SEND_ENABLED
                state = "ENABLED" if SEND_ENABLED else "PAUSED"
                print(f"[INFO] Robot sending {state}")
                if not SEND_ENABLED:
                    robot.flush_queue()  # drop stale commands when pausing
            elif key == ord("s"):
                # Single-step: only works while paused
                if SEND_ENABLED:
                    print("[INFO] Step ignored (sending is ON). Press 'p' to pause first.")
                else:
                    if last_cmd is None:
                        print("[INFO] No target available to step-send yet.")
                    else:
                        x_cmd, y_cmd = last_cmd
                        print(f"[STEP] Sending one target: X={x_cmd:.3f} Y={y_cmd:.3f} Z={ROBOT_Z_FIXED:.3f} U={ROBOT_U_FIXED:.3f}")
                        robot.send_target(x_cmd, y_cmd, ROBOT_Z_FIXED, ROBOT_U_FIXED)
                        # update last_xy so MIN_MOVE_MM is measured from the stepped point
                        last_xy = (x_cmd, y_cmd)
                        last_sent_t = time.time()

    finally:
        cv2.destroyAllWindows()
        try:
            picam2.stop()
        except Exception:
            pass
        robot.stop()


if __name__ == "__main__":
    main()
