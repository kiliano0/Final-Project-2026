
# --- Setup ---
from picamera2 import Picamera2
import time
import sys
import cv2
import numpy as np

MAX_RESOLUTION = (4056, 3040)  # Pi HQ Camera max resolution
PROC_WIDTH = 1280  # processing width for faster detection
WINDOW_NAME = 'Pi Camera Calibration & Detection'

# --- Calibration ---
def capture_calibration_image(picam2):
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr

def calibration_loop(picam2, world_points):
    image_points = []
    homography = None
    calib_img = capture_calibration_image(picam2)
    msg = "Click 4 points (corners), 'n'=new image, 'c'=calibrate, 'r'=reset, 'q'=quit"

    def mouse_cb(event, x, y, flags, param):
        nonlocal image_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(image_points) < 4:
                image_points.append([float(x), float(y)])
                print(f"Clicked image point {len(image_points)}: ({x}, {y})")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    while True:
        disp = calib_img.copy()
        for idx, (px, py) in enumerate(image_points):
            cv2.circle(disp, (int(px), int(py)), 8, (0, 255, 255), -1)
            cv2.putText(disp, str(idx + 1), (int(px) + 10, int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(disp, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            print("Quit requested during calibration")
            sys.exit(0)
        elif key == ord('r'):
            image_points = []
            print("Calibration reset")
        elif key == ord('n'):
            calib_img = capture_calibration_image(picam2)
            image_points = []
            print("New calibration image captured")
        elif key == ord('c'):
            if len(image_points) != 4:
                print("Need 4 image points to compute homography. Click 4 points first.")
                continue
            # Ask user to input real-world coordinates (mm) for each clicked point
            print("Enter real-world coordinates (in mm) for each clicked image point in the SAME order you clicked them.")
            print("Type in 'x y' (e.g. '0 0'). Type 'cancel' to abort and continue clicking.")
            user_world_pts = []
            aborted = False
            for i in range(4):
                while True:
                    try:
                        s = input(f"Point {i+1} image coords {image_points[i]} -> enter world X Y (mm): ").strip()
                    except EOFError:
                        s = 'cancel'
                    if s.lower() == 'cancel':
                        aborted = True
                        break
                    parts = s.split()
                    if len(parts) != 2:
                        print("Invalid input. Enter two numbers separated by space or 'cancel'.")
                        continue
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                    except ValueError:
                        print("Invalid numbers. Try again or type 'cancel'.")
                        continue
                    user_world_pts.append([x, y])
                    break
                if aborted:
                    break
            if aborted:
                print("Calibration aborted — continue clicking or press 'n' for new image.")
                continue

            user_world_np = np.array(user_world_pts, dtype=np.float32)
            # Ensure at least one point is (0,0) — prompt user if not
            if not any(np.allclose(p, [0.0, 0.0]) for p in user_world_np):
                print("Warning: none of the entered points is (0,0). It's recommended to have one origin point.")
                yn = input("Proceed anyway? (y/n): ").strip().lower()
                if yn != 'y':
                    print("Calibration cancelled — please re-enter coordinates or capture new image.")
                    continue

            img_pts_np = np.array(image_points, dtype=np.float32)
            H, status = cv2.findHomography(img_pts_np, user_world_np)
            if H is None:
                print("Homography computation failed")
                image_points = []
                continue
            homography = H.astype(np.float32)
            print("Homography computed. Proceeding to detection.")
            break
    cv2.destroyWindow(WINDOW_NAME)
    return homography

# --- Detection ---
def detect_rectangle(frame_bgr, homography):
    # frame_bgr is full-resolution image. We'll perform detection on a resized copy
    h_full, w_full = frame_bgr.shape[:2]
    proc_w = PROC_WIDTH
    proc_h = int((proc_w * h_full) / w_full)
    proc = cv2.resize(frame_bgr, (proc_w, proc_h))

    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad = None
    best_area = 0
    full_area = w_full * h_full
    area_min = max(2000, 0.0002 * full_area)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (area_min * (proc_w * proc_h) / full_area):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best_quad = approx

    result = {}
    if best_quad is not None and homography is not None:
        # scale quad back to full resolution
        scale_x = w_full / proc_w
        scale_y = h_full / proc_h
        quad_proc = best_quad.reshape(4, 2)
        quad_full = np.array([[pt[0] * scale_x, pt[1] * scale_y] for pt in quad_proc], dtype=np.float32)
        rect = cv2.minAreaRect(quad_full)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        (cx, cy), (w, h), angle = rect
        center_pixel = np.array([[[cx, cy]]], dtype=np.float32)
        try:
            real_pt = cv2.perspectiveTransform(center_pixel, homography)
            rx, ry = real_pt[0][0]
        except Exception:
            # if homography is invalid or transform fails, skip
            return {}
        result = {
            'box': box,
            'center_px': (cx, cy),
            'center_mm': (rx, ry),
            'angle': angle
        }
    return result

# --- Main Loop ---
def main():
    # Real-world coordinates for calibration (adjust to your table size)
    WORLD_POINTS = np.array([
        [0.0, 0.0],
        [1200.0, 0.0],
        [1200.0, 900.0],
        [0.0, 900.0],
    ], dtype=np.float32)

    print("Initializing camera at max resolution...")
    picam2 = Picamera2()
    config = picam2.create_still_configuration({'size': MAX_RESOLUTION})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("Starting calibration...")
    homography = calibration_loop(picam2, WORLD_POINTS)

    print("Starting detection loop. Press 'q' in the window to quit. Press 'n' to capture new calibration image at any time.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Capture as fast as possible at full resolution
            frame_bgr = capture_calibration_image(picam2)
            try:
                result = detect_rectangle(frame_bgr, homography)
            except Exception as e:
                print(f"Detection error (skipping frame): {e}")
                result = {}

            disp = frame_bgr.copy()
            if result:
                box = result['box']
                cx, cy = result['center_px']
                rx, ry = result['center_mm']
                angle = result['angle']
                cv2.drawContours(disp, [box], 0, (0, 255, 0), 3)
                cv2.circle(disp, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                label = f"X={rx:.1f}mm Y={ry:.1f}mm a={angle:.1f}deg"
                cv2.putText(disp, label, (int(cx) - 120, int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                print(f"Rectangle: center=({rx:.1f}mm, {ry:.1f}mm), angle={angle:.1f}")
            else:
                cv2.putText(disp, "No rectangle detected", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow(WINDOW_NAME, disp)

            # Poll keys with minimal delay to stay responsive
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('n') or key == ord('r') or key == ord('c'):
                print("Entering calibration (requested by user)")
                homography = calibration_loop(picam2, WORLD_POINTS)
                print("Resuming detection")
    finally:
        cv2.destroyAllWindows()
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == '__main__':
    main()