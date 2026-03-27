import socket
import tkinter as tk
from tkinter import messagebox
import threading
import queue

# Robot connection
IP = "192.168.1.5"
PORT = 2000

# Workspace in robot units (your rectangle)
X_MIN, X_MAX = -400.0, -200.0
Y_MIN, Y_MAX = 0.0, 200.0

# Fixed Z and U
Z_FIXED = -20.0
U_FIXED = 180.0

# UI sizes
CANVAS_W = 600
CANVAS_H = 400
PAD = 40

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def recv_line(sock, timeout=120):
    sock.settimeout(timeout)
    data = b""
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("Controller closed the connection")
        if b == b"\n":
            return data.decode("ascii", errors="replace").strip()
        data += b

class RobotWorker(threading.Thread):
    """Background thread: owns the socket and handles send/receive."""
    def __init__(self, status_cb):
        super().__init__(daemon=True)
        self.status_cb = status_cb
        self.cmd_q = queue.Queue()
        self.sock = None
        self.running = True

    def connect(self):
        self.status_cb("Connecting...")
        self.sock = socket.create_connection((IP, PORT), timeout=5)
        # Expect initial READY
        ready = recv_line(self.sock, timeout=10)
        self.status_cb(f"Connected: {ready}")

    def run(self):
        try:
            self.connect()
        except Exception as ex:
            self.status_cb(f"Connection failed: {ex}")
            self.running = False
            return

        while self.running:
            try:
                cmd = self.cmd_q.get()  # blocks until command
                if cmd is None:
                    break

                # Send command with CRLF (often required)
                self.status_cb(f"Sending: {cmd}")
                self.sock.sendall((cmd + "\r\n").encode("ascii"))

                # One reply per command (READY after motion completes)
                reply = recv_line(self.sock, timeout=120)
                self.status_cb(f"Controller: {reply}")

            except Exception as ex:
                self.status_cb(f"Comm error: {ex}")
                break

        # Cleanup
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

    def send(self, cmd: str):
        if self.running:
            self.cmd_q.put(cmd)

    def stop(self):
        self.running = False
        self.cmd_q.put(None)

class RobotGUI:
    def __init__(self, root):
        self.root = root
        root.title("Robot Click-to-Move (XY)")

        self.status = tk.StringVar(value="Starting...")
        tk.Label(root, textvariable=self.status, anchor="w").pack(fill="x", padx=10, pady=6)

        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H)
        self.canvas.pack(padx=10, pady=10)

        # Workspace rectangle in pixels
        self.rect_left = PAD
        self.rect_top = PAD
        self.rect_right = CANVAS_W - PAD
        self.rect_bottom = CANVAS_H - PAD

        self.canvas.create_rectangle(
            self.rect_left, self.rect_top, self.rect_right, self.rect_bottom, width=2
        )
        self.canvas.create_text(
            CANVAS_W // 2, 15,
            text=f"Robot XY: X[{X_MIN}..{X_MAX}]  Y[{Y_MIN}..{Y_MAX}]",
        )

        self.marker = None

        btn = tk.Frame(root)
        btn.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(btn, text="HOME", command=lambda: self.worker.send("HOME")).pack(side="left")
        tk.Button(btn, text="EXIT", command=self.on_exit).pack(side="right")

        # Bind click
        self.canvas.bind("<Button-1>", self.on_click)

        # Start worker thread
        self.worker = RobotWorker(self.thread_safe_status)
        self.worker.start()

    def thread_safe_status(self, text):
        # Tk updates must happen on the main thread
        self.root.after(0, lambda: self.status.set(text))

    def pixel_to_robot_xy(self, px, py):
        # clamp inside rectangle
        px = clamp(px, self.rect_left, self.rect_right)
        py = clamp(py, self.rect_top, self.rect_bottom)

        nx = (px - self.rect_left) / (self.rect_right - self.rect_left)
        ny = (py - self.rect_top) / (self.rect_bottom - self.rect_top)

        x = X_MIN + nx * (X_MAX - X_MIN)
        y = Y_MAX - ny * (Y_MAX - Y_MIN)  # invert Y

        return x, y

    def draw_marker(self, px, py):
        if self.marker is not None:
            self.canvas.delete(self.marker)
        r = 6
        self.marker = self.canvas.create_oval(px-r, py-r, px+r, py+r, width=2)

    def on_click(self, event):
        px, py = event.x, event.y

        # ignore clicks outside rectangle
        if not (self.rect_left <= px <= self.rect_right and self.rect_top <= py <= self.rect_bottom):
            self.status.set("Click inside the rectangle to move.")
            return

        self.draw_marker(px, py)
        x, y = self.pixel_to_robot_xy(px, py)

        cmd = f"GOP;{x:.3f};{y:.3f};{Z_FIXED:.3f};{U_FIXED:.3f}"
        self.worker.send(cmd)

    def on_exit(self):
        try:
            self.worker.send("EXIT")
        except Exception:
            pass
        self.worker.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()
