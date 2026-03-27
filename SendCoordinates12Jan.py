import socket

IP = "192.168.1.5"
PORT = 2000

commands = [
    "GOP;-400.000;200.000;-20.000;180.000",  # Point1
    "GOP;-400.000;0.000;-20.000;180.000",    # Point2
    "GOP;-200.000;0.000;-20.000;180.000",    # Point3
    "GOP;-200.000;200.000;-20.000;180.000",  # Point4
]

with socket.create_connection((IP, PORT), timeout=5) as sock:
    r = sock.makefile("r", encoding="ascii")
    w = sock.makefile("w", encoding="ascii")

    # Initial handshake
    print("Controller:", r.readline().strip())

    sock.settimeout(120)
    
    for cmd in commands:
        print("Sending:", cmd)
        w.write(cmd + "\r\n")
        w.flush()
        reply = r.readline().strip()

        # Wait until robot finishes motion
        reply = r.readline().strip()
        print("Controller:", reply)

    # Exit cleanly
    w.write("EXIT\r\n")
    w.flush()
    print("Controller:", r.readline().strip())