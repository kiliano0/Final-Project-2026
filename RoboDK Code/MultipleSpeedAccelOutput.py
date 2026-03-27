from robodk.robolink import Robolink, ITEM_TYPE_PROGRAM, ITEM_TYPE_ROBOT
import csv
import os

# ----------------------------
# USER SETTINGS
# ----------------------------
PROGRAM_NAME = "Prog3"
OUT_DIR = r"C:\Users\kytho\Desktop\RoboDK_Exports"

# Your discrete sweep values (absolute units)
MAX_JOINT_SPEED = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]          # deg/s
MAX_JOINT_ACCEL = [192, 405.4, 595.6, 790.3, 975.1, 1138.93, 1295, 1418,
                   1520.4, 1645.8]                                          # deg/s^2

# InstructionListJoints sampling resolution
MM_STEP = 1
DEG_STEP = 1
FLAGS = 3

# ----------------------------
# SETUP
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)

RDK = Robolink()

program = RDK.Item(PROGRAM_NAME, ITEM_TYPE_PROGRAM)
if not program.Valid():
    raise Exception(f"Program not found: {PROGRAM_NAME}")

robot = RDK.Item("", ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception("No robot item found in the station.")

# 4-axis header (as you requested)
header = [
    'J1_Position', 'J2_Position', 'J3_Position', 'J4_Position', 'Error',
    'MM_Step', 'Deg_step', 'Move_ID', 'Time', 'Time_Total', 'X', 'Y', 'Z',
    'J1_Velocity', 'J2_Velocity', 'J3_Velocity', 'J4_Velocity',
    'J1_Acceleration', 'J2_Acceleration', 'J3_Acceleration', 'J4_Acceleration'
]

summary_csv = os.path.join(OUT_DIR, "Summary_CycleTimes.csv")
with open(summary_csv, "w", newline="") as sf:
    sw = csv.writer(sf)
    sw.writerow(["Speed_%", "Accel_%", "Speed_abs", "Accel_abs",
                 "CycleTime_s", "OK", "ErrorMsg", "File", "RowsWritten"])

    # ----------------------------
    # MAIN SWEEP
    # ----------------------------
    for i_speed, speed_abs in enumerate(MAX_JOINT_SPEED):
        speed_pct = (i_speed + 1) * 10  # 10..100 labels

        for i_accel, accel_abs in enumerate(MAX_JOINT_ACCEL):
            accel_pct = (i_accel + 1) * 10  # 10..100 labels

            # Apply limits to robot (affects timing when the program is updated)
            robot.setSpeed(500, speed_abs)            # setSpeed(linear_mm_s, joints_deg_s)
            robot.setAccelerationJoints(accel_abs)    # joints accel (deg/s^2)

            # Update program timing with the new speed/accel
            instructions, time_val, travel, ok, error = program.Update()
            if not ok:
                sw.writerow([speed_pct, accel_pct, speed_abs, accel_abs,
                             time_val, ok, error, "", 0])
                print(f"Update failed: Speed={speed_pct} Accel={accel_pct} | {error}")
                continue

            # Extract joint trajectory data (positions/vel/acc)
            data_all = program.InstructionListJoints(mm_step=MM_STEP, deg_step=DEG_STEP, flags=FLAGS)

            # IMPORTANT: match your working script structure
            # Your working version does: data_all = data_all[1:] then joint_data = data_all[0]
            # That corresponds to using data_all[1] here.
            if not data_all or len(data_all) < 2:
                sw.writerow([speed_pct, accel_pct, speed_abs, accel_abs,
                             time_val, False, "InstructionListJoints returned <2 blocks", "", 0])
                print(f"No data blocks: Speed={speed_pct} Accel={accel_pct}")
                continue

            joint_rows = data_all[1]  # <-- key fix

            if not joint_rows or len(joint_rows) == 0:
                sw.writerow([speed_pct, accel_pct, speed_abs, accel_abs,
                             time_val, False, "Trajectory block empty (data_all[1])", "", 0])
                print(f"Empty trajectory: Speed={speed_pct} Accel={accel_pct}")
                continue

            # Write per-run CSV
            out_name = f"Speed{speed_pct:02d}_Accel{accel_pct:02d}_AllFourJoints.csv"
            out_path = os.path.join(OUT_DIR, out_name)

            time_total = 0.0
            rows_written = 0

            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)

                for row in joint_rows:
                    new_row = list(row)

                    # Time delta at index 8 (same as your working script)
                    try:
                        t_delta = float(row[8])
                    except Exception:
                        t_delta = 0.0

                    time_total += t_delta
                    new_row.insert(9, time_total)  # Time_Total right after Time
                    w.writerow(new_row)
                    rows_written += 1

            # Log summary line
            sw.writerow([speed_pct, accel_pct, speed_abs, accel_abs,
                         time_val, ok, error, out_name, rows_written])

            print(f"Saved: {out_name} | Rows: {rows_written} | CycleTime: {time_val} s | OK: {ok}")

print(f"\nDone. Files saved in: {OUT_DIR}")
print(f"Cycle time summary: {summary_csv}")
