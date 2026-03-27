# This example calculates program cycle time as a function of
# joint speed and joint acceleration and writes results to CSV

from robodk.robolink import *  # RoboDK API
import csv
import os

RDK = Robolink()

# Select program
program = RDK.ItemUserPick('Select a program', ITEM_TYPE_PROGRAM)
if not program.Valid():
    raise Exception("No program selected.")

# Get linked robot
robot = program.getLink(ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception("No robot linked to program.")

results = []

# -----------------------------
# SPEED SWEEP
# -----------------------------
for speed_joints in range(20, 3400, 50):

    robot.setAccelerationJoints(1645.795918)
    robot.setSpeed(500, speed_joints)

    result = program.Update()
    instructions, time_val, travel, ok, error = result

    print("Speed:", speed_joints)
    print("Cycle Time:", time_val)

    results.append([speed_joints, time_val, "", ""])

# -----------------------------
# ACCELERATION SWEEP
# -----------------------------
for accel_joints in range(100, 15100, 100):

    robot.setAccelerationJoints(accel_joints)
    robot.setSpeed(1, 420)

    result = program.Update()
    instructions, time_val, travel, ok, error = result

    print("Acceleration:", accel_joints)
    print("Cycle Time:", time_val)

    results.append(["", "", accel_joints, time_val])

# -----------------------------
# WRITE CSV
# -----------------------------
filename = "robodk_cycle_time_results.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["joint_speed", "cycle_time", "joint_accel", "cycle_time"])
    writer.writerows(results)

print("\nResults saved to:", os.path.abspath(filename))
