from robodk import robolink, robomath
import matplotlib.pyplot as plt
import csv

# Connect to RoboDK
RDK = robolink.Robolink()

# Select your program by name
program = RDK.Item('Prog3', robolink.ITEM_TYPE_PROGRAM)  # Change name as needed

# Run the program (optional — comment out if already simulated)
program.RunProgram()

# Retrieve joint data
data_all = program.InstructionListJoints(mm_step=1, deg_step=1, flags=3)
data_all = data_all[1:]

joint_data = data_all[0]
print(joint_data)

# Output file path (can be absolute or relative)
csv_file = 'C:/Users/kytho/Desktop/robot_data.csv'

if program.Valid():
    # Write to CSV

    # Add header row (titles) and append a cumulative Time_Total column
    header = [
        'J1_Position', 'J2_Position', 'J3_Position', 'J4_Position', 'Error',
        'MM_Step', 'Deg_step', 'Move_ID', 'Time', 'Time_Total', 'X', 'Y', 'Z',
        'J1_Velocity', 'J2_Velocity', 'J3_Velocity', 'J4_Velocity',
        'J1_Acceleration', 'J2_Acceleration', 'J3_Acceleration', 'J4_Acceleration'
    ]

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        # write header first (move all data one row down)
        writer.writerow(header)

        # compute cumulative Time_Total (Time is expected at index 8)
        time_total = 0.0
        for row in joint_data:
            # keep the original data unchanged; append Time_Total after Time
            new_row = list(row)
            try:
                # read time delta from column index 8
                t_delta = float(row[8])
            except Exception:
                # if parsing fails, treat delta as 0.0
                t_delta = 0.0
            time_total += t_delta
            # insert cumulative time after the existing Time column
            new_row.insert(9, time_total)
            writer.writerow(new_row)
        
    print("Joint data successfully written to 'robot_data.csv'")
else:
    print("No valid program selected.")
