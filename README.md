These repository contains all the files and scripts for my FYP. Everything is labelled clearly. The Epson_RC170_Drivers need to be unzipped before being installed through the device manager. There is now a computer in the lab with the Epson RC 5.0+ on it so you might not need them.

The final versions of each code are given below
Final Versions:
Vision System - CameraMotion0603.py
Energy Model - EnergyModel4.py
EpsonRC Framework - FinalFrameworkwithMoves.prg (For use with vision system. Follow-up moves can be removed.)

Vision System
The vision system has the instructions for running it in the top left. The file just needs to be run on a Raspberry Pi with the Pi Camera connected. Should be connected to the RC180 Controller via Ethernet so it can send the coordinates. It will automatically connect once the FinalFramework is running on the Controller, provided the Raspberry Pi is on the same subnet. I had them both on 192.169.1.XX. Make sure the final value is different on each device so they can actually connect.

Select 4 calibration points and press "C" to calibrate. Press "N" to take a new image if there was something in the way. Press "S" to send a command if the sending is paused. Press "P" to pause and unpause sending. Sending is by default on pause so the robot won't suddenly start moving.

Commands can also be sent directly from the Terminal.
Speed;XX
Accel;XX
GOP;XX;XX;XX;XX
GOJ;XX;XX;XX;XX
TAKT
HOME

TigerVNC was used to remote access the Raspberry Pi. Need to make sure SSH and VNC are open when setting up the OS, or you can change it from the settings later. Means everything can you controlled from your laptop.

How to run Energy Model. Examples used for calibration. First CSV is RoboDK files, second is the data from power analyser. Output data from the power analyser needs to be change to comma seperated instead of semi-colon seperated. Also need to add a time value starting from 0 seconds.
.\venv\Scripts\python.exe EnergyModel4.py --csv 0303_Joint1_Movement_with_waits.csv --calibration .\1403Power\J1_combined_max.CSV

.\venv\Scripts\python.exe EnergyModel4.py --csv 0303_Joint2_Movements_with_waits.csv --calibration .\1403Power\J2_combined_max.CSV

.\venv\Scripts\python.exe EnergyModel4.py --csv 0303_Joint3_Movements_with_waits.csv --calibration .\1403Power\J3_combined_max.CSV

.\venv\Scripts\python.exe EnergyModel4.py --csv Joint4_RoboDK_350Deg_with_waits.csv --calibration .\1403Power\J4_combined_max.CSV

How to run Batch Energy Model. input-dir is where all the files are taken from.
C:/Users/kytho/Documents/Energy_Model/venv/Scripts/python.exe EnergyModel_Batch.py --input-dir "C:/Users/kytho/Desktop/RoboDK_Exports" --output-csv Energy_summary.csv --time-col Time_Total --power-mode total_energy_j --plot-power-dir "C:/Users/kytho/Documents/Energy_Model/power_plots_RoboDK"

How to extract the 100 Seperate moves from the Power Analyser Data (This was done for all ten sets of data)
cd "C:\Users\kytho\Documents\Energy_Model"
python .\extract_move_energy_grid.py --input ".\AllSpeedsAllAccelsRealRobot\PowerAllFour2.csv" --output ".\move_energy_grid2.csv" --plot-output ".\plots2\detected_regions.png" ` --plot-detail-dir ".\plots2\detail_pages"

RoboDK:
RoboDK simulation files are under "RoboDK files". This is the Demo, test files etc. The data extraction files are in "RoboDK Code".

Before running any of these ensure that the robot is in the starting position for the program.
GetJointData.py (Output CSV file for a single program)
MultipleSpeedAccelOutput.py (Output CSV files for a range of accelerations and speeds for the same program. Make set there is no Set Speed within the program)

You can change the program it is referring to in the script. To run it for any RoboDK file, you just need to have it open in the left side menu.

No Longer used because iterative method was improved upon:
FullOutputCSVJointMove.py (Outputs cycle times for a program over a range of Joint speeds and accelerations)
FullOutputCSVLinearMove.py (Outputs cycle times for a program over a range of linear speeds and accelerations)

Under Results:
Speed Data.csv (contains the real speeds mapped to the commanded speeds)
AccelCurveData.csv (contains the Energy Model and Power Analyser Results. Final and organised values are found under Power Calculations tab)




