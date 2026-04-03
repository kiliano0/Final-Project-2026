Final Versions:
Vision System - CameraMotion0603.py
Energy Model - EnergyModel4.py
EpsonRC Framework - FinalFrameworkwithMoves.prg (For use with vision system. Follow-up moves can be removed.)

How to run Energy Model. Examples used for calibration. First CSV is RoboDK files, second is the data from power analyser.
.\venv\Scripts\python.exe EnergyModel4.py --csv 0303_Joint1_Movement_with_waits.csv --calibration .\1403Power\J1_combined_max.CSV

.\venv\Scripts\python.exe EnergyModel4.py --csv 0303_Joint2_Movements_with_waits.csv --calibration .\1403Power\J2_combined_max.CSV

.\venv\Scripts\python.exe EnergyModel4.py --csv 0303_Joint3_Movements_with_waits.csv --calibration .\1403Power\J3_combined_max.CSV

.\venv\Scripts\python.exe EnergyModel4.py --csv Joint4_RoboDK_350Deg_with_waits.csv --calibration .\1403Power\J4_combined_max.CSV

How to run Batch Energy Model. input-dir is where all the files are taken from.
C:/Users/kytho/Documents/Energy_Model/venv/Scripts/python.exe EnergyModel_Batch.py --input-dir "C:/Users/kytho/Desktop/RoboDK_Exports" --output-csv Energy_summary.csv --time-col Time_Total --power-mode total_energy_j --plot-power-dir "C:/Users/kytho/Documents/Energy_Model/power_plots_RoboDK"

RoboDK Scripts
Before running any of these ensure that the robot is in the starting position for the program.
GetJointData.py (Output CSV file for a single program)
MultipleSpeedAccelOutput.py (Output CSV files for a range of accelerations and speeds for the same program. Make set there is no Set Speed within the program)

No Longer used because iterative method was improved upon.
FullOutputCSVJointMove.py (Outputs cycle times for a program over a range of Joint speeds and accelerations)
FullOutputCSVLinearMove.py (Outputs cycle times for a program over a range of linear speeds and accelerations)

Under Results:
Speed Data.csv (contains the real speeds mapped to the commanded speeds)
AccelCurveData.csv (contains the Energy Model and Power Analyser Results. Final and organised values are found under Power Calculations tab)
