Function showPosition
    Real PreviousSpeed, check

    TmReset 1
    OpenNet #201 As Server
    WaitNet #201
    Print #201, "READY"

    PreviousSpeed = gSpeedVar   ' initialise so first loop is clean

    Do
        check = gSpeedVar - PreviousSpeed
        If check <> 0 Then
            Print #201, "EVENT,SPEED,", gSpeedVar, ",", ",", ",", Tmr(1)
        EndIf

        P99 = RealPos
        Print #201, "DATA,", CX(P99), ",", CY(P99), ",", CZ(P99), ",", CU(P99), ",", Tmr(1)
		PreviousSpeed = gSpeedVar
        Wait 0.01
    Loop
Fend
