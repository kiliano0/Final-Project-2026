Function FinalFramework
    String line$, ch$, token$, msg$
    String cmd$, tag$
    String rest$
    Real X_Coord, Y_Coord, Z_Coord, U_Coord, v, overHead
    Real J1_Coord, J2_Coord, J3_Coord, J4_Coord
    Integer p, i, idx, e

    Motor On
    Power High
    Speed 10
	Accel 10, 10
    Go JA(90.000, 0.000, 0.000, 0.000)

Reconnect:
    OpenNet #201 As Server
    WaitNet #201
    Print #201, "READY"

    TmReset 0
	
	Retry:
    Do
        OnErr GoTo NetErr
        Input #201, line$
        OnErr GoTo 0

        Print line$

        ' -------------------------
        ' Basic commands
        ' -------------------------
        If line$ = "EXIT" Then
            Print #201, "BYE"
            GoTo ExitLine

        ElseIf line$ = "HOME" Then
            Go JA(0.000, 0.000, 0.000, 0.000)
            Print #201, "READY"
            GoTo Retry

        ElseIf line$ = "TAKT" Then
            overHead = Tmr(0)
            Print #201, overHead
            Print #201, "READY"
            TmReset 0
            GoTo Retry
        
        ElseIf line$ = "ACTIVATE_VAC" Then
            Print #201, "ACTIVATE_VAC"
            Print #201, "READY"
            GoTo Retry
        
        ElseIf line$ = "DEACTIVATE_VAC" Then
            Print #201, "DEACTIVATE_VAC"
            Print #201, "READY"
            GoTo Retry

        Else
            p = InStr(line$, ";")
            
            If p = 0 Then
                Print #201, "ERROR;NO_DELIM"
                Print #201, "READY"
            ElseIf p <= 1 Then
                GoTo LoopContinue
            Else
                cmd$ = UCase$(Left$(line$, p - 1))
                rest$ = Mid$(line$, p + 1)

                ' -------- SPEED --------
                If cmd$ = "SPEED" Then
                    v = Val(rest$)
                    If v <= 0 Then
                        Print #201, "ERROR;BAD_SPEED"
                    Else
                        Speed v
                    EndIf
                    Print #201, "READY"

                ' -------- ACCEL --------
                ElseIf cmd$ = "ACCEL" Then
                    v = Val(rest$)
                    If v <= 0 Then
                        Print #201, "ERROR;BAD_ACCEL"
                    Else
                        Accel v, v
                    EndIf
                    Print #201, "READY"

                ' ================= GOJ =================
                ElseIf cmd$ = "GOJ" Then
                    token$ = ""
                    idx = 0

                    J1_Coord = 0
                    J2_Coord = 0
                    J3_Coord = 0
                    J4_Coord = 0

                    For i = 1 To Len(rest$)
                        ch$ = Mid$(rest$, i, 1)
                        If ch$ = ";" Then
                            idx = idx + 1
                            v = Val(token$)
                            If idx = 1 Then J1_Coord = v
                            If idx = 2 Then J2_Coord = v
                            If idx = 3 Then J3_Coord = v
                            If idx = 4 Then J4_Coord = v
                            token$ = ""
                        Else
                            If ch$ <> " " Then token$ = token$ + ch$
                        EndIf
                    Next i

                    idx = idx + 1
                    v = Val(token$)
                    If idx = 4 Then J4_Coord = v

                    If idx <> 4 Then
                        Print #201, "ERROR;BAD_ARGS"
                        Print #201, "READY"
                    Else
                        OnErr GoTo MoveErrGOJ
                        Go JA(J1_Coord, J2_Coord, J3_Coord, J4_Coord)
                        OnErr GoTo NetErr
                        Print #201, "READY"
                    EndIf

                ' ================= GOP =================
                ElseIf cmd$ = "GOP" Then
                    token$ = ""
                    idx = 0

                    X_Coord = 0
                    Y_Coord = 0
                    Z_Coord = 0
                    U_Coord = 0

                    For i = 1 To Len(rest$)
                        ch$ = Mid$(rest$, i, 1)
                        If ch$ = ";" Then
                            idx = idx + 1
                            v = Val(token$)
                            If idx = 1 Then X_Coord = v
                            If idx = 2 Then Y_Coord = v
                            If idx = 3 Then Z_Coord = v
                            If idx = 4 Then U_Coord = v
                            token$ = ""
                        Else
                            If ch$ <> " " Then token$ = token$ + ch$
                        EndIf
                    Next i

                    idx = idx + 1
                    v = Val(token$)
                    If idx = 4 Then U_Coord = v

                    If idx <> 4 Then
                        Print #201, "ERROR;BAD_ARGS"
                        Print #201, "READY"
                    Else
						OnErr GoTo MoveErrGOP
                        Go XY(X_Coord, Y_Coord, Z_Coord, U_Coord)
                        Go XY(X_Coord, Y_Coord, -154, U_Coord)
						Print #201, "ACTIVATE_VAC"
	                    Go JA(180, -0, -50, 90)

                        Go JA(180, -90, -154, 90)
						Print #201, "DEACTIVATE_VAC"
						Wait 0.5
						Go JA(180, -90, -50, 90)

                        Go JA(90, 0, 0, 0)
                        
                        OnErr GoTo NetErr
                        Print #201, "READY"
                    EndIf

                Else
                    Print #201, "ERROR;UNKNOWN_CMD"
                    Print #201, "READY"
                EndIf
            EndIf
        EndIf
    Loop

MoveErrGOJ:
    e = Err
    msg$ = ErrMsg$(e)
    OnErr GoTo 0
    Print #201, "ERROR;MOVE;" + Str$(e) + ";" + msg$
    Print #201, "READY"
    OnErr GoTo NetErr

MoveErrGOP:
    e = Err
    msg$ = ErrMsg$(e)
    OnErr GoTo 0
    Print #201, "ERROR;MOVE;" + Str$(e) + ";" + msg$
    Print #201, "READY"
    OnErr GoTo NetErr

NetErr:
    e = Err
    msg$ = ErrMsg$(e)
    CloseNet #201
    GoTo Reconnect

LoopContinue:
	Print #201, "ERROR;UNKNOWN COMMAND"
    Print #201, "READY"
    GoTo Retry
		
ExitLine:
    Go JA(0.000, 0.000, 0.000, 0.000)
    Motor Off
Fend










