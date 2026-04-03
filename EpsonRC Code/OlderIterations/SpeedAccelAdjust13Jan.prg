Function SpeedAdjust
    String line$, ch$, token$, msg$
    String cmd$, tag$
    String rest$
    Real X_Coord, Y_Coord, Z_Coord, U_Coord, v, overHead
    Integer p, i, idx, e

    Motor On
    Power High
    Speed 50
    Accel 50, 50

    ' Home at startup
    Go JA(90.000, 0.000, 0.000, 0.000)
    
Reconnect:
    OpenNet #201 As Server
    WaitNet #201
    Print #201, "READY"
	
	TmReset 0
	
    Do
        OnErr GoTo NetErr
        Input #201, line$

        OnErr GoTo 0

        Print line$

        ' -------------------------
        ' Basic commands (no ';')
        ' -------------------------
        If line$ = "EXIT" Then
            Print #201, "BYE"
            GoTo ExitLine

        ElseIf line$ = "HOME" Then
            Go JA(0.000, 0.000, 0.000, 0.000)
            Print #201, "READY"

        ElseIf line$ = "TAKT" Then
        	overHead = Tmr(0)
            Print #201, overHead
            TmReset 0

        Else
            ' -------------------------
            ' Commands with ';'
            '   SPEED;100
            '   ACCEL;100
            '   GOP;X;Y;Z;U
            '   GOP;RED;X;Y;Z;U
            ' -------------------------

            p = InStr(line$, ";")
            If p = 0 Then
                Print #201, "ERROR;NO_DELIM"
            Else
                cmd$ = UCase$(Left$(line$, p - 1))
                rest$ = Mid$(line$, p + 1)

                ' --- SPEED ---
                If cmd$ = "SPEED" Then
                    v = Val(rest$)
                    If v <= 0 Then
                        Print #201, "ERROR;BAD_SPEED"
                    Else
                        Speed v
                        Print #201, "READY"
                    EndIf

                ' --- ACCEL ---
                ElseIf cmd$ = "ACCEL" Then
                    v = Val(rest$)
                    If v <= 0 Then
                        Print #201, "ERROR;BAD_ACCEL"
                    Else
                        Accel v, v
                        Print #201, "READY"
                    EndIf

                ' --- GOP ---
                ElseIf cmd$ = "GOP" Then
                    token$ = ""
                    idx = 0
                    tag$ = ""

                    X_Coord = 0
                    Y_Coord = 0
                    Z_Coord = 0
                    U_Coord = 0

                    For i = 1 To Len(rest$)
                        ch$ = Mid$(rest$, i, 1)

                        If ch$ = ";" Then
                            ' We completed one token
                            If Len(token$) > 0 Then
                                ' If first token is non-numeric -> treat as tag
                                If idx = 0 And Val(token$) = 0 And token$ <> "0" Then
                                    tag$ = UCase$(token$)
                                Else
                                    idx = idx + 1
                                    v = Val(token$)
                                    If idx = 1 Then X_Coord = v
                                    If idx = 2 Then Y_Coord = v
                                    If idx = 3 Then Z_Coord = v
                                    If idx = 4 Then U_Coord = v
                                EndIf
                            EndIf
                            token$ = ""
                        Else
                            If ch$ <> " " Then token$ = token$ + ch$
                        EndIf
                    Next i

                    ' Last token
                    If Len(token$) > 0 Then
                        ' Last token must be numeric in our formats
                        idx = idx + 1
                        v = Val(token$)
                        If idx = 1 Then X_Coord = v
                        If idx = 2 Then Y_Coord = v
                        If idx = 3 Then Z_Coord = v
                        If idx = 4 Then U_Coord = v
                    EndIf

                    If idx <> 4 Then
                        Print #201, "ERROR;BAD_ARGS"
                    Else
						' --- Try the move, but catch motion errors locally ---
						OnErr GoTo MoveErr
						
						If tag$ = "RED" Then
						    Go XY(X_Coord, Y_Coord, Z_Coord, U_Coord)
						Else
						    Go XY(X_Coord, Y_Coord, Z_Coord, U_Coord)
						EndIf
						
						OnErr GoTo NetErr     ' restore main handler (network + everything else)
						Print #201, "READY"
						GoTo AfterMove
						
						MoveErr:
						    e = Err
						    msg$ = ErrMsg$(e)
						
						    ' Clear error state and continue listening
						    OnErr GoTo 0
						
						    ' Tell the PC what happened (so your client doesn't hang waiting for READY)
						    Print #201, "ERROR;MOVE;" + Str$(e) + ";" + msg$
						
						    ' Optional: stop/jog recovery could be added here if needed
						    ' For safety you might do: Stop
						
						    OnErr GoTo NetErr     ' restore main handler
						    GoTo AfterMove
						
						AfterMove:

                    EndIf

                Else
                    Print #201, "ERROR;UNKNOWN_CMD"
                EndIf
            EndIf
        EndIf
    Loop

NetErr:
    e = Err
    msg$ = ErrMsg$(e)

    If e = 2902 Then
        CloseNet #201
        GoTo Reconnect
    Else
        Print #201, "ERROR;" + Str$(e) + ";" + msg$
        CloseNet #201
        GoTo Reconnect
    EndIf

ExitLine:
    Go JA(0.000, 0.000, 0.000, 0.000)
    Motor Off
Fend


