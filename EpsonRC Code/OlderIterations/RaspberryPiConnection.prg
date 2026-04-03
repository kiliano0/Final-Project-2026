Function RaspberryPi
    String line$, ch$, token$, msg$
    Real Joint1, Joint2, Joint3, Joint4, v
    Integer p, i, idx, e

    Motor On
    Speed 100

    ' Home at startup
    Go JA(0.000, 0.000, 0.000, 0.000)

	Reconnect:
    OpenNet #201 As Server
    WaitNet #201
    Print #201, "READY"

    Do
        OnErr GoTo NetErr
        Input #201, line$
        Print line$

        If line$ = "EXIT" Then
        	  Print #201, "BYE"
            Exit Do

        ElseIf line$ = "HOME" Then
            Go JA(0.000, 0.000, 0.000, 0.000)
            Print #201, "READY"

        Else
            ' Expect: GOJ;J1;J2;J3;J4
            p = InStr(line$, ";")
            If p = 0 Then
                Print #201, "ERROR;NO_DELIM"
            Else
                ' Reset parsing state
                token$ = ""
                idx = 0
                Joint1 = 0
                Joint2 = 0
                Joint3 = 0
                Joint4 = 0

                ' Parse values after first ';'
                For i = p + 1 To Len(line$)
                    ch$ = Mid$(line$, i, 1)

                    If ch$ = ";" Then
                        idx = idx + 1
                        v = Val(token$)
                        If idx = 1 Then Joint1 = v
                        If idx = 2 Then Joint2 = v
                        If idx = 3 Then Joint3 = v
                        If idx = 4 Then Joint4 = v
                        token$ = ""
                    Else
                        If ch$ <> " " Then token$ = token$ + ch$
                    EndIf
                Next i

                ' Last token
                If Len(token$) > 0 Then
                    idx = idx + 1
                    v = Val(token$)
                    If idx = 1 Then Joint1 = v
                    If idx = 2 Then Joint2 = v
                    If idx = 3 Then Joint3 = v
                    If idx = 4 Then Joint4 = v
                EndIf

                If idx <> 4 Then
                    Print #201, "ERROR;BAD_ARGS"
                Else
                    Go JA(Joint1, Joint2, Joint3, Joint4)
                    Print #201, "READY"
                EndIf
            EndIf
        EndIf
    Loop

	NetErr:
    ' Handle Ethernet read failures / disconnects gracefully
    e = Err
    msg$ = ErrMsg$(e)

    ' If the client disconnected, just reset the connection
    ' 2902 = Failed to read from Ethernet port (common on disconnect)
    If e = 2902 Then
        CloseNet #201
        GoTo Reconnect
    Else
        ' For other errors, report and reconnect
        Print #201, "ERROR;" + Str$(e) + ";" + msg$
        CloseNet #201
        GoTo Reconnect
    EndIf
    
    Go JA(0.000, 0.000, 0.000, 0.000)
    Motor Off
Fend

