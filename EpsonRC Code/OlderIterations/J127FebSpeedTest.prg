Global Real gSpeedVar
Global Real gAccelVar

Function NewSpeedTestJ1
    Motor On
    Power High
    Real overHead, SpeedVar, i
    Speed 100

    TmReset 0
    Accel 50, 50
    Go JA(0.000, 0.000, 0.000, 0.000)
    Wait 1
    'Xqt showPosition
    'Wait 10

    For i = 1 To 10
        SpeedVar = -((i - 11) * 10)
        Speed SpeedVar

        Print "Speed:", SpeedVar

        Go JA(0.000, 0.000, 0.000, 0.000)
        Go JA(180.000, 0.000, 0.000, 0.0000)
        overHead = Tmr(0)
        Go JA(0.000, 0.000, 0.000, 0.000)
        Wait 1
        Print overHead
    Next
Fend

