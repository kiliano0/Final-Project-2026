Function JointsTestTCP
	String line$, ch$, token$, cmd$
	Real Joint1, Joint2, Joint3, Joint4, v
    Integer p, i, idx
    'Joint1 = 41.640
    'Joint2 = 45.120
    'Joint3 = -81.820
    'Joint4 = 0
    
	Motor On
	Speed 100
	Go JA(0.000, 0.000, 0.000, 0.000)
	OpenNet #201 As Server
	WaitNet #201
	Print #201, "Ready"
	
	Do While line$ <> "Stop"
		cmd$ = ""
    	token$ = ""
    	idx = 0
    	Joint1 = 0
    	Joint2 = 0
    	Joint3 = 0
    	Joint4 = 0
    	v = 0
		
		Input #201, line$
		Print line$
		
		If line$ = "EXIT" Then
			Exit Do
		EndIf
			
		p = InStr(line$, ";")
		Print p
		
		idx = 0
	
		For i = p + 1 To Len(line$)
	        ch$ = Mid$(line$, i, 1)
		    If ch$ = ";" Then
		        If Len(token$) > 0 Then
		            idx = idx + 1
		            v = Val(token$)
		            If idx = 1 Then
		                Joint1 = v
		            ElseIf idx = 2 Then
		                Joint2 = v
		            ElseIf idx = 3 Then
		                Joint3 = v
		            ElseIf idx = 4 Then
		                Joint4 = v
		            EndIf
		            token$ = ""
		        EndIf
		    Else
		        If Mid$(line$, i, 1) <> " " Then
		            token$ = token$ + Mid$(line$, i, 1)  ' or use + if & isn’t supported
		        EndIf
		    EndIf
		Next i
	    
		If Len(token$) > 0 Then
		    idx = idx + 1
		    v = Val(token$)
		    If idx = 1 Then Joint1 = v
		    If idx = 2 Then Joint2 = v
		    If idx = 3 Then Joint3 = v
		    If idx = 4 Then Joint4 = v
		EndIf
		
		Print Joint1
		Print Joint2
		Print Joint3
		Print Joint4
		Go JA(Joint1, Joint2, Joint3, Joint4)
		Print #201, "READY"
	Loop
	
	Go JA(0.000, 0.000, 0.000, 0.000)
	Motor Off
Fend


