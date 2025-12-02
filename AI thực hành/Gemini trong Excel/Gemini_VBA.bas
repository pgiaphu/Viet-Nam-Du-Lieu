Attribute VB_Name = "Gemini_VBA"
' UTF-8 Bytes -> Unicode | Sua loi dinh dang tieng Viet trong VBA
' ============================================================
Private Function BytesToString(bytes() As Byte) As String
    With CreateObject("ADODB.Stream")
        .Type = 1: .Open: .Write bytes: .Position = 0: .Type = 2: .Charset = "UTF-8"
        BytesToString = .ReadText: .Close
    End With
End Function


' Range -> JSON
' ============================================================
Function RangeToJson(Optional rng As Range) As String
    On Error GoTo ErrHandler
    If rng Is Nothing Then Set rng = Selection
    If rng.Rows.Count < 2 Then
        RangeToJson = "[]"
        Exit Function
    End If
    
    Dim headers() As Variant: headers = rng.Rows(1).Value
    Dim data As New Collection
    Dim rowIndex As Long, colIndex As Long
    
    For rowIndex = 2 To rng.Rows.Count
        Dim dict As Object: Set dict = CreateObject("Scripting.Dictionary")
        For colIndex = 1 To UBound(headers, 2)
            Dim key As String: key = Trim(headers(1, colIndex))
            If key = "" Then key = "Column" & colIndex
            Dim cellValue As Variant: cellValue = rng.Cells(rowIndex, colIndex).Value
            If IsDate(cellValue) Then
                dict(key) = Format(cellValue, "yyyy-mm-ddThh:nn:ssZ")
            ElseIf IsError(cellValue) Then
                dict(key) = Null
            ElseIf VarType(cellValue) = vbBoolean Then
                dict(key) = CBool(cellValue)
            Else
                dict(key) = cellValue
            End If
        Next colIndex
        data.Add dict
    Next rowIndex
    
    RangeToJson = JsonConverter.ConvertToJson(data, Whitespace:=0)
    Exit Function
    
ErrHandler:
    RangeToJson = "{""error"": """ & Err.Description & """}"
End Function


' Ham goi AI
' ============================================================
Public Function Gemini(dataRange As Range, _
                        prompt As String, _
                        Optional apiKey As String = "thay-the-api-vao-day", _
                        Optional model As String = "gemini-2.0-flash") As String
    On Error GoTo ErrHandler
    

    Dim fullPrompt As String
    fullPrompt = prompt & vbCrLf & vbCrLf & RangeToJson(dataRange)
    
    Dim http As Object
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    
    Dim safePrompt As String
    safePrompt = Replace(fullPrompt, "\", "\\")
    safePrompt = Replace(safePrompt, """", "\""")
    safePrompt = Replace(safePrompt, vbCrLf, "\n")
    safePrompt = Replace(safePrompt, vbLf, "\n")
    safePrompt = Replace(safePrompt, vbCr, "\n")
    
    
    Dim url As String
    url = "https://generativelanguage.googleapis.com/v1/models/" & model & ":generateContent?key=" & apiKey
    
    Dim body As String
    body = "{""contents"":[{""role"":""user"",""parts"":[{""text"":""" & safePrompt & """}]}]}"
    
    http.Open "POST", url, False
    http.setRequestHeader "Content-Type", "application/json"
    http.send body
    
    Dim raw As String
    raw = http.responseText
    
    Dim json As Object
    Set json = JsonConverter.ParseJson(raw)
    
    Dim resultText As String
    resultText = json("candidates")(1)("content")("parts")(1)("text")
    Gemini = resultText
    Exit Function
    
ErrHandler:
    Gemini = "Error: " & Err.Description & vbCrLf & raw
End Function
