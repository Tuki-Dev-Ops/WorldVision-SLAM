# 임포터가 **컴파일되는지** 확인한다. Unity 없이.
#
# 에디터를 열었을 때 가장 흔히 깨지는 것은 컴파일 에러다. Unity 를 설치하지
# 않고도 그건 없앨 수 있다 - UnityStubs.cs 가 임포터가 쓰는 API 만 흉내 내고,
# 시그니처가 어긋나면 컴파일러가 잡는다.
#
# 동작을 검증하지는 않는다. 스텁은 아무 것도 하지 않는다. 좌표 변환이 맞는지는
# engine/README.md 의 "확인한 것과 확인하지 못한 것" 을 참조.
#
# 사용:  powershell -ExecutionPolicy Bypass -File engine/unity/Check/check.ps1

$ErrorActionPreference = "Stop"
# 콘솔 코드페이지가 949 면 한글 출력이 깨진다. 파일은 UTF-8 이므로 출력만 맞춘다.
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$importer = Join-Path $here "..\Editor\WorldVisionSceneImporter.cs"
$stubs = Join-Path $here "UnityStubs.cs"

# **Runtime 을 통째로 넣는다.**
#
# 여기가 여섯 파일 중 넷만 골라 넣고 있었다. Boot 와 Log 가 빠져 있었는데
# Sim 이 그 둘을 쓰므로, 이 검사는 통과하기는커녕 **오늘까지 계속 실패하고
# 있었다** - 'Boot 형식을 찾을 수 없습니다'. 검사가 늘 빨간불이면 아무도
# 그 빨간불을 읽지 않는다.
#
# 골라 넣을 이유가 애초에 없다. 파일이 늘면 목록도 같이 늘어야 하는데 그
# 손질을 잊는 것이 정확히 이 사고다. 폴더를 통째로 넘긴다 - Route 와 Stats
# 처럼 클래스마다 파일을 가르는 규칙이 있으므로 (WorldVisionRoute.cs 의
# 주석 참조) 파일은 앞으로도 늘어난다.
$runtime = Get-ChildItem (Join-Path $here "..\Runtime") -Filter "*.cs" |
           ForEach-Object { $_.FullName }

# Roslyn 은 Visual Studio 나 Build Tools 와 함께 온다. 없으면 .NET Framework
# 쪽 csc 를 쓴다 - 이 코드는 언어 기능을 특별히 쓰지 않아 둘 다 통과한다.
$candidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\Roslyn\csc.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe",
    "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
)
$csc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $csc) {
    Write-Error "csc.exe 를 못 찾았다. Visual Studio 또는 Build Tools 가 필요하다."
}

$out = Join-Path $env:TEMP "wv_unity_check.dll"
$src = @($stubs, $importer) + $runtime
& $csc /nologo /target:library /warn:4 /out:$out $src
if ($LASTEXITCODE -ne 0) {
    Write-Error "임포터가 컴파일되지 않는다."
}
Write-Output "컴파일 통과: 임포터 + Runtime $($runtime.Count) 파일"
