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
& $csc /nologo /target:library /warn:4 /out:$out $stubs $importer
if ($LASTEXITCODE -ne 0) {
    Write-Error "임포터가 컴파일되지 않는다."
}
Write-Output "임포터 컴파일 통과: $importer"
