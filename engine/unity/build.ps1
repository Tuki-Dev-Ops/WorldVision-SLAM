# 인식한 장면으로 걸어 다닐 수 있는 실행 파일을 만든다.
#
#   powershell -ExecutionPolicy Bypass -File engine/unity/build.ps1
#   powershell ... -File engine/unity/build.ps1 -Scene results/scene/kitti_05.json
#
# 배치 모드라 에디터를 열지 않는다. 다만 라이선스는 필요하다 - 활성화되지
# 않은 상태에서는 -batchmode 도 시작하지 않는다.
param(
    [string]$Editor = "D:\UnityEditors\6000.0.81f1\Editor\Unity.exe",
    [string]$Scene  = "results/scene/kitti_00.json",
    [string]$Out    = "build/unity",
    [switch]$Run
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$proj = Join-Path $repo "engine\unity\Project"

# 소스는 engine/unity 아래에 두고 프로젝트에는 복사해 넣는다. 프로젝트
# 폴더는 Unity 가 자기 마음대로 건드리므로 원본을 거기 두지 않는다.
New-Item -ItemType Directory -Force (Join-Path $proj "Assets\Editor")  | Out-Null
New-Item -ItemType Directory -Force (Join-Path $proj "Assets\Scripts") | Out-Null
Copy-Item (Join-Path $repo "engine\unity\Editor\WorldVisionSceneImporter.cs") `
          (Join-Path $proj "Assets\Editor\") -Force
Copy-Item (Join-Path $repo "engine\unity\Runtime\WorldVisionPlayer.cs") `
          (Join-Path $proj "Assets\Scripts\") -Force

# 프로젝트를 열고 있는 Unity 가 남아 있으면 잠금 때문에 배치 모드가 아예
# 시작하지 않는다 - 로그 파일조차 안 생겨서 원인이 잘 안 보인다.
Get-Process -Name Unity -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

$log = Join-Path $env:TEMP "wv_unity_build.log"
$scenePath = (Resolve-Path (Join-Path $repo $Scene)).Path
$outPath = Join-Path $repo $Out

& $Editor -batchmode -quit -projectPath $proj `
    -executeMethod WorldVision.SceneImporter.BuildPlayerFromCommandLine `
    -wvScene $scenePath -wvOut $outPath -logFile $log

# 배치 모드는 셰이더 컴파일이 남은 채로 제어를 돌려주기도 한다.
$n = 0
while ((Get-Process -Name Unity -ErrorAction SilentlyContinue) -and $n -lt 240) {
    Start-Sleep -Seconds 5; $n++
}

$lines = Get-Content $log -Encoding UTF8 | Select-String "WorldVision:|error CS"
$lines | ForEach-Object { Write-Output $_.Line }

$exe = Join-Path $outPath "WorldVision.exe"
if (-not (Test-Path $exe)) {
    Write-Error "빌드 실패. 로그: $log"
    exit 1
}
Write-Output "빌드 완료: $exe"
if ($Run) { Start-Process -FilePath $exe }
