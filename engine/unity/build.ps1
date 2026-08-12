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
Copy-Item (Join-Path $repo "engine\unity\Runtime\*.cs") `
          (Join-Path $proj "Assets\Scripts\") -Force
Copy-Item (Join-Path $repo "engine\unity\Runtime\*.shader") `
          (Join-Path $proj "Assets\Scripts\") -Force

# **생성물은 매번 지우고 시작한다.**
#
# 임포터가 만드는 메시와 재질은 에셋이고, 그것을 지웠다 같은 경로에 다시
# 만들면 이전 씬이 들고 있던 참조가 어긋난다. 빌드는 성공하는데 실행하면
# "level0 is corrupted / Position out of bounds" 로 죽었다 - 그것도 매번이
# 아니라 가끔이라 더 나빴다.
Remove-Item (Join-Path $proj "Assets\WorldVision") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $proj "Assets\WorldVision.meta") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $proj "Assets\WorldVisionScene.unity") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $proj "Assets\WorldVisionScene.unity.meta") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $proj "Assets\WorldVisionSim.unity") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $proj "Assets\WorldVisionSim.unity.meta") -Force -ErrorAction SilentlyContinue
# Library 의 빌드 캐시도 함께 버린다. 남겨 두면 지운 에셋을 가리키는 항목이
# 살아남아, 빌드는 성공하는데 실행할 때 level0 을 못 읽는 일이 생긴다.
Remove-Item (Join-Path $proj "Library\BuildPlayerData") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $proj "Library\Bee") -Recurse -Force -ErrorAction SilentlyContinue

# 프로젝트를 열고 있는 Unity 가 남아 있으면 잠금 때문에 배치 모드가 아예
# 시작하지 않는다 - 로그 파일조차 안 생겨서 원인이 잘 안 보인다.
Get-Process -Name Unity -ErrorAction SilentlyContinue | Stop-Process -Force
# **돌고 있는 플레이어도 먼저 죽인다.** 그것이 WorldVision_Data 를 잡고
# 있으면 출력 폴더 삭제가 반쪽만 되고, 그다음 빌드는 성공했다고 하면서
# 데이터 폴더가 없는 실행 파일을 남긴다 - 실행하면 "Data folder not found".
Get-Process -Name WorldVision -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

$log = Join-Path $env:TEMP "wv_unity_build.log"
$scenePath = (Resolve-Path (Join-Path $repo $Scene)).Path
$outPath = Join-Path $repo $Out

& $Editor -batchmode -quit -projectPath $proj `
    -executeMethod WorldVision.SceneImporter.BuildSimFromCommandLine `
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
