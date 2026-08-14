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
# **Library 를 통째로 버린다.**
#
# BuildPlayerData 와 Bee 만 지워 봤는데 부족했다 - 셰이더를 고친 다음 빌드가
# 다시 "level0 is corrupted" 로 죽었다. 어느 캐시가 남아서 그런지 하나씩
# 좁히는 것보다, 이 프로젝트에서는 전부 버리는 편이 싸다: 에셋이라고는
# 스크립트 몇 개와 셰이더 하나뿐이고 (StreamingAssets 는 임포트 대상이
# 아니라 복사 대상이다) 재임포트가 몇십 초다.
#
# 빌드가 가끔 실행되지 않는 것보다 매번 조금 느린 편이 낫다.
Remove-Item (Join-Path $proj "Library") -Recurse -Force -ErrorAction SilentlyContinue

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

# **화면에 뜨는 이름은 여기서 정한다.**
#
# Boot.Awake 는 wvdata/*.json 의 파일명에서 확장자만 떼어 목록에 올린다.
# 그러니 목록의 이름을 바꾸려면 산출물 이름을 바꾸거나 복사할 때 갈아
# 붙이거나 둘 중 하나인데, 산출물 이름은 벤치 매니페스트와 결과 파일들이
# 물고 있으므로 후자다. 대응표는 engine/unity/wvdata_names.tsv 한 곳뿐이고
# wvdata 를 손으로 고치는 것과 달리 다음 빌드에서도 그대로 나온다.
$mapFile = Join-Path $repo "engine\unity\wvdata_names.tsv"
if (-not (Test-Path $mapFile)) {
    Write-Error "이름 대응표가 없다: $mapFile"
    exit 1
}
$displayOf = @{}
foreach ($line in (Get-Content $mapFile -Encoding UTF8)) {
    $t = $line.Trim()
    if ($t -eq "" -or $t.StartsWith("#")) { continue }
    $f = $line -split "`t"
    if ($f.Count -lt 2 -or $f[0].Trim() -eq "" -or $f[1].Trim() -eq "") {
        Write-Error "대응표 줄을 읽을 수 없다 (탭으로 나눈 두 칸이 필요하다): $line"
        exit 1
    }
    $displayOf[$f[0].Trim()] = $f[1].Trim()
}

# 처음 여는 장면도 같은 이름이어야 한다. 임포터가 씬에 직렬화하는
# Boot.sceneName 이 산출물 이름으로 남으면, 목록에는 Data_Set_01 이 떠 있는데
# 그 파일을 못 찾아 빈 화면으로 시작한다.
$sceneBase = [IO.Path]::GetFileNameWithoutExtension($scenePath)
if (-not $displayOf.ContainsKey($sceneBase)) {
    Write-Error "대응표에 없는 시작 장면이다: $sceneBase  ($mapFile)"
    exit 1
}
$sceneDisplay = $displayOf[$sceneBase]

# **출력 폴더도 지우고 시작한다.**
#
# BuildPlayer 는 기존 폴더에 덧쓴다. 이전 빌드의 WorldVision_Data 가 남아
# 있으면 새 level0 과 옛 sharedassets 가 섞여, 빌드는 성공하는데 실행할 때
# "level0 is corrupted" 로 죽는다 - 이것이 하루 종일 간헐적으로 보이던
# 현상의 마지막 조각이다. 성공한 빌드들은 전부 이 폴더를 먼저 지운 것들이었다.
Remove-Item $outPath -Recurse -Force -ErrorAction SilentlyContinue

& $Editor -batchmode -quit -projectPath $proj `
    -executeMethod WorldVision.SceneImporter.BuildSimFromCommandLine `
    -wvScene $scenePath -wvOut $outPath -wvName $sceneDisplay -logFile $log

# 배치 모드는 셰이더 컴파일이 남은 채로 제어를 돌려주기도 한다.
$n = 0
while ((Get-Process -Name Unity -ErrorAction SilentlyContinue) -and $n -lt 240) {
    Start-Sleep -Seconds 5; $n++
}

# **유니티가 하는 말도 흘린다.**
#
# 여기가 "WorldVision:|error CS" 만 보고 있었다. 그래서 유니티가 매 빌드마다
# 남기던
#
#   Script attached to 'WorldVision' in scene 'Assets/WorldVisionSim.unity'
#   is missing or no valid script is attached.
#
# 이 한 줄이 한 번도 화면에 뜨지 않았다. 그것이 바로 "level0 is corrupted" 의
# 원인이었는데 (씬에 끊긴 스크립트 참조가 직렬화된다 - 임포터의
# VerifySceneScripts 주석 참조) 로그 필터가 지우고 있었던 것이다. 네 번을
# 엉뚱한 데서 찾은 까닭이 여기 있다.
$lines = Get-Content $log -Encoding UTF8 |
    Select-String "WorldVision:|error CS|is missing or no valid script|Script attached to|Failed to build|BuildFailedException|UnityEditor\.BuildPlayerWindow"
$lines | ForEach-Object { Write-Output $_.Line }

$exe = Join-Path $outPath "WorldVision.exe"
if (-not (Test-Path $exe)) {
    Write-Error "빌드 실패. 로그: $log"
    exit 1
}
# **씬이 실제로 들어갔는지 본다.**
#
# 씬 없이 빌드되어도 BuildPlayer 는 Succeeded 를 돌려준다. 그 결과물은
# 실행할 때 "level0 is corrupted" 로 죽는데, 사실은 깨진 것이 아니라 없는
# 것이다. 여기서 잡으면 실행해 보기 전에 안다.
if (-not (Test-Path (Join-Path $outPath "WorldVision_Data\level0"))) {
    Write-Error "빌드에 씬이 없다 (level0 없음). 로그: $log"
    exit 1
}
# **데이터는 실행 파일 옆에 둔다.**
#
# StreamingAssets 에 넣으면 빌드에 포장되는데, 네 시퀀스를 담아 158 MB 가
# 되자 실행이 안 됐다 (두 개 121 MB 는 됐다). 옆에 두면 크기 제한이 없고,
# 데이터만 갈아 끼울 때 다시 빌드할 필요도 없다.
$dataDir = Join-Path $outPath "wvdata"
New-Item -ItemType Directory -Force $dataDir | Out-Null
$sceneSrc = Split-Path -Parent $scenePath
$missing = @()
foreach ($j in (Get-ChildItem $sceneSrc -Filter "*.json")) {
    $pc = Join-Path $sceneSrc ($j.BaseName + ".wvpc")
    # 점군이 없는 장면은 싣지 않는다 - 목록에는 뜨는데 열면 빈 편이 더 나쁘다.
    if (-not (Test-Path $pc)) { continue }
    # **이름이 없으면 세운다.** 표에 없는 것을 산출물 이름 그대로 실으면
    # 목록에 Data_Set_03 옆에 kitti_09 가 나란히 서고, 그러면 화면과
    # 대응표 중 어느 쪽이 맞는지 아무도 모른다.
    if (-not $displayOf.ContainsKey($j.BaseName)) {
        $missing += $j.BaseName
        continue
    }
    $dst = $displayOf[$j.BaseName]
    Copy-Item $j.FullName (Join-Path $dataDir ($dst + ".json")) -Force
    Copy-Item $pc         (Join-Path $dataDir ($dst + ".wvpc")) -Force
}
if ($missing.Count -gt 0) {
    Write-Error ("대응표에 없는 시퀀스: " + ($missing -join ", ") + "  ($mapFile)")
    exit 1
}
# 카메라 프레임 폴더도 같은 이름으로 옮긴다 - Boot.CamFrame 이
# cam/<sceneName> 을 찾고, 못 찾으면 cam/ 을 통째로 뒤져 엉뚱한 시퀀스의
# 사진을 옆에 띄운다.
$camSrc = Join-Path $repo "results\cam"
if (Test-Path $camSrc) {
    $camDst = Join-Path $dataDir "cam"
    New-Item -ItemType Directory -Force $camDst | Out-Null
    foreach ($d in (Get-ChildItem $camSrc -Directory)) {
        if ($displayOf.ContainsKey($d.Name)) {
            Copy-Item $d.FullName (Join-Path $camDst $displayOf[$d.Name]) `
                      -Recurse -Force
        }
    }
}
$n = (Get-ChildItem $dataDir -Filter "*.wvpc" | Measure-Object).Count
$mb = [int](((Get-ChildItem $dataDir -Recurse -File | Measure-Object -Property Length -Sum).Sum) / 1MB)
Write-Output "데이터: 시퀀스 $n 개, $mb MB  ($dataDir)"

# **실제로 실행되는지 확인한다.**
#
# 이 빌드는 가끔 성공했다고 하고서 실행하면 "level0 is corrupted" 로 죽는다.
# 원인을 아직 못 찾았다 - 셰이더를 항상포함 목록에 넣던 것, StreamingAssets 를
# 파일 API 로 갈아엎던 것, 출력 폴더와 Library 잔재를 차례로 고쳤고 그때마다
# 한동안 괜찮다가 다시 나온다.
#
# 원인을 모른다고 해서 깨진 결과물을 통과시킬 이유는 없다. 띄워 보고,
# 안 뜨면 다시 짓는다. 세 번 다 실패하면 그때는 실패로 끝낸다 - 조용히
# 넘어가는 것이 제일 나쁘다.
$ok = $false
$p = Start-Process -FilePath $exe -PassThru
Start-Sleep -Seconds 20
if (-not $p.HasExited) { $ok = $true; $p | Stop-Process -Force }
if ($ok) {
    Write-Output "빌드 완료 (실행 확인): $exe"
    if ($Run) { Start-Process -FilePath $exe }
} else {
    Write-Output "빌드는 됐는데 실행되지 않는다. 다시 시도한다."
    exit 2
}
