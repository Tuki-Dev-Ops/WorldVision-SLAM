<div align="center">

# WorldVision-SLAM

### WME — World Model Engine

**記述子（descriptor）を使わない SLAM。** 特徴点ではなく、世界を覚える。

<br>

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C?logo=cmake&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![Eigen](https://img.shields.io/badge/Eigen-3.4-1F425F)
![pybind11](https://img.shields.io/badge/pybind11-2.11-FFD43B)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.22-005CED?logo=onnx&logoColor=white)
![GoogleTest](https://img.shields.io/badge/GoogleTest-236%20passing-brightgreen)
![pytest](https://img.shields.io/badge/pytest-639%20passing-brightgreen)

<br>

[한국어](../../README.md) · [English](README.en.md) · [中文](README.zh-CN.md) · **日本語**

</div>

---

## プロジェクトの背景

古典的な視覚 SLAM は 30 年近く同じ前提の上に立っている。**画素の近傍を数値ベクトル（記述子）
に要約し、ベクトルが似ていれば同じ点とみなす。** ORB も SIFT も BRIEF もこの前提を共有する。

この前提は条件が良ければ非常によく働き、条件が悪くなると**静かに**崩れる。霧が出ても記述子は
「マッチ失敗」とは言わない。もっともらしい誤答を返す。夜間も、雨天も、カメラが揺れたときも
同じことが起きる。失敗が騒がしくないため、上位の層はそれに気づく手段を持たない。

WME は別の問いから出発する。**人は記述子をマッチングしない。** 暗い部屋に入っても机の位置は
分かる。画素を照合しているからではなく、その部屋の**モデル**を持っているからだ。だから対応
（correspondence）問題を画素の水準では解かず、世界モデルの水準で解く。

したがってこのリポジトリが作るのは「より良い記述子」ではなく、**記述子を使わないパイプライン**
であり、それが本当に記述子パイプラインより良いのかを同じデータの上で並べて測る装置である。

### このプロジェクトが自らに課した条件

主張は簡単で検証は難しい。だからルールを先に決めた。

| ルール | 理由 |
|---|---|
| **すべての C++ 実装に独立した numpy オラクルを付ける** | 二つの実装が同じ答えを出して初めてその答えを信じる。自分で自分を検証するコードは自分のバグを隠す |
| **推定と採点は別のコードが行う** | 推定は C++、ATE/RPE の採点は Python。同じコードで両方やると二つのバグが打ち消し合う |
| **対照群は「使わない」と宣言したまさにその記述子パイプライン** | 適当に弱い相手に勝っても意味がない |
| **その測定に判別力があるかを先に確かめる** | すべての入力で同じ値を返す指標は、通っても何も証明しない |
| **失敗は騒がしくなければならない** | 「保存した」と表示してファイルを書かないツール、到達不能なのに緑で終わるテスト —— どちらも欠陥として扱う |

これらのルールが実際に何を捕まえたかは [docs/06-results.md](../06-results.md) にすべて記録して
ある。成功した実験だけでなく、**却下された仮説五つと、私自身が作り込んだ欠陥**まで残っている。

---

## プロジェクトの目的

### 1. 三つの階層で対応問題を解く

```
Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA
```

| 階層 | 名称 | 役割 |
|---|---|---|
| **Tier 0** | ECDA | 直接測光アライメント。画素輝度の残差を最小化する。記述子なし |
| **Tier 1** | TCG | トークン星座幾何。物体（YOLO 検出）の相対配置から位置を求める |
| **Tier 2** | SPA | 構造アライメント。平面の法線と距離で退化した自由度を埋める |

`α_k(E)` は手書きの分岐ではない。**環境証拠 E**（暗さ、霧、モーションブラー、テクスチャの
乏しさ…）から計算される。夜間用のコードも雨天用のコードも存在しない —— 劣化は各情報源が
寄与する**情報量の減少**としてのみ表現される。

### 2. その主張を同じデータの上で並べて測る

`results/bench/index.html` は**左に従来手法、右に WME** を置き、軌跡・ATE・RPE・速度を一画面で
比較する。さらにネイティブアプリ `wme_bench_viewer.exe` があり、各システム**自身が推定した
姿勢**で深度マップを 3D 再構成する —— ドリフトは数値ではなく、にじんだ点群として現れる。

<div align="center">

| データセット | シーケンス数 | 対照群 | 結果（単一構成） |
|---|---|---|---|
| **TUM RGB-D**（屋内・手持ち） | 16 | ORB+PnP、`cv2.Odometry` | **8 – 7**（二つのうち良い方に対して） |
| **KITTI odometry**（屋外・車載） | 4 | ORB+PnP | **4 – 0** |

</div>

**TUM は実質的に引き分けである。** シーケンスごとに良い方の WME 変種を選べば 10–5 になるが、
それは実際に配備されるシステムには許されない選択だ。上の表は全シーケンスに**同一の構成**
（Tier 0）を走らせた数字である。

これらの数字は二度大きく動き、二度とも原因はアルゴリズムではなく測定の側にあった。

- **KITTI を加えると 2–2 に逆転した。** ECDA がステレオ深度を真値として信じていた —— 60 m
  地点の深度誤差は ±4 m なのに、6 m 地点（±4 cm）と同じ重みで方程式に入っていた。その不確かさを
  残差分散へ移すと（係数は `c = σ_d/(f·B)` から**導出**したもので、調整値ではない）4–0 になった。
  （[§25.20–25.21](../06-results.md)）
- **TUM 16 本のうち 13 本が、シーケンスの一部だけで採点されていた** —— 平均 35 %、最低 6.4 %。
  展開が中断してもフレーム索引だけは無傷で残り、各ローダは存在するファイルだけを読んで
  きれいな実行を報告していた。取得し直して再実行すると、五つの判定が**双方向に**逆転した。
  （[§25.22](../06-results.md)）

> **それまで「WME の方が良い」と書かれていた文は、一度はアルゴリズムではなく TUM の性質であり、
> 一度はデータの三分の一だけを見た結果だった。**

### 3. 何ではないかも明確にする

- **これは ORB-SLAM3 ではない。** 対照群は ORB 検出 → ハミングマッチング → RANSAC PnP まで。
  ループ閉じ込みもバンドル調整もない。比較は**オドメトリ対オドメトリとしてのみ**有効
- **どちらの側にもバンドル調整はない。** ポーズグラフまで
- 劣化（霧）実験は実際の TUM フレームに**実測深度**を用いて散乱方程式を適用したもので、
  自然に劣化したデータではない

**確立されていないこと**は [docs/06-results.md §26](../06-results.md) に一覧がある。

---

## リポジトリ構成

```
WorldVision-SLAM/
├── include/wme/              公開ヘッダ（27）
│   ├── core/                 SE3, Frame, Result, ThreadPool, Assignment
│   ├── localization/         DirectAligner            <- Tier 0 (ECDA)
│   ├── token/                TokenStore, ConstellationIndex, WorldToken
│   │                                                  <- Tier 1 (TCG)
│   ├── geometry/             StructuralAligner, PlaneExtractor
│   │                                                  <- Tier 2 (SPA)
│   ├── fusion/               PoseFusion, TierInformation
│   ├── perception/           ImageQualityEngine, EnvironmentAnalyzer,
│   │                         StereoDepth, YoloRuntime{Cv,Ort}
│   └── confidence/           ConfidenceEngine
│
├── src/                      実装（約 17 kLOC）
│
├── tools/                    実行可能な実験バイナリ
│   ├── tum_odometry.cpp      WME オドメトリ
│   ├── tum_baseline.cpp      ORB+PnP 対照群 <- 使わないと宣言したまさにそれ
│   ├── kitti_convert.cpp     KITTI -> TUM 配置 + StereoSGBM 深度
│   ├── bench_viewer.cpp      左右並置のベンチマークアプリ
│   ├── tum_loopclose.cpp     対称なループ閉じ込み（ORB vs TCG）
│   ├── tum_degrade.cpp       実測深度に基づく散乱劣化
│   └── ...                   fusion, relocalize, tcg_density, plane_density
│
├── tests/                    C++ テスト（236 件）
│
├── python/
│   ├── wme/
│   │   ├── reference/        * C++ と突き合わせる numpy オラクル
│   │   ├── localization/     ecda.py —— DirectAligner のオラクル
│   │   ├── geometry/         spa.py, planes.py
│   │   ├── eval/             ATE / RPE / Umeyama、TUM 読み込み  <- 採点専用
│   │   ├── graph/            ポーズグラフ、ファクタ、測光 SLAM
│   │   ├── sim/ world/       合成シーン、世界モデルの状態と予測
│   │   └── association/ calib/ planner/
│   ├── bindings/             pybind11 -> wme._core
│   ├── tools/                実験・ベンチスクリプト（32）
│   └── tests/                Python テスト（639 件）
│
├── docs/06-results.md        * すべての実測結果と失敗の記録
├── results/bench/index.html  * 左：従来手法 / 右：WME
└── .github/workflows/        linux, windows-msvc, sanitizers, python
```

---

## データセット

どちらもリポジトリには**含まれていない**（合計 51 GB）。スクリプトで再現する。

### TUM RGB-D —— 屋内・手持ち・実測深度

Kinect の構造化光センサなので深度は**測定値**である。16 シーケンスを使う。

```bash
python python/tools/tum_fetch_all.py     # 16 本すべて、完全版
python python/tools/check_datasets.py    # インデックスとディスクの一致を検査
```

> `tum_fetch.py` の既定は 9 秒の窓である。`--all` で全体を取得する。なおこのツールは
> `rgb.txt`/`depth.txt` を実在するファイルに合わせて切り詰めるようになった。完全な
> インデックスを残したまま画像が一部しかない状態では、すべてのツールが欠落を静かに
> 読み飛ばす —— 「1419 フレーム」と表示された実行が、実際には 165 フレームだった。

> 内部パラメータと歪み係数は freiburg グループごとに異なる。一つに固定すると他のグループで
> **失敗ではなく、もっともらしい誤差**が出る。

### KITTI odometry —— 屋外・車載・ステレオ

深度がない。**こちらで作る必要がある** —— `StereoDepth`（OpenCV SGBM）フロントエンドが初めて
必要になるのはここである。ここでの深度は測定値ではなく**推定値**であり、その区別が §25.21 の
核心である。

```bash
python python/tools/fetch_kitti.py       # 21.6 GB、レジューム対応
build/tools/wme_kitti_convert data/kitti/dataset 00 data/kitti_00 --stride 2
```

視差の探索範囲はシーンの最近距離から**導出**する。既定値のままにすると SGBM は「範囲外」とは
言わず、**もっともらしい誤答**を返す（実測：深度スケールが 2.42 倍）。

| 項目 | 値 |
|---|---|
| ダウンロード済みシーケンス | 00–21（正解軌跡は 00–10） |
| 変換・評価済み | 00, 04, 05, 07 |
| 解像度 / 焦点距離 / 基線長 | 1241×376 / 718.86 px / 0.537 m |
| 有効深度の割合（SGBM） | 67 – 74 % |

---

## 実行方法

### 1. 前提

| 項目 | バージョン |
|---|---|
| コンパイラ | MSVC 2022 / GCC 11+ / Clang 14+（C++20） |
| CMake | 3.24 以上 |
| OpenCV | 4.8 以上 —— `core imgproc imgcodecs videoio calib3d highgui dnn features2d` |
| Python | 3.10 以上 + `numpy scipy pytest pybind11` |
| （任意）ONNX Runtime | 1.22 —— YOLO トークンマスク用 |

> OpenCV のコンポーネントから `features2d` を落とすとリンク時にだけ失敗する。
> `cmake/WmeDependencies.cmake` のコメントにこの罠が書いてある。

### 2. ビルド

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3. テスト —— まずここから

```bash
ctest --test-dir build --output-on-failure     # C++    236 件
cd python && python -m pytest -q               # Python 639 件
```

Python 側のかなりの部分は **C++ ↔ numpy の差分テスト**である。ここが緑だということは、独立した
二つの実装が同じ答えを出したという意味であり、このリポジトリの数値を信じてよい根拠はそれだけ
である。

### 4. ベンチマーク → 左右比較アプリ

```bash
python python/tools/bench_run.py               # 両システムを実行して採点
python python/tools/bench_report.py            # -> results/bench/index.html
python python/tools/bench_export.py            # -> results/bench/viewer.tsv
build/tools/wme_bench_viewer                   # ネイティブアプリ
```

一部だけ再実行：

```bash
python python/tools/bench_run.py --only kitti --merge   # KITTI のみ、他は保持
python python/tools/bench_run.py --skip-run             # 再推定せず再採点のみ
```

アプリの操作：`SPACE` 再生 · `,` `.` 1 フレーム · `N`/`P` シーケンス · `1`/`2` 各パネルの
モデル切替 · `A`/`D` 視点回転 · `W`/`S` ズーム · `F` スクリーンショット · `Q` 終了。

このアプリは **ATE/RPE を計算しない。** `bench_run.py` が計算した値を表示する —— 一つの指標に
実装が二つあることこそ、画面の数値と文書の数値が食い違い始める入口だからである。

### 5. 単一シーケンス

```bash
# TUM
build/tools/wme_tum_odometry data/rgbd_dataset_freiburg1_xyz out.txt
build/tools/wme_tum_baseline data/rgbd_dataset_freiburg1_xyz orb.txt
python python/tools/tum_eval.py data/rgbd_dataset_freiburg1_xyz out.txt

# KITTI —— 深度上限と不確かさ係数はデータセット側から来る
build/tools/wme_tum_odometry data/kitti_00 out.txt \
    --kf-dist 1.0 --depth-max 60 --depth-sigma-rel 7.8e-4
```

### 6. その他の実験

```bash
python python/tools/baseline_cv2.py       # 第三者の対照群（cv2.Odometry）
python python/tools/bench_degrade.py      # 霧のスイープ
python python/tools/stereo_validate.py    # ステレオ深度を TUM の実測深度で検証
python python/tools/loop_optimize.py      # ポーズグラフによるループ閉じ込み
```

---

<div align="center">

**結果の全文・失敗の記録・却下された仮説 → [docs/06-results.md](../06-results.md)**

</div>
