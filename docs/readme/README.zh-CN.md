<div align="center">

# WorldVision-SLAM

### WME — World Model Engine

**无描述子 SLAM。** 它记住的是世界，而不是特征点。

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

[한국어](../../README.md) · [English](README.en.md) · **中文** · [日本語](README.ja.md)

</div>

---

## 项目背景

经典视觉 SLAM 近三十年来一直建立在同一个前提上：**把像素邻域概括成一个数值向量（描述子），
向量相似就认为是同一个点。** ORB、SIFT、BRIEF 都共享这个前提。

这个前提在条件好的时候非常有效，条件变差时则会**悄无声息地**崩溃。起雾时描述子不会报告
"匹配失败"，它会给出一个看似合理的错误答案。夜间、雨天、相机抖动时都是如此。因为失败并不
喧闹，上层系统根本无从察觉。

WME 从另一个问题出发。**人不匹配描述子。** 走进一间暗房，你依然知道桌子在哪里 —— 不是靠
比对像素，而是因为你拥有那个房间的**模型**。所以对应问题不在像素层面解决，而在世界模型
层面解决。

因此本仓库要做的不是"更好的描述子"，而是一条**不使用描述子的流水线**，以及在同一份数据上
并排检验它是否真的优于描述子流水线的装置。

### 本项目给自己定下的条件

主张容易，验证不易，所以规则先行。

| 规则 | 理由 |
|---|---|
| **每个 C++ 实现都配一个独立的 numpy 参考实现** | 两个实现给出同一答案，那个答案才可信。自己检查自己的代码会掩盖自己的缺陷 |
| **估计与评分由不同的代码完成** | 估计用 C++，ATE/RPE 评分用 Python。同一套代码兼任两者，两个缺陷会互相抵消 |
| **对照组正是我们声明"不使用"的那条描述子流水线** | 战胜一个随意挑选的弱对手毫无意义 |
| **先确认这个度量是否具有区分力** | 对所有输入都给出同一数值的指标，即使通过也什么都没证明 |
| **失败必须喧闹** | 打印"已保存"却没写文件的工具、永远不可达却显示绿色的测试 —— 都按缺陷处理 |

这些规则实际抓出了什么，全部记录在 [docs/06-results.md](../06-results.md) —— 不只是成功的
实验，还包括**五个被否决的假设，以及我自己制造的缺陷**。

---

## 项目目的

### 1. 用三个层次求解对应问题

```
Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA
```

| 层次 | 名称 | 作用 |
|---|---|---|
| **Tier 0** | ECDA | 直接光度对齐。最小化像素亮度残差，不使用描述子 |
| **Tier 1** | TCG | 令牌星座几何。依据物体（YOLO 检测）的相对布局定位 |
| **Tier 2** | SPA | 结构对齐。用平面法向与距离补足退化的自由度 |

`α_k(E)` 不是手写的分支，而是由**环境证据 E**（黑暗、雾霾、运动模糊、纹理贫乏……）计算得出。
代码里既没有夜间分支也没有雨天分支 —— 退化**只**表现为各信息源贡献信息量的减少。

### 2. 在同一份数据上并排测量这一主张

`results/bench/index.html` 把**经典方法放在左边、WME 放在右边**，在一个界面上比较轨迹、ATE、
RPE 和速度。此外还有原生程序 `wme_bench_viewer.exe`，它进一步用各系统**自己估计的位姿**
把深度图三维重建出来 —— 漂移会表现为拖影般的点云，而不只是一个数字。

<div align="center">

| 数据集 | 序列数 | 对照组 | 结果 |
|---|---|---|---|
| **TUM RGB-D**（室内，手持） | 16 | ORB+PnP、`cv2.Odometry` | **9 – 6**（对两者中较强的一方） |
| **KITTI odometry**（室外，车载） | 4 | ORB+PnP | **4 – 0** |

</div>

只有 TUM 时，这个比较是 9–6。加入 KITTI 后**翻转为 2–2**。追查原因发现：ECDA 把双目深度当作
真值 —— 60 m 处的深度误差是 ±4 m，却与 6 m 处（±4 cm）以相同权重进入方程。把这份不确定性
移入残差方差之后（系数由 `c = σ_d/(f·B)` **推导**而来，并非调参），结果变为 4–0。

> **此前所有写着"WME 更好"的句子，都是 TUM 的性质，而不是算法的性质。**
> 完整经过见 [§25.20–25.21](../06-results.md)。

### 3. 同样明确它不是什么

- **这不是 ORB-SLAM3。** 对照组只做 ORB 检测 → 汉明匹配 → RANSAC PnP，没有回环检测，没有
  光束法平差。比较**仅在"里程计对里程计"的意义上成立**
- **两边都没有光束法平差**，最多到位姿图
- 退化（雾霾）实验是把散射方程作用在真实 TUM 帧上，并使用**实测深度**生成的，并非自然退化数据

哪些结论**尚未确立**，列在 [docs/06-results.md §26](../06-results.md)。

---

## 仓库结构

```
WorldVision-SLAM/
├── include/wme/              公开头文件（27）
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
├── src/                      实现（约 17 kLOC）
│
├── tools/                    可执行实验程序
│   ├── tum_odometry.cpp      WME 里程计
│   ├── tum_baseline.cpp      ORB+PnP 对照组 <- 正是我们声明不使用的那条
│   ├── kitti_convert.cpp     KITTI -> TUM 布局 + StereoSGBM 深度
│   ├── bench_viewer.cpp      左右并排的基准测试程序
│   ├── tum_loopclose.cpp     对称回环检测（ORB vs TCG）
│   ├── tum_degrade.cpp       基于实测深度的散射退化
│   └── ...                   fusion、relocalize、tcg_density、plane_density
│
├── tests/                    C++ 测试（236 例）
│
├── python/
│   ├── wme/
│   │   ├── reference/        * 与 C++ 对照的 numpy 参考实现
│   │   ├── localization/     ecda.py —— DirectAligner 的参考实现
│   │   ├── geometry/         spa.py, planes.py
│   │   ├── eval/             ATE / RPE / Umeyama、TUM 读取  <- 只负责评分
│   │   ├── graph/            位姿图、因子、光度 SLAM
│   │   ├── sim/ world/       合成场景、世界模型状态与预测
│   │   └── association/ calib/ planner/
│   ├── bindings/             pybind11 -> wme._core
│   ├── tools/                实验与基准脚本（32）
│   └── tests/                Python 测试（639 例）
│
├── docs/06-results.md        * 全部实测结果与失败记录
├── results/bench/index.html  * 左：经典方法 / 右：WME
└── .github/workflows/        linux、windows-msvc、sanitizers、python
```

---

## 数据集

两个数据集都**不包含**在仓库中（合计 51 GB），均可用脚本复现。

### TUM RGB-D —— 室内、手持、实测深度

Kinect 结构光传感器，因此深度是**测量值**。使用 16 个序列。

```bash
python python/tools/tum_fetch_all.py     # 全部 16 个，完整下载
python python/tools/check_datasets.py    # 校验索引与磁盘是否一致
```

> `tum_fetch.py` 默认只取 9 秒窗口，加 `--all` 取完整序列。注意该工具现在会把
> `rgb.txt`/`depth.txt` 裁剪到实际存在的文件。此前保留完整索引却只有部分图像，导致所有工具
> 静默跳过缺口 —— 标称"1419 帧"的一次运行实际只处理了 165 帧。

> 内参与畸变系数因 freiburg 分组而异。固定成一套会让其他分组产生**看似合理的误差，而不是失败**。

### KITTI odometry —— 室外、车载、双目

没有深度，**必须由我们生成** —— 这正是 `StereoDepth`（OpenCV SGBM）前端首次成为必需的地方。
这里的深度是**估计值**而非测量值，这一区分正是 §25.21 的核心。

```bash
python python/tools/fetch_kitti.py       # 21.6 GB，支持断点续传
build/tools/wme_kitti_convert data/kitti/dataset 00 data/kitti_00 --stride 2
```

视差搜索范围由场景最近距离**推导**而来。若沿用默认值，SGBM 不会说"超出范围"，而是给出
**看似合理的错误答案**（实测：深度尺度偏差 2.42 倍）。

| 项目 | 数值 |
|---|---|
| 已下载序列 | 00–21（00–10 有真值轨迹） |
| 已转换并评测 | 00、04、05、07 |
| 分辨率 / 焦距 / 基线 | 1241×376 / 718.86 px / 0.537 m |
| 有效深度比例（SGBM） | 67 – 74 % |

---

## 运行方法

### 1. 前置条件

| 项目 | 版本 |
|---|---|
| 编译器 | MSVC 2022 / GCC 11+ / Clang 14+（C++20） |
| CMake | 3.24 以上 |
| OpenCV | 4.8 以上 —— `core imgproc imgcodecs videoio calib3d highgui dnn features2d` |
| Python | 3.10 以上 + `numpy scipy pytest pybind11` |
| （可选）ONNX Runtime | 1.22 —— 用于 YOLO 令牌掩码 |

> OpenCV 组件中漏掉 `features2d` 只会在链接阶段报错。
> `cmake/WmeDependencies.cmake` 的注释里记录了这个陷阱。

### 2. 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3. 测试 —— 请从这里开始

```bash
ctest --test-dir build --output-on-failure     # C++    236 例
cd python && python -m pytest -q               # Python 639 例
```

Python 侧相当一部分是 **C++ ↔ numpy 差分测试**。这里显示绿色意味着两个独立实现给出了同一
答案，而这是本仓库中任何数字值得相信的唯一依据。

### 4. 基准测试 → 左右对比程序

```bash
python python/tools/bench_run.py               # 运行两个系统并评分
python python/tools/bench_report.py            # -> results/bench/index.html
python python/tools/bench_export.py            # -> results/bench/viewer.tsv
build/tools/wme_bench_viewer                   # 原生程序
```

部分重跑：

```bash
python python/tools/bench_run.py --only kitti --merge   # 只跑 KITTI，保留其余
python python/tools/bench_run.py --skip-run             # 不重新估计，只重新评分
```

程序按键：`SPACE` 播放 · `,` `.` 单帧 · `N`/`P` 切换序列 · `1`/`2` 切换各面板显示的模型 ·
`A`/`D` 旋转视角 · `W`/`S` 缩放 · `F` 截图 · `Q` 退出。

该程序**不计算 ATE/RPE**，只显示 `bench_run.py` 计算的结果 —— 因为一个指标有两份实现，正是
屏幕上的数字与文档中的数字开始不一致的起点。

### 5. 单个序列

```bash
# TUM
build/tools/wme_tum_odometry data/rgbd_dataset_freiburg1_xyz out.txt
build/tools/wme_tum_baseline data/rgbd_dataset_freiburg1_xyz orb.txt
python python/tools/tum_eval.py data/rgbd_dataset_freiburg1_xyz out.txt

# KITTI —— 深度上限与不确定性系数来自数据集本身
build/tools/wme_tum_odometry data/kitti_00 out.txt \
    --kf-dist 1.0 --depth-max 60 --depth-sigma-rel 7.8e-4
```

### 6. 其他实验

```bash
python python/tools/baseline_cv2.py       # 第三方对照组（cv2.Odometry）
python python/tools/bench_degrade.py      # 雾霾扫描
python python/tools/stereo_validate.py    # 用 TUM 实测深度检验双目深度
python python/tools/loop_optimize.py      # 位姿图回环优化
```

---

<div align="center">

**完整结果、失败记录与被否决的假设 → [docs/06-results.md](../06-results.md)**

</div>
