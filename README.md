# DWT · Drizzle

> 在 DWT（PWS 分区加权叠加）之上，补上**配准 + Drizzle 几何重采样**，让一堆未对齐的
> 原始子帧直接变成一张更干净、更锐的线性成品；彩机（OSC / 拜耳阵列）也能直接读。

当前版本 **v2.7.4** ｜ 作者 打野波比（GitHub [@bobiastro](https://github.com/bobiastro)）

BUG 反馈 / 使用问题：**微信 15795179443**

![主界面](docs/images/ui-main.png)

---

## 它做的事

上游 DWT 只做「加权叠加」，前提是帧**已经对齐**。这个版本把前后两段补齐：

| 能力 | 说明 |
| --- | --- |
| **配准** | 星点检测 + 三角配准兜底，支持旋转/尺度/平移；失败帧进「被排除」表并写明原因 |
| **Drizzle** | 1× / 2× / 1.5× 几何重采样，`pixfrac` 可调；2× 后再折回原生口径，星点更细 |
| **PWS 叠加** | 沿用上游算法：帧清晰度 C、帧信噪比 S、分区权重场 R 三者融合的一次加权平均 |
| **彩机直读** | `--cfa auto` 自动识别 RGGB/BGGR/GRBG/GBRG，全分辨率拆色后三通道共用一次配准，另出 `_RGB.xisf` |
| **输出网格 / 对齐裁剪** | `--frame` 直接按 WBPP master 的画幅与朝向出图；`--crop-cover` 按覆盖计数裁掉薄边 |
| **输入已对齐** | `--aligned` 跳过配准与插值，比走一遍 drizzle 再重采样更锐（实测 2.47px vs 2.66/2.72） |
| **自动评价 / 权重场 / 运行监控** | 星点 FWHM 分半径统计、轴比、N_eff、排除帧表、SNR 大字 |
| **自动调参** | `--tune N` 抽样试 A × R 场定义 × 排异系数，按 FWHM/信噪比给建议 |
| **GPU 加速** | 按显存预算分带跑 drizzle 与配准，结果与 CPU 逐位一致 |

![叠加评价](docs/images/ui-eval.png)

---

## 快速开始

### 用打包版（推荐，不用装 Python）

从 [Releases](../../releases) 下载对应平台：

| 平台 | 文件 |
| --- | --- |
| Windows x64 | `DWT-Drizzle-2.7.zip` |
| macOS Apple Silicon | `DWT-Drizzle-2.7-macos-arm64.zip` |
| macOS Intel | `DWT-Drizzle-2.7-macos-intel.zip` |
| 已有 Python 的 Mac | `DWT-Drizzle-2.7-macos.zip` |
| 三平台合集 | `DWT-Drizzle-2.7-all.zip` |

Windows 解压后双击 **`启动-图形界面.bat`**。四个空填完就能跑：

1. **素材目录**：WBPP 的 `calibrated\`（已校准、**未**对齐；`registered\` 不要用，除非勾「输入已对齐」）
2. **输出目录**：随便给个空文件夹
3. **成品名**：比如 `m81`
4. **几何目录**：WBPP 的 `registered\`（里面有一堆 `.xdrz`）——跨夜有旋转时必填

### 从源码跑

```bash
pip install -r requirements.txt
python main.py            # 图形界面
python main.py --cli --help   # 命令行
```

一次最小命令行：

```bash
python main.py --cli --photos "D:/M81/calibrated" --out "D:/M81/out" --tag m81 \
       --cfa auto --drizzle --drizzle-scale 2 --drizzle-pix 1.0 --gpu auto
```

### 在 PixInsight WBPP 中使用 Rust 后端

Windows 运行 `.\core-rs\wbpp\install.ps1`，macOS 运行 `bash core-rs/wbpp/install.sh`。
脚本会安装 Rust 后端和 WBPP Pipeline Builder。在 WBPP 的 **Pipeline → Pipeline scripting → Pipeline builder**
中启用并选择安装后显示的 `pipeline-builder-ui.js`。[安装与使用说明](core-rs/wbpp/README.md)

---

## 成品

| 文件 | 内容 |
| --- | --- |
| `PWS_<成品名>_<滤镜>.xisf` | 单通道线性 Float32 成品 |
| `PWS_<成品名>_R/_G/_B.xisf` | 彩机三通道成品 |
| `PWS_<成品名>_RGB.xisf` | 彩机合成（三通道像素级对齐，合成前按 25% 分位天空对齐 R/B） |
| `*_R.npy` / `*_neff.npy` / `*_weights.txt` | 分区权重场、等效帧数、逐帧权重（评价页/权重场页要用） |
| `*_autocrop.xisf` | 「自动裁边」的产物（按覆盖率裁掉四角与薄边带） |

成品是**原始线性数据**，请在 PixInsight / Siril 里做 STF 拉伸再看。

![示例：M81/M82，60 帧彩机 300 s](docs/images/example-m81.jpg)

---

## 目录结构

```
main.py              入口（GUI / --cli 两条，共用 core/pws_params.py 的参数定义）
gui.py               图形界面（tkinter）
xisf_io.py           XISF 读写（纯标准库实现）
DWT_DetailCore.py    「细节/平滑」分区判据（上游）
core/
  pws.py             叠加主流程（读帧→星表→R 场→drizzle→导出）
  pws_params.py      参数唯一真源：界面控件与命令行都由它生成
  drizzle.py         drizzle 重采样（CPU/GPU）
  cfa.py             拜耳阵列识别与拆色
  matchref.py        --frame / --match 的画布与变换对齐
  trireg.py          三角配准（星点乱序时的兜底）
  cropcover.py       自动裁边
  bggrad.py          逐帧平面背景梯度
  tune.py            自动调参
  gpu_dz.py, gpu_band.py, xdrz.py
tests/               自检（python -m unittest / 直接跑单文件）
tools/               分析脚本：对比、体检、制图
docs/                优化历程记录、截图
```

---

## 自检

```bash
cd tests && python test_save.py && python test_nan.py && python test_crop.py
```

GUI 冒烟（不需要显示器）：

```bash
python gui.py --selftest     # 打印 SELFTEST-OK
```

---

## 与上游的关系

- 叠加算法 **PWS（Partition-Weighted Stacking）** 由 **D.Cikey** 提出，
  上游实现：<https://github.com/DDCikey/Partition-Weighted-Stacking>（MIT）。
- 本仓库的 `DWT_DetailCore.py`、`core/pws.py`、`core/pws_params.py` 以上游为本，
  **新增**的是配准、drizzle、彩机拆色、自动裁边、评价/调参、图形界面等；
  `drizzle.patch` 是相对上游的完整差异，便于对照。
- 上游版权与 MIT 许可全文保留在 [LICENSE](LICENSE)；方法的署名（D.Cikey）请在使用与
  二次分发时一并保留。

## 许可

MIT，见 [LICENSE](LICENSE)。
