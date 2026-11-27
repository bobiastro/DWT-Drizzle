/// Re-exported so the GUI and the CLI print the same release string.
pub use dwt_backend::VERSION;

pub const LICENSE_TEXT: &str = "\
分区加权叠加（PWS）+ Drizzle 几何重采样

本软件的方法基于 D.Cikey 提出的分区权重叠加法
（Partition-Weighted Stacking，PWS）。

原文（GitHub）：
https://github.com/DDCikey/
Partition-Weighted-Stacking/tree/main/docs

PDF 版可在微信公众号「小丁的星空」下载。

软件作者：打野波比

版权归原作者所有；本软件是在该方法公开描述基础上做的
独立实现，并增加了配准、drizzle、GPU 加速、自动裁边、
评价与调参等功能。";

pub const HELP_TEXT: &str = "\
DWT · Drizzle 使用说明

一、常用页
1. 素材目录选择「已校准、未对齐」的 XISF 目录（WBPP 的 calibrated）。
2. 输出目录选择成品位置，成品名决定 PWS_<成品名>_<滤镜>.xisf。
3. 点「扫描通道」列出可叠通道；不选 = 全部，也可只勾选 H / O / S 等。
4. 清晰度 A：彩机建议 6，黑白建议 20，14 折中。
5. 排异 κ：3.5σ，越大越保守；星点数 120；限帧 0 = 全部。
6. Drizzle：勾选后做几何重采样；取消 = 1× 网格，仍会配准。
7. 点「开始叠加」。日志实时滚动，可随时点「停止」。

二、高级页
包含 FWHM 搜索、单帧份额上限、R 场包络（窗 / 噪声项 / 形状幂）、
排异尺度、清晰度范围、flux_pow、match_scale、彩机 CFA 与
rgb_balance 等全部参数。修改后点「存为默认」写入 config.toml。

三、叠加评价
选择成品 .xisf，自动寻找同名 _R_q4.npy / _neff_q4.npy / _weights.txt，
显示天空背景、噪声、星点、FWHM 与径向曲线，以及 N_eff / R 统计。

四、权重场
载入成品后显示 R 场与 N_eff 的伪彩色图；可切换 N_eff 对数色标。
";

pub const URL: &str = "https://github.com/DDCikey/Partition-Weighted-Stacking/tree/main/docs";
