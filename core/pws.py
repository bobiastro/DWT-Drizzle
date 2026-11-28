"""PWS（Partition-Weighted Stacking，分区权重叠加）引擎

实现《分区权重叠加法的原理与实现》所述算法。三类权重经一个连续分区场融合为
逐像素权重，再做唯一一次加权平均（式 6、式 7）：

    W[i,r] = C_i^(A·R_r) · S_i^(B·(1−R_r))         再逐像素归一 mean_i(W) = 1

    C_i  帧清晰度权重（帧级标量）  = FWHM_中位 / FWHM_i        越大越锐   式(4)
    S_i  帧信噪比权重（帧级标量）  = σ_中位 / σ_i               越大越干净 式(5)
    R_r  分区权重场（逐像素连续场）  ∈ [0,1]   1 = 细节型  0 = 朦胧型    式(8)

    B = 2 由逆方差加权的最优性给出（3.3 节），不需标定；A 是唯一需经验标定的
    参数。分区场 R 由最锐参考帧的亮度归一包络构造，其径向轮廓只含 r/σ_psf、
    与星点亮度无关，故换目标不需重新标定。

输入前提：已校准（暗/平/偏）、已对齐（旋转+平移）的帧；本软件不做校准与对齐。
输出：线性 Float32 灰度。全程只有线性运算（曝光归一常数、量纲回乘常数、
    天空加性平移、加权平均），不做任何拉伸 / 归一 / 截断。

第三方依赖（见 THIRD_PARTY_LICENSES.md）：
    photutils.detection.DAOStarFinder   星点检测（3.5 节）
    photutils.psf.fit_fwhm              高斯轮廓拟合 FWHM（3.5 节）
    scipy.ndimage                       滤波与形态学运算
"""
from __future__ import annotations

import argparse

try:  # 包内（python -m core.pws）与脚本直跑两种入口都要能 import
    from . import cfa as _cfa
except Exception:
    try:
        import cfa as _cfa
    except Exception:  # 模块缺失时退化为纯单色
        _cfa = None

# CFA 上下文：值由 _run 设置，prepare_frames 读帧时消费。
#   ch=None 表示按普通单色帧读（不解马赛克）。
_CFA = {'ch': None, 'pattern': None, 'method': 'bilinear',
        'raw': False, 'origin': (0, 0)}
_MISC = {'crop_origin': (0, 0)}     # 当前组的裁剪起点 (x0,y0)，--frame 折算用
import contextlib
import io
import csv
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

# 仓库根在上两级之外（xisf_io.py / DWT_DetailCore.py 位于仓库根）；pws_params.py
#   与 pws.py 同目录，按顶层模块导入，不依赖包结构。
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
for _p in (str(ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from DWT_DetailCore import EXPTIME_REF, read_frame                      # noqa: E402
from pws_params import PARAMS, PwsParams, cli_flag                      # noqa: E402
import bggrad
import cropcover                                                          # noqa: E402
try:                                                                   # noqa: E402
    import matchref
except Exception:              # 缺 scipy 等依赖时只是没有裁切功能
    matchref = None

# ---------------------------------------------------------------------------
# 引擎参数（与界面同源，唯一定义见 pws_params.PARAMS）
# ---------------------------------------------------------------------------
A_SHARP = 20.0       # 细节区清晰度幂上限：R=1 处 W = C^A。
                     #   A 是唯一需要标定的参数，默认 20 在 NGC 7331（166 帧）与
                     #   NGC 6888（24 帧）上共用同一套参数：成品 FWHM 5.51/3.99px
                     #   （无分区 7.12/5.12px），星周亮度平台降幅 24.3%/20.3%
                     #   （无分区基线 23.3%/21.3%）；
                     #   A 偏小更保守（A=12：FWHM 5.68/4.13px，天空 σ 仅升 3.8%/2.3%），
                     #   A 偏大更锐（A=30：FWHM 5.36/3.88px）但天空 σ 升到 6.9%/6.3%、
                     #   细节区等效帧数降至 3.93。
B_SNR = 2.0          # 朦胧区信噪比幂上限：R=0 处 W = S^B。B=2 由逆方差加权给出，不需标定
# R 的映射只有一种：亮度归一包络（见 _r_envelope）。其径向轮廓只含 r/σ_psf，
#   与星点亮度无关 → 过渡恒落在 ~2σ_psf 内，换目标不改参数、也不改含义。
ENV_WIN_MULT = 2.0   # 包络归一的局部极大窗 = 该倍数 × 2σ_psf（约 4σ_psf 见方）
ENV_EPS = 16.0       # 分母噪声项（以 σ 为单位）：压住空白天区，使 R→0
                     #   12 组 568 帧全量标定：ε 由 8 增至 16，星点尾部余晖收紧
                     #   0.6%~36%（11/12 组），星核半径与天空噪声不变；再增至 24 收益
                     #   进入测量噪声带，故取 16。
ENV_P = 1.0          # 形状幂；E 已是 exp(−2r/σ) 量级，1.0 即够，>1 收得更紧
STAR_GROW = 0.0      # 星点锐帧区外扩（×σ_psf）：最亮星的外扩量，按星点亮度自适应，
                     #   暗星拿 STAR_GROW_FLOOR 的保底、亮星扩满。0 = 关闭（原行为）。
STAR_GROW_LO_SNR = 20.0    # 自适应外扩的下端：局部峰值 < 该倍数×天空σ → 只拿保底量
STAR_GROW_HI_SNR = 1000.0  # 自适应外扩的上端：局部峰值 ≥ 该倍数×天空σ → 扩满 star_grow
STAR_GROW_FLOOR = 0.5      # 保底系数：拖线长度由风决定、与星点亮度无关，故每颗星都必须
                           #   拿到足以盖住自身拖线的那部分外扩（grow=0 时会把这些星的拖线
                           #   又放回来：实测中等星伸长率 0.10→0.15）。亮星再按亮度加成。
TRAIL_AP = 3.7             # 测拖线用的二阶矩口径 = 该倍数 × σ_psf（≈99.9% 的高斯能量）
TRAIL_QUANTILE = 90.0      # 自动外扩取"逐帧拖线长度"的该百分位：瞄准最糊的那几帧
                           #   （漏进星点分区的正是它们；取中位会被大量好帧稀释）
R_FLOOR = 0.0        # R 的全局下限：R_eff = R_FLOOR + (1−R_FLOOR)·R
_RDEC = max(1, int(__import__('os').environ.get('DWT_RDEC', '2') or 2))  # R 场降采样倍率（1=全分辨率，2≈4× 提速）
                     #   用于抑制星周亮度平台（"亮—暗—亮"伪影环）。环的幅度取决于
                     #   两区有效 PSF 之差：加全局下限使空白天区也保留部分清晰度偏好，
                     #   糊帧在全图被压低 → 两区 PSF 差缩小 → 环变浅。代价是空白天区
                     #   等效帧数下降（信噪比）。0 = 关闭（朦胧区严格吃满信噪比）。
R_WIN_MULT = 2.0     # 结构 RMS 的空间窗 = 该倍数 × σ_psf（过大将摊薄点源能量）
REJ_K = 3.5          # 逐像素排异：|偏差| > κ·1.4826·MAD + m·R·(a_r−天空) 即剔除（a_r = 跨帧中位比较基准）。
                     #   κ 以 σ 为单位（1.4826·MAD = σ 估计）；2 更严，1 会成批误排噪声尾
_lsp_half = 0.5
REJ_MAX_FRAC = 1 / 3  # 剔除比例上限：超额被剔帧数 > 该比例 × 总帧数 → 判为共有结构而非
                      #   孤立离群，该像素不排异
FLUX_R = 7.0         # 测吞吐的定孔径半径（原生像素）
FLUX_POW = 0.0       # 吞吐项默认关闭：天空背景下 T^2 会把亮帧过度加权（见 docs/实验记录-分支E.md）
N_STAR = 120         # 固定星表星数（测 C_i 用；中位数在 100 颗以上已稳定）
REF_SHARP_N = max(1, int(__import__('os').environ.get('DWT_REF_K', '8') or 8))
                    # 参考像 = 最锐的这么多帧的等权中位（默认 k=8，DWT_REF_K 可覆盖）

# 各阶段的并行进程上限（实际进程数还受"核数−1"与内存预算约束，见 plan_workers）
#   叠加：每进程持一条条带的临时量，内存随进程数倍增 → 上限保守
#   读帧：受磁盘带宽约束，多开无益
#   测星点：每帧仅取星点小窗（约 0.5 MiB），几乎不占内存 → 上限可放宽到核数量级
CAP_WORKERS = 8
CAP_FWHM = 12
#   叠加（drizzle 分带）：内存带宽受限，实测 4/6/8/10 进程为 25.7/23.8/24.8/24.4s
#   （M81 40 帧 1600² ×2，6 核以上机器），6 最划算；仍受"核数−2"约束
class _Skip(Exception):
    """内部信号：该 GPU 路径按配置跳过，不是失败，不写日志。"""


_FS_CACHE: dict = {}

CAP_DRIZZLE = 32

# 逐帧行固定纹去除（row FPN）：默认开，DWT_NO_ROWFPN=1 关掉做 A/B
_ROWFPN = bool(os.environ.get('DWT_ROWFPN'))   # 实验开关：默认关，实测未改善成品行纹


# ---------------------------------------------------------------------------
# 与界面之间的三个钩子（引擎本体不依赖任何界面代码，命令行下自动退回 stdout）
# ---------------------------------------------------------------------------

class Cancelled(Exception):
    """用户请求终止：在检查点抛出，由调用方清理并提示"""


_LOG_SINK: List = [None]        # callable(str)            → 界面 append 日志
_PROGRESS: List = [None]        # callable(阶段, 百分比, 说明) → 界面进度
_CANCEL: List = [None]          # callable() -> bool       → 取消检查回调

# 阶段 → 全局百分比区间（单调递增，界面只显示换算后的这一个数）
_PHASE_SPAN = (
    ('读帧', 0.00, 0.25),
    ('星表', 0.25, 0.30),
    ('测星点', 0.30, 0.45),
    ('R 场', 0.45, 0.52),
    ('叠加', 0.52, 0.95),
    ('输出', 0.95, 1.00),
)
_SPAN = {name: (lo, hi) for name, lo, hi in _PHASE_SPAN}

# 多滤镜分组：_GROUP = [组号, 总组数, 日志前缀标签]。
#   单组时 = [0, 1, '']——_emit 不缩放、log 不加前缀。
_GROUP: List = [0, 1, '']


def _set_group(gi: int, gn: int, label: str = '') -> None:
    """进入某滤镜组：进度换算缩到该组的全局区间，日志行加 [滤镜] 前缀"""
    _GROUP[0], _GROUP[1] = gi, max(1, gn)
    _GROUP[2] = f'[{label}] ' if label else ''


def _emit_console(msg: str) -> None:
    """往控制台写一行；遇到控制台编码不支持的字符（²、σ、·、→）只替换不崩。

    中文 Windows 默认 cp936，`print('…%d² 窗…')` 会抛 UnicodeEncodeError，
    而这条日志出现在「自动定位裁剪中心」分支，等于整个命令行跑不起来。
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        try:
            sys.stdout.write(str(msg).encode(enc, 'replace').decode(enc, 'replace') + '\n')
            sys.stdout.flush()
        except Exception:
            sys.stdout.write(str(msg).encode('ascii', 'replace').decode('ascii') + '\n')
            sys.stdout.flush()


def log(msg: str = '') -> None:
    """引擎唯一输出口：界面接管后写成日志行，未接管则打 stdout（回调异常也退回）。
    多滤镜分组时自动带组前缀（空行不加）——否则几十上百行日志分不清是哪组滤镜的。"""
    pre = _GROUP[2]
    if pre and msg.strip():
        msg = f'{pre}{msg}'
    sink = _LOG_SINK[0]
    if sink is None:
        _emit_console(msg)
        return
    try:
        sink(msg)
    except Exception:
        _emit_console(msg)


def _emit(phase: str, frac: float, text: str = '') -> None:
    """报进度：frac 是**本阶段内**完成度，此处换算成全局百分比后交给界面。
    多滤镜分组时再缩进当前组的全局区间：(组号 + 组内值) / 总组数。"""
    fn = _PROGRESS[0]
    if fn is None:
        return
    lo, hi = _SPAN.get(phase, (0.0, 1.0))
    v = lo + (hi - lo) * min(1.0, max(0.0, float(frac)))
    gi, gn = _GROUP[0], _GROUP[1]
    if gn > 1:
        v = (gi + v) / gn
    try:
        fn(phase, v, text)
    except Exception:
        pass


def _check_cancel() -> None:
    """取消检查点：读帧每帧、测 C_i 每帧、叠加每条带各一处"""
    fn = _CANCEL[0]
    if fn is not None and fn():
        raise Cancelled('用户终止')


def set_hooks(on_log=None, on_progress=None, should_cancel=None) -> None:
    """绑定/解绑三个钩子（传 None 即解绑）"""
    _LOG_SINK[0] = on_log
    _PROGRESS[0] = on_progress
    _CANCEL[0] = should_cancel


def parse_center(text: str) -> Optional[Tuple[int, int]]:
    """解析裁剪中心 'y,x'（界面里就是一行文本）；留空返回 None → 自动定位延展源"""
    t = (text or '').strip()
    if not t:
        return None
    parts = [v for v in t.replace('，', ',').split(',') if v.strip()]
    if len(parts) < 2:
        raise SystemExit(f'裁剪中心要写成 y,x 两个整数（收到 {text!r}）')
    return (int(float(parts[0])), int(float(parts[1])))


# ---------------------------------------------------------------------------
# 阶段 0：帧加载与帧级归一
# ---------------------------------------------------------------------------


# ---------------------------- PSF 拟合告警 ----------------------------
# photutils 每颗拟合失败的星都会 logger.warning 一条
#   "WARNING: One or more fit(s) may not have converged. ... [photutils.psf.utils]"
# astropy 在协方差奇异时还会抛 RuntimeWarning（cov_x 里除零）。两者都是
#   "这颗星不参与中位数"的正常噪声，一帧 120 颗、几百帧就是上万行，必须收敛成一行汇总。
_FIT_STAT = {'n': 0, 'conv': 0, 'frames': 0}


class _FitWarnCounter(logging.Handler):
    """挂在 photutils logger 上数"未收敛"条数，同时不让它冒到 stderr"""

    def emit(self, rec):
        try:
            msg = rec.getMessage()
        except Exception:
            return
        if 'converged' in msg:
            _FIT_STAT['conv'] += 1


@contextlib.contextmanager
def _quiet_fit():
    """拟合期间静音 photutils 日志与 astropy/numpy 的 RuntimeWarning，只计数"""
    lg = logging.getLogger('photutils')
    prev_level, prev_prop = lg.level, lg.propagate
    h = _FitWarnCounter()
    lg.addHandler(h)
    # level 必须放到 DEBUG，否则 WARNING 记录在 Logger 层就被丢掉、计不到数；
    # propagate=False + 不装输出 handler，保证它们不会冒到 stderr
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', UserWarning)
            yield
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev_level)
        lg.propagate = prev_prop


def fit_warn_line() -> str:
    """拟合告警汇总（无告警返回空串）"""
    n, c = _FIT_STAT['n'], _FIT_STAT['conv']
    n = max(n, c)
    if not n:
        return ''
    return (f'[星点] PSF 拟合 {n} 颗次，其中 {c} 颗未收敛/协方差奇异'
            f'（{100.0 * c / n:.1f}%，已剔除，不影响中位数）')


def reset_fit_stat() -> None:
    _FIT_STAT['n'] = _FIT_STAT['conv'] = _FIT_STAT['frames'] = 0


FRAME_SUFFIXES = ('.xisf', '.fit', '.fits', '.fts')


def collect_frames(photos_str: str, allow_aligned: bool = False) -> Tuple[List[Path], List[Path]]:
    """把素材声明展开成帧文件列表（多源：目录与单张可混用，用 ; 分隔）

    目录按**递归**收集：深空数据常按 目标/日期/滤镜 分级存放，子文件夹里的帧也收。
    XISF / FITS 混合树的取舍按**子目录**判断——同一目录里两种格式并存视为同一批
    数据的两种导出，只收 XISF（避免同帧重复叠加），不同子目录各用各的格式、互不影响。
    单张文件按扩展名直接收录。去重按规范路径（同一文件被两个来源重复声明
    只算一帧），最终按路径全名排序保证读取顺序确定。返回 (帧列表, 来源列表)。
    """
    sources: List[Path] = []
    seen_src = set()
    for raw in (photos_str or '').split(';'):
        raw = raw.strip().strip('"')
        if not raw:
            continue
        p = Path(raw)
        key = str(p.resolve()).lower()
        if key in seen_src:
            continue
        seen_src.add(key)
        sources.append(p)
    if not sources:
        raise SystemExit('请选择素材（目录或单张帧；多个来源用 ; 分隔）')

    files: List[Path] = []
    seen_file = set()
    for src in sources:
        if src.is_dir():
            by_dir: Dict[Path, List[Path]] = {}
            for f in src.rglob('*'):
                if f.is_file() and f.suffix.lower() in FRAME_SUFFIXES:
                    by_dir.setdefault(f.parent, []).append(f)
            batch: List[Path] = []
            for d in sorted(by_dir, key=lambda p: str(p).lower()):
                grp = sorted(by_dir[d], key=lambda f: str(f).lower())
                xisf = [f for f in grp if f.suffix.lower() == '.xisf']
                batch.extend(xisf or grp)
            if not batch:
                raise SystemExit(f'{src}（含子文件夹）里没有 XISF / FITS 帧')
        elif src.is_file():
            if src.suffix.lower() not in FRAME_SUFFIXES:
                raise SystemExit(f'不支持的帧格式：{src.name}（{src}）')
            batch = [src]
        else:
            raise SystemExit(f'素材不存在：{src}')
        for f in batch:
            key = str(f.resolve()).lower()
            if key not in seen_file:
                seen_file.add(key)
                files.append(f)
    if not files:
        raise SystemExit('素材里没有帧')
    files.sort(key=lambda f: str(f).lower())
    files, n_kind, n_reg = filter_stackable_frames(files, allow_aligned=allow_aligned)
    if n_kind:
        log(f'[分组] 跳过 {n_kind} 张非 Light 帧（Flat / Dark / Bias）')
    if n_reg:
        log(f'[分组] 跳过 {n_reg} 张 registered/ 下的帧（已被重采样对齐，'
            f'再 drizzle 只会二次插值）')
    if not files:
        raise SystemExit('素材里只有非 Light 帧或已对齐帧，没有可叠加的原始 Light 帧')
    return files, sources


def is_registered_path(p: Path) -> bool:
    """路径里出现 registered / registration / aligned 段 → 认为已被重采样对齐

    WBPP 把对齐（Lanczos-3 重采样）的帧放在 registered/ 下，尺寸已被裁到参考帧；
    这类帧再喂给 drizzle 只会二次插值，拿不回任何采样率。
    """
    for seg in p.parts:
        if seg.lower() in ('registered', 'registration', 'aligned'):
            return True
    return False


CALIB_DIRS = {'flat', 'flats', 'dark', 'darks', 'bias', 'biases',
              'darkflat', 'darkflats', 'calibrationframes'}


def filter_stackable_frames(files: List[Path], allow_aligned: bool = False):
    """只留"能直接叠加"的帧：Light 帧 + 没被对齐过的路径。

    返回 (保留的帧, 跳过的非 Light 张数, 跳过的 registered 张数)。
    文件头读不出帧类型（老格式/损坏）时保留——宁可叠进去再排异，
    也不要因为读不到头就悄悄丢帧。
    """
    from DWT_DetailCore import read_frame_hint
    keep: List[Path] = []
    n_kind = n_reg = 0
    for f in files:
        if is_registered_path(f) and not allow_aligned:
            n_reg += 1
            continue
        try:
            _fil, typ = read_frame_hint(f)
        except Exception:
            typ = ''
        if typ and typ not in ('LIGHT', 'LIGHTFRAME', 'OBJECT'):
            n_kind += 1
            continue
        # 兜底：帧头读不出类型时按目录名判断（WBPP 常把校准场放 flat/ dark/ bias/ 下）
        if any(seg.lower() in CALIB_DIRS for seg in f.parts[:-1]):
            n_kind += 1
            continue
        keep.append(f)
    return keep, n_kind, n_reg


_INPUT_FLAGS = {'per_channel': False}
_CH_PATH_RE = re.compile(r'(?:^|__)([RGB])$')


def _channel_from_path(f: Path) -> str:
    """从路径判断这一帧是不是"已按通道拆分"的单色帧（WBPP registered/__R 之类）。

    这类目录里 R/G/B 各自是独立文件，像素已经是解过马赛克的结果，
    绝不能再当拜耳马赛克按 2×2 相位取样。
    """
    for seg in reversed(f.parts[:-1]):
        m = _CH_PATH_RE.search(seg)
        if m:
            return m.group(1)
    m = _CH_PATH_RE.search(f.stem)
    return m.group(1) if m else ''


def group_frames_by_filter(files: List[Path]) -> Dict[str, List[Path]]:
    """按帧头滤镜名分组（分组叠加第一步）。

    只读文件头不解码像素（read_filter_hint：FITS 读 FILTER 卡，XISF 只读
    XML 元数据段），几百帧只需数秒。头里没有滤镜信息的帧归入 ''（单独成组
    照常叠加，日志点名）；同一滤镜名的帧合为一组。顺序 = 首次出现
    （files 已按路径排序，结果确定）。
    """
    from DWT_DetailCore import read_filter_hint
    groups: Dict[str, List[Path]] = {}
    t0 = time.perf_counter()
    for f in files:
        _check_cancel()
        name = ''
        try:
            name = read_filter_hint(f)
        except Exception:
            pass
        if not name:
            _ch = _channel_from_path(f)
            if _ch:
                name = _ch
                _INPUT_FLAGS['per_channel'] = True
        groups.setdefault(name, []).append(f)
    log(f'[分组] 读头 {len(files)} 帧 {time.perf_counter() - t0:.1f}s → '
        f'{len(groups)} 组：'
        + '  '.join(f'{k or "无滤镜信息"}×{len(v)}' for k, v in groups.items()))
    if '' in groups:
        log('[分组] 注意：以上帧的文件头没有滤镜信息（FILTER / Instrument:Filter:Name），'
            '已单独成组')
    return groups


def scan_channels(photos_str: str) -> Dict:
    """只读帧头，列出素材里有哪些通道（滤镜）与各自帧数，不读像素

    给界面/命令行一个"先看有哪些通道再决定叠哪个"的入口：
    校准场（Flat/Dark/Bias，按帧头类型与目录名双重排除）与 registered/ 已对齐帧
    都在这里就被剔除，剩下的按 FILTER 帧头分组。
    """
    files, _sources = collect_frames(photos_str)
    groups = group_frames_by_filter(files)
    chans = [{'name': k, 'label': (k or '无滤镜信息'), 'n': len(v),
              'dir': str(v[0].parent)} for k, v in groups.items()]
    chans.sort(key=lambda c: (-c['n'], c['label'].lower()))
    return {'channels': chans, 'total': len(files),
            'dirs': sorted({c['dir'] for c in chans})}


def _safe_seg(s: str) -> str:
    """文件名段安全化：路径非法字符换下划线"""
    return re.sub(r'[\\/:*?"<>|]+', '_', (s or '').strip()).strip()


def _out_stem(tag: str, fname: str, multi: bool) -> str:
    """成品文件主名：一律以 PWS_ 开头。
    有成品名 → PWS_<成品名>，分组时再接滤镜段 → PWS_<成品名>_<滤镜>；
    无成品名 → PWS_<滤镜>（单组且无滤镜信息时退化为 PWS）。"""
    t, f = _safe_seg(tag), _safe_seg(fname)
    parts = ['PWS']
    if t:
        parts.append(t)
    if f and (multi or not t):
        parts.append(f)
    return '_'.join(parts)


def resolve_center(frames: List[Path], crop: int,
                   center: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """裁剪中心留空时**自动定位延展源**（核心层 locate_extended_source，
    32px 分块中值法找最亮延展团）。默认值不能写死目标坐标：换目标时会静默用错中心，
    裁到空白天区，导致 R 场与星表失效。全幅（crop=0）时返回 None。
    """
    if not crop or crop <= 0 or center is not None:
        return center
    from DWT_DetailCore import locate_extended_source
    # 只用第一帧定位会被"过中天翻转"或不同夜晚的取景带偏：翻转帧里目标
    # 出现在镜像位置，按它裁出来的窗对其它帧是错的天区，之后配准全灭。
    # 改成抽若干帧各定位一次取中位，抗单帧异常。
    n = len(frames)
    idx = sorted(set(int(round(k * (n - 1) / 7.0)) for k in range(8))) if n > 1 else [0]
    cys, cxs = [], []
    for i in idx:
        with contextlib.redirect_stdout(io.StringIO()):
            d0, _ = read_frame(frames[i])
        cy, cx = locate_extended_source(np.asarray(d0, dtype=np.float32), bin_px=32)
        del d0
        cys.append(float(cy))
        cxs.append(float(cx))
    cy = int(round(float(np.median(cys))))
    cx = int(round(float(np.median(cxs))))
    spread = max(max(cys) - min(cys), max(cxs) - min(cxs))
    if spread > 0.1 * float(crop):
        log('[裁剪] 警告：各帧定位中心相差 %.0fpx（>裁窗的 10%%，可能有翻转帧或取景差异）；'
            '已取中位 (%d,%d)，必要时用 --center 手动指定' % (spread, cy, cx))
    log('[裁剪] 自动定位延展源中心 (y,x)=(%d,%d) → %d^2 窗（抽样 %d 帧取中位）'
        % (cy, cx, crop, len(idx)))
    return (int(cy), int(cx))


_PICK_CACHE: Dict[str, str] = {}


def _frames_state_file() -> Path:
    base = os.environ.get('APPDATA') or os.environ.get('HOME') or '.'
    return Path(base) / 'DWT_Drizzle' / 'frames_dirs.txt'


def _record_frames_dir(p: Path) -> None:
    """记住用过的 memmap 目录，供下次启动清理崩溃残留（最多 8 条）。"""
    try:
        f = _frames_state_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        old = []
        if f.exists():
            old = [ln.strip() for ln in f.read_text(encoding='utf-8').splitlines() if ln.strip()]
        cur = str(p)
        items = [cur] + [x for x in old if x != cur]
        f.write_text(chr(10).join(items[:8]) + chr(10), encoding='utf-8')
    except Exception:
        pass


def pick_frames_dir(need_bytes: int, want: Optional[str] = None,
                    frame_hint: Optional[str] = None) -> Path:
    """给 memmap 帧文件挑盘：需要 1.25× 余量

    全幅 166 帧约 166×6388×9576×4B = 40.6 GiB。若目标盘余量不足，会在读完大量帧
    之后才因空间不足失败，浪费大量时间。故开跑前先按余量选盘，并把选择写入日志。
      优先级（取**第一个够用**的，不追求余量最大）：
        ① 用户 --frames-dir
        ② 素材所在盘（大文件通常就放在这块盘上，余量大；且不会写满系统盘）
        ③ 软件所在盘  ④ 系统 TEMP  ⑤ 其余固定盘按余量降序

    注意 ② 是 2.7.66 加的：之前默认写软件安装盘，装在 C: 的用户处理
    20+ GiB 素材时会把系统盘写满（ENOSPC 发生在读了几十 GB 之后，很难受）。
    """
    import shutil
    _key = str(want or '')
    _hit = _PICK_CACHE.get(_key)
    if _hit is not None and Path(_hit).parent.exists():
        return Path(_hit)
    need = int(need_bytes * 1.25)
    head: List[Path] = []
    if want:
        head.append(Path(want))
    # 素材所在盘：frames_dir 未指定时，按数据盘根目录建临时目录（驱动器根，好找也好清）
    try:
        import os as _os
        _p0 = Path(frame_hint) if frame_hint else None
        if _p0 is not None:
            _anchor = Path(_p0.anchor) / '_DWT_frames'
            if _anchor not in head:
                head.append(_anchor)
    except Exception:
        pass
    if getattr(sys, 'frozen', False):
        # 打包版以软件所在位置为先（与源码版一致）：用 exe 所在目录而非 __file__
        #   （后者在 onedir 下会落进 _internal 依赖目录）
        head.append(Path(sys.executable).resolve().parent / '_DWT_frames')
    else:
        head.append(Path(__file__).resolve().parent / '_DWT_frames')
    tmp = os.environ.get('TEMP', '') or os.environ.get('TMP', '')
    if tmp:
        head.append(Path(tmp) / 'DWT_frames')

    def free_of(p: Path):
        try:
            anchor = p if p.exists() else p.parent
            if not anchor.exists():
                return None, 0
            return anchor, shutil.disk_usage(str(anchor)).free
        except OSError:
            return None, 0

    seen, head_ok, rest = set(), [], []
    for p in head:
        anchor, free = free_of(p)
        if anchor is None or str(anchor).lower() in seen:
            continue
        seen.add(str(anchor).lower())
        head_ok.append((p, free))
    for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
        root = Path(f'{letter}:\\')
        if not root.exists() or str(root).rstrip('\\').lower() in seen:
            continue
        seen.add(str(root).rstrip('\\').lower())
        try:
            rest.append((root / 'DWT_frames', shutil.disk_usage(str(root)).free))
        except OSError:
            continue
    rest.sort(key=lambda t: -t[1])                   # 其余盘按余量降序
    for p, free in head_ok + rest:                   # head 保序优先，其余按余量
        if free >= need:
            try:
                p.mkdir(parents=True, exist_ok=True)
                probe = p / '.write_test'
                probe.write_bytes(b'')
                probe.unlink()                       # 只读目录（如装在 Program Files）在此跳过
            except OSError:
                continue
            log(f'[memmap] 帧文件目录 {p}（该盘余量 {free / 2 ** 30:.1f} GiB，'
                f'需 {need / 2 ** 30:.1f} GiB）')
            _PICK_CACHE[_key] = str(p)
            _record_frames_dir(p)
            return p
    raise SystemExit(f'没有任何盘能放下 {need / 2 ** 30:.1f} GiB 的 memmap 帧文件'
                     f'（需要 1.25× 余量）。请用 --frames-dir 指到大盘。')


class FrameSet:
    """帧集合：**memmap 文件**（常规）与 **父进程内存 list**（无盘可落时的兜底）

    两种后端都是"一整块 (n,h,w) 的 float32 数组"，对外接口完全一致。

    必须提供 memmap 后端：全幅一帧 6388×9576×4B = 244 MiB，166 帧 = 40.6 GiB，
               足以吃满内存；而排异要求"同一像素跨帧比较"，条带内层必须是帧，
               无法用"帧外层循环 + 累加器"绕开随机访问。故落盘 memmap，按条带随机读。
    不使用 shared_memory 承载"装得进内存"的帧：页文件支撑的共享段在 Windows 上按
               **整段大小**计入系统提交量，多个工作进程各自映射同一大段会失败于
               WinError 1450/1455（系统资源不足 / 页面文件太小）。文件支撑的 memmap
               其脏页由文件本身兜底、不计入提交量，且文件页驻留系统缓存，读写速度
               与内存一致 → 统一走这一条路。
    父进程内存 list 只在"一个盘都放不下"时兜底：它不能跨进程（spawn 建子进程需将
               整份帧 pickle 过去），故该后端自动退化为串行。见 prepare_frames。

    天空加性平移（off_i）统一在**读出时减**，两种后端一致：
               memmap = 盘上即"曝光归一后"，off_i 读出时减。
               两种后端给出的数值逐位一致：都只做 D − 常数。
    """

    def __init__(self, n: int, shape: Tuple[int, int], resident: bool,
                 arrays: Optional[List[np.ndarray]] = None,
                 mm: Optional[np.ndarray] = None, path: Optional[Path] = None):
        self.n = int(n)
        self.shape = (int(shape[0]), int(shape[1]))
        self.resident = bool(resident)
        self.arrays = arrays       # 兜底后端（父进程内存，不可跨进程，仅串行）
        self.mm = mm               # memmap 视图（可跨进程）
        self.path = path           # memmap 的文件路径（兜底后端为 None）
        self.skies = np.zeros(n, dtype=np.float64)
        self.sigma = np.zeros(n, dtype=np.float64)
        self.off = np.zeros(n, dtype=np.float64)
        self.sky0 = 0.0            # 平移后的公共天空水平（排异信号项的零点）
        self.metas: List[Dict] = []
        self.names: List[str] = []
        self.bgcoef = None         # (n,5) 逐帧背景梯度系数；None = 不校正
        self.bgdeg = 0             # 0 关闭 / 1 平面 / 2 二阶
        self.bgrdeg = 0            # 径向偶次项阶数（2/4，0 关闭）
        self.bgrcoef = None        # (n,2) 径向系数
        self.bg_stats: Dict = {}
        self.bgshape = (int(shape[0]), int(shape[1]))
        # 读帧时顺手量的逐帧星点 FWHM（C 权重）；NaN = 这帧没量出来
        self.fwhms: Optional[np.ndarray] = None
        # 实测块中位背景场（LN 风格）：(n, GH, GW)，None = 不用
        self.bgtiles = None
        self.bgtile_block = 32

    def __len__(self) -> int:
        return self.n

    def frame(self, i: int) -> np.ndarray:
        """第 i 帧全幅（曝光归一 + 天空平移 + 背景梯度后）"""
        if self.mm is None:
            d = self.arrays[i]
        else:
            d = self.mm[i] - np.float32(self.off[i])
        m = self.bg_model(int(i), 0, self.shape[0], self.shape[1])
        if m is None:
            return d
        if self.mm is None:                 # 兜底后端：arrays 是共享容器，不能就地改
            # 注意：bg_model 可能返回 float64，直接 d - m 会先开一份 float64 全幅
            # 临时数组（6248×4176 = 208 MB），多进程叠加时足以把内存打爆。
            m32 = np.asarray(m, np.float32)
            return np.subtract(d, m32, dtype=np.float32)
        d -= m
        return d

    def strip(self, i: int, y0: int, y1: int) -> np.ndarray:
        """第 i 帧的 [y0,y1) 行条带"""
        if self.mm is None:
            d = self.arrays[i][y0:y1]
        else:
            d = self.mm[i, y0:y1] - np.float32(self.off[i])
        m = self.bg_model(int(i), int(y0), int(y1), self.shape[1])
        if m is None:
            return d
        if self.mm is None:
            m32 = np.asarray(m, np.float32)   # 同 frame()：避免 float64 临时数组
            return np.subtract(d, m32, dtype=np.float32)
        d -= m
        return d

    def bg_model(self, i: int, y0: int, y1: int, w: int):
        """第 i 帧 [y0,y1)×w 的背景梯度模型（None = 不校正）

        bgtiles 非空时用「实测块中位」代替低阶多项式：块是真实测量，能吃到
        月光/薄云这类不规则大尺度梯度（多项式一阶/二阶跟不上），这就是 PI 的
        LocalNormalization 那套思路的本土版。
        """
        if self.bgtiles is not None:
            return self._tile_field(i, y0, y1, w)
        if self.bgcoef is None or self.bgdeg <= 0:
            return None
        _rc = None
        if self.bgrdeg > 0 and self.bgrcoef is not None:
            _rc = self.bgrcoef[int(i)]
        return bggrad.model_strip(self.bgcoef[i], self.bgdeg, y0, y1, w,
                                  self.bgshape[0], rcoef=_rc)

    def _tile_field(self, i: int, y0: int, y1: int, w: int) -> np.ndarray:
        """把第 i 帧的块中位场放大到 [y0,y1)×w（块内常数，网格已 3×3 平滑）"""
        g = self.bgtiles[int(i)]                    # (GH, GW)，已减去自身中位
        gh, gw = g.shape
        H = int(self.bgshape[0])
        # 网格行号 → 输出行：块中心对齐
        yy = np.clip(((np.arange(y0, y1) + 0.5) * gh / max(H, 1)).astype(np.int32), 0, gh - 1)
        xx = np.clip(((np.arange(w) + 0.5) * gw / max(w, 1)).astype(np.int32), 0, gw - 1)
        return np.asarray(g[yy][:, xx], np.float32)

    def stack_strips(self, y0: int, y1: int) -> np.ndarray:
        """条带内所有帧堆成 (n, y1-y0, W) —— 排异与加权平均的唯一输入"""
        return np.stack([self.strip(i, y0, y1) for i in range(self.n)])

    def fwhm(self, i: int, stars: List[Tuple[int, int]]) -> float:
        """第 i 帧在固定星表上的 FWHM 中位 —— 只取星点小窗，不物化整帧"""
        if self.mm is None:
            return measure_fwhm(self.arrays[i], stars)
        return measure_fwhm_frame(self.mm, i, self.off, stars)

    def trail(self, i: int, stars: List[Tuple[int, int]],
              sigma_psf: float) -> Tuple[float, float]:
        """第 i 帧的长轴方差中位与长轴方向（估拖线长度用）"""
        if self.mm is None:
            return measure_trail(self.arrays[i], stars, sigma_psf)
        return measure_trail_frame(self.mm, i, self.off, stars, sigma_psf)

    @property
    def parallelable(self) -> bool:
        """能否跨进程交给工作进程（memmap 可以，兜底列表不行）"""
        return self.mm is not None

    @property
    def reader(self) -> Tuple[object, tuple]:
        """工作进程打开本帧集的 (initializer, initargs)"""
        return _worker_open, (str(self.path), self.n, self.shape[0], self.shape[1], self.off)

    @property
    def writer(self) -> Tuple[object, tuple]:
        """读帧阶段（要**写**帧数据）的 (initializer, initargs)"""
        return _worker_open_rw, (str(self.path), self.n, self.shape[0], self.shape[1])

    @property
    def resident_bytes(self) -> int:
        """帧数据占着**系统内存**的量 —— 用来给叠加阶段的条带高度算内存预算

        帧在盘上时这份内存由系统缓存兜着，但它确实与工作进程争内存，
            故照样从预算里扣（宁可条带矮一点，也不要因为缓存被挤掉而反复读盘）。
            盘都放不下、退回父进程内存 list 时，它记的是进程自己的占用。
        """
        return self.n * self.shape[0] * self.shape[1] * 4 if self.resident else 0

    #: 三通道共用同一份帧存储时置 True：通道结束时只 flush，不释放/不删文件，
    #: 由 _run 在最后一个通道跑完后统一 close()（见 _run 末尾）。
    _shared = False

    def close(self, keep: bool = False) -> None:
        if self.mm is not None:
            self.mm.flush()
            del self.mm
            self.mm = None
        self.arrays = None
        if not keep and self.path is not None and self.path.exists():
            try:
                self.path.unlink()
                log(f'[memmap] 已删除临时帧文件 {self.path}')
            except OSError as e:
                log(f'[memmap] 删除失败（可手动删）：{self.path}  {e}')


class _PlaneFS:
    """给注册/测星用的“拆色视图”代理：帧存储里是原始马赛克，这里按需拆色。

    真 CFA drizzle 需要原始样本进散射；但配准与 PSF 估计必须看插值后的图层，
    否则星点被 2×2 马赛克切碎。用代理把两种视角分开，互不污染。
    """

    def __init__(self, fs, pattern: str, ch: str):
        self._fs = fs
        self._pat = pattern
        self._ch = ch
        self.shape = fs.shape
        self.n = fs.n

    def __len__(self):
        return len(self._fs)

    def __getattr__(self, name):
        return getattr(self._fs, name)

    def frame(self, i: int):
        return _cfa.plane(self._fs.frame(int(i)), self._pat, self._ch)

    def strip(self, i: int, y0: int, y1: int):
        return _cfa.plane(self._fs.strip(int(i), int(y0), int(y1)),
                           self._pat, self._ch, y0=int(y0))



def prepare_frames(frames: List[Path], label: str, crop: int,
                   center: Optional[Tuple[int, int]],
                   limit: Optional[int] = None, frames_dir: str = '',
                   mem_frac: float = 0.55, keep_frames: bool = False,
                   bgdeg: int = 0, bg_r: int = 0, star_n: int = 0,
                   ) -> FrameSet:
    """读帧 → 裁剪 → float32 → **曝光归一**（×EXPTIME_REF/t_i），
    同时逐帧量天空中位与噪声尺度；按内存预算自动决定驻留还是落 memmap。

    曝光归一必须在最前面：本方法允许"曝光时间可以不同"，归一后所有帧
              等价于同一曝光时长，σ_i 才可以直接跨帧比较（S_i 的前提）。
    天空中位与 σ_i 在这一趟里顺带算掉：memmap 后端若事后再算，
               就要把 40 GiB 从盘上重读一遍。
    """
    files = list(frames)
    if limit:
        files = files[:limit]
    n = len(files)
    log(f'[帧] {n} 帧  来源 {label}')

    with contextlib.redirect_stdout(io.StringIO()):
        probe, _ = read_frame(files[0])
    fh, fw = probe.shape[:2]
    del probe
    if crop and crop > 0:
        h = w = min(int(crop), fh, fw)
    else:
        h, w = fh, fw
    if crop and crop > 0:
        cy, cx = (h // 2, w // 2) if center is None else (int(center[0]), int(center[1]))
        half = crop // 2
        y0 = max(0, min(cy - half, fh - crop))
        x0 = max(0, min(cx - half, fw - crop))
    else:
        y0 = x0 = 0
    _MISC['crop_origin'] = (int(x0), int(y0))   # --frame 折算裁剪原点要用
    # CFA 拆色靠 (y0,x0) 的奇偶确定拜耳相位：中心点或裁剪边长是奇数时，
    # 每帧相位会整体偏一格，红蓝互换且星点出棋盘格。
    if _CFA.get('ch') and (y0 % 2 or x0 % 2):
        _y, _x = y0 - y0 % 2, x0 - x0 % 2
        log(f'[CFA] 裁剪起点 ({x0},{y0}) 不是偶数，已挪到 ({_x},{_y}) 以保持拜耳相位')
        y0, x0 = _y, _x

    need = n * h * w * 4
    avail = _avail_bytes()          # 与并行阶段共用同一套预算（含 DWT_AVAIL_MB 覆盖）

    # 一律落盘 memmap（见 FrameSet 文档：页文件支撑的共享段在 Windows 上按整段大小
    #   记入提交量，装不下就 1450/1455；文件支撑的映射不进提交量，页仍在系统缓存里）。
    #   只有"一个盘都放不下"时才退回父进程内存 list —— 那会退化为串行。
    #   内存够不够只用于提示：装得进内存时盘上这份基本全程待在缓存里。
    resident = need <= int(avail * mem_frac)
    mm = None
    mmpath = None
    arrays: Optional[List[np.ndarray]] = None
    try:
        # 文件名带 pid：上一次崩溃（或并行候选）留下的 .dat 被占用时，
        # Windows 会抛 WinError 32 让整个 run 直接失败；换名字即可绕开。
        mmpath = pick_frames_dir(need, frames_dir or None, str(frames[0]) if frames else None) / (
            f'pws_frames_{h}x{w}_{n}_{os.getpid()}.dat')
        mm = np.memmap(str(mmpath), dtype=np.float32, mode='w+', shape=(n, h, w))
    except SystemExit as e:
        log(f'[帧] 没有盘能放下 {need / 2 ** 30:.2f} GiB 的帧文件 → 退回父进程内存'
            f'（该组数据将串行）：{e}')
        mmpath, mm, resident = None, None, True
        arrays = []
    log(f'[帧] 形状 ({h},{w})  内部曝光归一到 {EXPTIME_REF:g}s（读帧时按 基准/曝光 缩放，'
        f'导出前再乘回，成品量纲与素材一致）  需 {need / 2 ** 30:.2f} GiB  '
        f'可用 {avail / 2 ** 30:.1f} GiB → '
        f'{"父进程内存（串行）" if mm is None else "memmap 流式（并行）"}')

    fs = FrameSet(n, (h, w), (mm is None), arrays=arrays, mm=mm, path=mmpath)
    if _CFA.get('raw') and _CFA['ch']:
        _CFA['origin'] = tuple(int(v) for v in _MISC.get('crop_origin', (0, 0)))
        log('[CFA] 真 CFA drizzle：帧存储保留原始马赛克（不插值）')
    # C 权重用的星表：在探针帧（第 0 帧）上建一次，交给读帧进程顺手量 FWHM。
    #   探针帧在裁剪之后重新读一次（只 1 帧，0.2s），换来省掉读帧后那一整遍。
    fw_stars: List[Tuple[int, int]] = []
    fw_guess = 6.0
    _fw_ser = np.full(n, np.nan, np.float64)   # 串行分支顺手量的逐帧 FWHM
    if star_n and int(star_n) > 0:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _p, _ = read_frame(files[0])
            _p = np.ascontiguousarray(_p[y0:y0 + h, x0:x0 + w]).astype(np.float32)
            if _CFA['ch'] and _cfa is not None:
                _p = _cfa.plane(_p, _CFA['pattern'], _CFA['ch'])
            fw_stars = build_star_table(_p, n_star=int(star_n))
            _sig = _noise_sigma(_p)
            if _sig > 0:
                fw_guess = float(np.clip(2.355 * 1.6 * _sig / max(float(np.median(_p)), 1e-9),
                                         2.0, 12.0))
            del _p
            log(f'[C] 读帧时并行量星点 FWHM：探针帧上选出 {len(fw_stars)} 颗')
        except Exception as _e:
            fw_stars, fw_guess = [], 6.0
            log(f'[C] 探针帧建星表失败（{_e}），C 权重改回读帧后单独量')
    bg_grids: List[object] = [None] * n      # 逐帧块中位（仅 bgdeg>0 时填）
    t0 = time.perf_counter()
    # 逐帧独立（各读各的源文件、各写各的槽位）→ 可分给多进程。
    #   工作进程各自只读打开同一个帧文件，直接写自己那一格，
    #   父进程收回标量（帧号、天空中位、σ）+ 元数据字典（含 FITS 关键字与
    #   Property 原始元素，供成品透传；每帧几十 KB，pickle 代价可忽略）。
    #   每进程内存按"整帧 ×3"估：read_frame 的整帧缓冲 + float32 裁剪副本
    #   + median 内部副本（np.median 对浮点数组会先复制一份）。
    ini_rw, ini_rw_args = fs.writer
    nw = 0 if (not fs.parallelable or _no_parallel()) \
        else plan_workers(fh * fw * 4 * 4 + (512 << 20), '读帧')
    _ln_ok = [0]
    _ln_bad = [0]
    _ln_miss = [0]
    try:
        if nw > 1:
            fs.metas = [None] * n
            fs.names = [Path(str(f)).name for f in files]
            ex = ProcessPoolExecutor(max_workers=nw, initializer=ini_rw,
                                     initargs=ini_rw_args)
            try:
                jobs = ((i, str(files[i]), EXPTIME_REF, y0, x0, h, w,
                         int(bgdeg),
                         ('' if _CFA.get('raw') else _CFA['ch']),
                         _CFA['pattern'], fw_stars, fw_guess)
                        for i in range(n))
                _fwhm_buf = np.full(n, np.nan, np.float64)
                for i, med, sig, bgm, meta, fwk in ex.map(_task_read_frame, jobs,
                                                          chunksize=1):
                    _check_cancel()
                    fs.skies[i] = med
                    fs.sigma[i] = sig
                    bg_grids[i] = bgm
                    fs.metas[i] = meta
                    _fwhm_buf[i] = fwk
                    if (i + 1) % 20 == 0:
                        log(f'  读帧 {i + 1}/{n}  {time.perf_counter() - t0:.0f}s')
                    _emit('读帧', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
            if fw_stars:
                fs.fwhms = _fwhm_buf
        else:
            for i, f in enumerate(files):
                _check_cancel()                     # 读帧阶段也要能终止
                with contextlib.redirect_stdout(io.StringIO()):
                    data, meta = read_frame(f)
                d = np.ascontiguousarray(data[y0:y0 + h, x0:x0 + w]).astype(np.float32)
                del data
                d *= np.float32(EXPTIME_REF / float(meta['exptime']))
                if _CFA['ch'] and _cfa is not None and not _CFA.get('raw'):
                    d = _cfa.plane(d, _CFA['pattern'], _CFA['ch'])
                if _ROWFPN:
                    _strip_rowfpn(d)
                if False:  # [removed]
                    _xn = xnml.sibling_path(str(f))
                    if os.path.exists(_xn):
                        _fld = xnml.load_fields(_xn)
                        if _fld is not None:
                            xnml.apply_to(d, _fld)
                            _ln_ok[0] += 1
                        else:
                            _ln_bad[0] += 1
                    else:
                        _ln_miss[0] += 1
                _smp = d[::4, ::4]
                med = float(np.median(_smp))
                del _smp
                if fw_stars:
                    try:
                        _v = measure_fwhm_each(d, fw_stars, fwhm_guess=fw_guess)
                        _ok = np.isfinite(_v)
                        if _ok.sum() >= 5:
                            _fw_ser[i] = float(np.median(_v[_ok]))
                    except Exception:
                        pass
                fs.skies[i] = med
                fs.sigma[i] = _noise_sigma(d)
                if bgdeg:
                    bg_grids[i] = bggrad.block_medians(d, bggrad.BG_BLOCK)
                if mm is not None:
                    mm[i] = d
                else:
                    arrays.append(d)
                del d
                fs.metas.append(meta)
                fs.names.append(Path(str(f)).name)
                if (i + 1) % 20 == 0:
                    log(f'  读帧 {i + 1}/{n}  {time.perf_counter() - t0:.0f}s')
                _emit('读帧', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
        if fw_stars:
            fs.fwhms = _fw_ser
    except BaseException:
        # 读帧半途失败（终止 / 工作进程异常 / 盘满）都要把半成品临时帧文件清掉，
        #   否则一块几十 GiB 的文件会一直占着盘；清完再把异常往上抛
        fs.close(keep=keep_frames)
        raise

    if False:  # [removed]
        log(f'[LN] WBPP 本地归一化：已应用 {_ln_ok[0]} 帧（乘 Scale + 加 ZeroOffset，'
            f'网格双线性插值），缺 .xnml {_ln_miss[0]} 帧，解析失败 {_ln_bad[0]} 帧')
        if _ln_ok[0] == 0:
            log('[LN] 没有任何帧找到 .xnml，本次按原样叠加')
    # 天空加性平移：基准帧 = 天空最接近全体帧中位的那一帧。memmap 后端只记 off_i、
    #   读出时减（见 FrameSet.strip）；父进程内存后端则在建集时就地烘焙（见下）。
    anchor = int(np.argmin(np.abs(fs.skies - np.median(fs.skies))))
    sky0 = float(fs.skies[anchor])
    fs.off = fs.skies - sky0
    fs.sky0 = sky0
    if arrays is not None:
        for i in range(n):
            arrays[i] -= np.float32(fs.off[i])
        fs.off[:] = 0.0            # 已就地烘焙，读出时不再减
    if mmpath is not None:
        mm.flush()
    # ---- 背景梯度：逐帧加性低阶面（在 sky0 平移之后拟合，模型不含常数）----
    if int(bgdeg) > 0:
        coefs, bstats = bggrad.fit_all(bg_grids, int(bgdeg), (h, w),
                                       fs.sigma, offs=fs.skies - sky0, log=log,
                                       rdeg=int(bg_r or 0))
        if coefs is not None:
            fs.bgcoef = coefs
            fs.bgdeg = int(bgdeg)
            fs.bgrdeg = int(bg_r or 0)
            fs.bgrcoef = bstats.get('rcoefs') if int(bg_r or 0) > 0 else None
            fs.bg_stats = bstats
        else:
            log('[背景] 梯度拟合不可用（块数不足），按不校正处理')
    # ---- 实测块中位背景场（--bg-tiles 1）：比多项式更贴不规则梯度 ----
    if False:  # [removed]
        try:
            from scipy.ndimage import gaussian_filter as _gf
            _G = [g for g in bg_grids if g is not None]
            if len(_G) == n:
                _st = np.stack([np.asarray(g, np.float32) for g in _G])
                _med = np.median(_st.reshape(n, -1), axis=1).reshape(n, 1, 1)
                _st = _st - _med                       # 只留形状，常数已被天空平移吸收
                _st = np.stack([_gf(t, 1.0, mode='nearest') for t in _st])
                fs.bgtiles = _st.astype(np.float32)
                fs.bgtile_block = int(bggrad.BG_BLOCK)
                log(f'[背景] 用实测块中位场（{_st.shape[1]}×{_st.shape[2]} 格 × '
                    f'{fs.bgtile_block}px）代替多项式：逐帧减自身中位后 3×3 平滑')
        except Exception as _e:
            fs.bgtiles = None
            log(f'[背景] 块场构造失败，按多项式处理：{_e}')
    bg_grids = None
    span = float(fs.skies.max() - fs.skies.min())
    log(f'[帧] 耗时 {time.perf_counter() - t0:.1f}s  '
        f'{"父进程内存 " + f"{fs.resident_bytes / 2 ** 30:.2f} GiB" if mm is None else f"盘上 {need / 2 ** 30:.2f} GiB（{mmpath.parent}）"}')
    log(f'[归一] 天空 {fs.skies.min():.4e}~{fs.skies.max():.4e}  基准帧 #{anchor}={sky0:.4e}  '
        f'平移极差 {span:.3e}（天空的 {span / max(sky0, 1e-30):.2%}）')
    return fs


def frame_sigma(fs: FrameSet) -> np.ndarray:
    """逐帧噪声尺度 σ_i = 1.4826 × MAD（天空背景主导，稳健、抗星点/目标）
    已在 prepare_frames 的读帧趟里量好，这里直接取。
    """
    return fs.sigma


# ---------------------------------------------------------------------------
# 阶段 A：C_i（星点 FWHM，开源工具）
# ---------------------------------------------------------------------------

def build_star_table(ref: np.ndarray, n_star: int = N_STAR,
                     iso_mult: float = 5.0) -> List[Tuple[int, int]]:
    """固定星表：DAOStarFinder 检测 → 未饱和 → 孤立 → 取最亮 n_star 颗

    必须使用固定星表：逐帧检测时阈值 = 5×该帧噪声，糊帧检出得少，"帧质量"
               会混进"星点强度"，导致糊帧被系统性高估（式 4 要求所有帧在同一批
               星点位置上测量）。
    检测器采用 DAOStarFinder，保证各帧星点位置与测光口径一致。
    """
    import photutils
    if hasattr(photutils, 'future_column_names'):
        photutils.future_column_names = True
    from photutils.detection import DAOStarFinder

    sky = float(np.median(ref))
    nz = 1.4826 * float(np.median(np.abs(ref - sky)))
    fwhm0 = 6.0
    with _quiet_fit():
        tbl = DAOStarFinder(fwhm=fwhm0, threshold=5.0 * nz,
                            exclude_border=True)(ref - sky)
    if tbl is None or len(tbl) == 0:
        return []
    names = tbl.colnames
    xk = 'x_centroid' if 'x_centroid' in names else 'xcentroid'
    yk = 'y_centroid' if 'y_centroid' in names else 'ycentroid'
    x = np.asarray(tbl[xk], float)
    y = np.asarray(tbl[yk], float)
    pk = np.asarray(tbl['peak'], float)

    vmax = float(np.max(ref))
    keep = pk < 0.8 * vmax
    x, y, pk = x[keep], y[keep], pk[keep]
    order = np.argsort(pk)[::-1]
    x, y, pk = x[order], y[order], pk[order]

    # 孤立判据：与已选星的最小距离 ≥ iso_mult × 拟合窗半宽，保证窗口内无混星
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x, y]))
    sel: List[Tuple[int, int]] = []
    h, w = ref.shape
    margin = 26
    for i in range(len(x)):
        xi, yi = int(round(x[i])), int(round(y[i]))
        if xi < margin or yi < margin or xi >= w - margin or yi >= h - margin:
            continue
        if sel:
            nb = tree.query((x[i], y[i]), k=2)[0][1]
            if nb < iso_mult * 5.0:
                continue
        sel.append((xi, yi))
        if len(sel) >= n_star:
            break
    return sel


def _fwhm_each(kut, stars: List[Tuple[int, int]], fwhm_guess: float,
               half: int, fit_shape: int, with_flux: bool = False):
    """逐星量高斯 PSF FWHM 的公共实现；kut(xi,yi) 给一颗星的小窗（越界给 None）

    小窗来源有两处（整帧切片 / 帧后端直接取），两者的数值路径完全相同：
              逐元素减常数与切片顺序无关，故"只取小窗"与"先物化整帧再切"等价。
    """
    from photutils.psf import fit_fwhm
    out = np.full(len(stars), np.nan, dtype=np.float64)
    fl = np.full(len(stars), np.nan, dtype=np.float64)
    with _quiet_fit():
        for j, (xi, yi) in enumerate(stars):
            cut = kut(xi, yi)
            if cut is None:
                continue
            p = np.asarray(cut).astype(np.float64)
            yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
            rr = np.hypot(xx, yy)
            p -= float(np.median(p[(rr >= half - 3) & (rr <= half)]))
            _FIT_STAT['n'] += 1
            try:
                v = float(np.atleast_1d(
                    fit_fwhm(p, xypos=[(half, half)], fwhm=fwhm_guess,
                             fit_shape=fit_shape))[0])
            except Exception:
                _FIT_STAT['conv'] += 1
                continue
            if np.isfinite(v) and 0.5 < v < 40.0:
                out[j] = v
    return (out, fl) if with_flux else out


def measure_fwhm_each(img: np.ndarray, stars: List[Tuple[int, int]],
                      fwhm_guess: float = 6.0, half: int = 16,
                      fit_shape: int = 29) -> np.ndarray:
    """逐星量高斯 PSF FWHM，返回与 stars 等长的数组（失败/越界为 NaN）

    用 PSF 拟合而不是二阶矩：二阶矩对翼部极敏感（PI 的 FWHMEccentricity
               会把 r50 +0.2px / 翼部 +20% 放大成数 px），拟合口径才是"星点细不细"的稳定度量。
    拆出逐星版本：配对比较必须逐星做（同一颗星在两个栈上的差），
               只给中位数会丢掉分布信息。
    """
    h, w = img.shape

    def kut(xi: int, yi: int):
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            return None
        return img[yi - half:yi + half + 1, xi - half:xi + half + 1]

    return _fwhm_each(kut, stars, fwhm_guess, half, fit_shape)


def measure_fwhm_frame(mm: np.ndarray, i: int, off: np.ndarray,
                       stars: List[Tuple[int, int]], fwhm_guess: float = 6.0,
                       half: int = 16, fit_shape: int = 29, with_flux: bool = False,
                       search: int = 0):
    """从帧后端（memmap）直接取星点小窗量 FWHM 中位 —— **不物化整帧**

    一帧 244 MiB，而 120 颗星的小窗合起来才 ~0.5 MiB。先 mm[i] − off 物化整帧
              再切片，等于为了 0.5 MiB 去读写 244 MiB（memmap 后端就是每帧从盘上
              重读 244 MiB）；直接取小窗后这一步几乎免费。
    数值与 measure_fwhm(mm[i] − off, stars) 等价：两种写法都是"先做一次 float32
              减常数、再取同一片"，逐元素减法与切片顺序无关。
    """
    offi = np.float32(off[i])
    _n, h, w = mm.shape

    def kut(xi: int, yi: int):
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            return None
        # 逐帧重定心：帧间抖动会把星点挪出 seed 坐标，直接按 seed 坐标切窗会让
        # 拟合吃到背景/星点边缘（实测同一批星在不同帧给出 8.4px 与 3.2px 的假值）。
        # 在 ±search 内找最亮像元再把窗口居中到它；搜不到就退回原坐标。
        cx, cy = xi, yi
        if search > 0:
            x0, x1 = max(0, xi - half - search), min(w, xi + half + search + 1)
            y0, y1 = max(0, yi - half - search), min(h, yi + half + search + 1)
            big = mm[i, y0:y1, x0:x1] - offi
            if big.size:
                k = int(np.argmax(big))
                cy = y0 + k // big.shape[1]
                cx = x0 + k % big.shape[1]
                if (cx - half < 0 or cy - half < 0
                        or cx + half >= w or cy + half >= h):
                    cx, cy = xi, yi
        return mm[i, cy - half:cy + half + 1, cx - half:cx + half + 1] - offi

    if with_flux:
        v, fl = _fwhm_each(kut, stars, fwhm_guess, half, fit_shape, with_flux=True)
    else:
        v, fl = _fwhm_each(kut, stars, fwhm_guess, half, fit_shape), None
    okv = np.isfinite(v)
    if okv.sum() < 5:
        return (float('nan'), None) if with_flux else float('nan')
    fw = float(np.median(v[okv]))
    if not with_flux:
        return fw
    return fw, fl




def measure_flux_frame(mm, i, off, stars, th=0.0, kk=1.0, dx=0.0, dy=0.0,
                       r=None, half=16):
    """在**本帧**的位置上测星点流量（定孔径、环带平面扣背景）。

    (x, y) 是 seed 帧的星表坐标，本帧里同一颗星在
        X = kk*(cosθ·x − sinθ·y) − dx    Y = kk*(sinθ·x + cosθ·y) − dy
    （与 drizzle 里 X = k·R·x + u、u = −dx 的约定一致）。
    """
    rad = FLUX_R if r is None else float(r)
    _n, h, w = mm.shape
    offi = 0.0 if off is None else float(off[i])
    # 位移/旋转/尺度里只要有一个是 NaN（配准被拒的帧），就没法把星表投回来，
    # 直接返回全 NaN，交给上层按"这帧没有吞吐测量"处理。
    if not (np.isfinite(th) and np.isfinite(kk) and np.isfinite(dx) and np.isfinite(dy)):
        return np.full(len(stars), np.nan, np.float64)

    def _cut(X, Y):
        xi = int(round(X))
        yi = int(round(Y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            return None
        return mm[i, yi - half:yi + half + 1, xi - half:xi + half + 1] - offi

    th_r = np.radians(float(th))
    c, s = float(np.cos(th_r)), float(np.sin(th_r))
    inv = 1.0 / float(kk) if float(kk) else 1.0
    out = np.full(len(stars), np.nan, np.float64)
    for j, (x, y) in enumerate(stars):
        # 星表在 seed(=参考) 帧上；本帧里那颗星在 (1/k)·R(−θ)·(q + d)，
        # d 即 measure_shifts 给的 dx/dy（X = k·R·x − d 的逆）。
        qx, qy = x + float(dx), y + float(dy)
        X = inv * (c * qx + s * qy)
        Y = inv * (-s * qx + c * qy)
        p = _cut(X, Y)
        if p is None or p.shape[0] < half * 2 + 1:
            continue
        hh = p.shape[0]
        ctr = hh // 2
        yy, xx = np.mgrid[0:hh, 0:hh]
        yy = yy - ctr
        xx = xx - ctr
        rr = np.hypot(xx, yy)
        ring = (rr >= half - 4) & (rr <= half)
        if int(ring.sum()) < 8:
            continue
        am = np.stack([np.ones(int(ring.sum())), yy[ring], xx[ring]], axis=1)
        cf = np.linalg.lstsq(am, p[ring], rcond=None)[0]
        base = cf[0] + cf[1] * yy + cf[2] * xx
        out[j] = float((p - base)[rr <= rad].sum())
    return out


class _ResidentStack:
    """把常驻内存的 FrameSet 伪装成 memmap（只实现 shape 与按帧取图）。

    --limit 1 或内存充足时读帧走常驻路径、不建 memmap，而测吞吐等任务函数
    只认 _MEM['mm']；这里补一个薄适配器，避免 KeyError: 'mm'。
    """

    def __init__(self, fs, n: int, h: int, w: int):
        self._fs = fs
        self.shape = (int(n), int(h), int(w))

    def __getitem__(self, key):
        if isinstance(key, tuple):            # 形如 (i, ys, xs) 的切片访问
            i, ys, xs = key
            d = np.asarray(self._fs.frame(int(i)), np.float32)
            return d[ys, xs]
        return np.asarray(self._fs.frame(int(key)), np.float32)


def _task_flux(job):
    i, stars, th, kk, dx, dy = job
    fl = measure_flux_frame(_MEM['mm'], i, _MEM['off'], stars, th=th, kk=kk,
                            dx=dx, dy=dy)
    return int(i), fl

def trail_pixels(frame: np.ndarray, ds: int = 2, sigma: float = 6.0,
                 min_len: float = 400.0, box_r: int = 4):
    """数一帧里「长直线结构」（卫星 / 流星 / 飞机拖尾）的像素，一遍扫完。

    步骤（无迭代，比 RANSAC 更省时）：
      1. 1/ds 降采样 → 均值盒滤波高通（累积和实现，O(N)）；
      2. MAD 定噪声 σ，按 sigma·σ 二值化；
      3. 8 邻域连通域，逐个算二阶矩：长轴 3.5·√λmax ≥ min_len、长短轴比
         ≥ 6·ds、填充率 ≤ 0.45 判为拖尾。

    星点核是圆的（长短轴比 ≈ 1）、星云团块填充率高，都过不了这几条；
    只有细长直线结构会命中。返回 ``(覆盖像素数, 最长拖尾长度 px)``，
    干净帧返回 ``(0, 0.0)``。
    """
    a = np.asarray(frame, np.float32)
    if a.ndim != 2 or min(a.shape) < 64:
        return 0, 0.0
    ds = max(1, int(ds))
    sub = a[::ds, ::ds]
    # 均值盒滤波高通：拖尾是一条横贯上千像素的直线，大尺度上是"背景"，
    # 高通后它整条留下，星点只留下核心，星云留下的团块填充率高。
    r = max(1, int(box_r))
    k = 2 * r + 1
    ap = np.pad(np.asarray(sub, np.float64), r, mode='edge')
    cs = np.cumsum(np.cumsum(np.pad(ap, ((1, 0), (1, 0))), 0), 1)
    hh, ww = sub.shape
    bg = (cs[k:k + hh, k:k + ww] - cs[0:hh, k:k + ww]
          - cs[k:k + hh, 0:ww] + cs[0:hh, 0:ww]) / float(k * k)
    d = sub - bg
    s = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    if not np.isfinite(s) or s <= 0:
        return 0, 0.0
    m = d > float(sigma) * s
    if not bool(m.any()):
        return 0, 0.0
    lab, nl = ndimage.label(m, structure=np.ones((3, 3), int))
    if nl < 1:
        return 0, 0.0
    objs = ndimage.find_objects(lab)
    tot, longest = 0, 0.0
    _min_bbox = max(8, int(min_len / ds / 2))
    for kk in range(1, nl + 1):
        sl = objs[kk - 1]
        if sl is None:
            continue
        h_ = sl[0].stop - sl[0].start
        w_ = sl[1].stop - sl[1].start
        if max(h_, w_) < _min_bbox:
            continue
        ys, xs = np.nonzero(lab[sl] == kk)
        if len(xs) < 24:
            continue
        cxx = np.cov(np.vstack([xs.astype(np.float64), ys.astype(np.float64)]))
        ev = np.linalg.eigvalsh(cxx)
        if ev[1] <= 0.0:
            continue
        el = float(np.sqrt(ev[1] / max(ev[0], 1e-9)))          # 长短轴比
        ln = 3.5 * float(np.sqrt(ev[1])) * ds                  # 全长 ≈ 3.5σ 长轴
        fill = len(xs) / float(max(1, (xs.max() - xs.min() + 1)
                                   * (ys.max() - ys.min() + 1)))
        if el >= 6.0 * ds and ln >= float(min_len) and fill <= 0.45:
            tot += len(xs) * ds * ds
            longest = max(longest, ln)
    return int(tot), float(longest)


def measure_fwhm(img: np.ndarray, stars: List[Tuple[int, int]],
                 fwhm_guess: float = 6.0, half: int = 16,
                 fit_shape: int = 29) -> float:
    """固定星表上的 FWHM 中位（成品评价用这一个数）"""
    v = measure_fwhm_each(img, stars, fwhm_guess, half, fit_shape)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float('nan')
    return float(np.median(v))


def band_cfield(C, C_tiles, y0: int, y1: int, h: int, w: int):
    """取第 y0:y1 行条带的逐瓦片清晰度场（最近邻放大到条带大小）"""
    if C_tiles is None:
        return C
    k = int(C_tiles.shape[1])
    ys = (np.arange(y0, y1) * k // max(int(h), 1)).astype(np.int64)
    xs = (np.arange(w) * k // max(int(w), 1)).astype(np.int64)
    return np.asarray(C_tiles[:, ys][:, :, xs], np.float64)


def tile_fwhm_map(vals, stars, h: int, w: int, k: int) -> np.ndarray:
    """把逐星 FWHM 分到 k×k 瓦片，返回 (k,k) 中位图（瓦片内 <2 颗星则 NaN）

    为什么按瓦片：清晰度权重原来用**全帧一个** FWHM（C_i），于是视宁最好的那几帧
    在整幅图上都被放大——好天区被推到 2.7px、边缘仍 3.7px，PSF 空间分布被拉开到
    1.37×（WBPP 1.15×）。SXT 这类单一 PSF 假设的工具就会在锐区留残差、把紧凑
    小星系判成恒星。按瓦片各测各的，每个区域自己挑最锐的帧，锐度与均匀性可以兼得。
    """
    out = np.full((k, k), np.nan, np.float64)
    v = np.asarray(vals, np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return out
    xs = np.array([st[0] for st in stars], np.int64)[ok]
    ys = np.array([st[1] for st in stars], np.int64)[ok]
    vv = v[ok]
    ix = np.clip(ys * k // max(int(h), 1), 0, k - 1)
    jx = np.clip(xs * k // max(int(w), 1), 0, k - 1)
    for a in range(k):
        for b in range(k):
            m = (ix == a) & (jx == b)
            if int(m.sum()) >= 2:
                out[a, b] = float(np.median(vv[m]))
    return out


# ---------------------------------------------------------------------------
# 拖线长度：逐帧二阶矩的长轴方差（比 circular fit_fwhm 更直接地反映"拉长")
# ---------------------------------------------------------------------------

def _trail_each(kut, stars: List[Tuple[int, int]], r_ap: float, half: int):
    """逐星二阶矩 → 帧级长轴方差 λ1 与长轴方向（返回 (λ1 中位, PA)）

    为什么用二阶矩而不是拟合：拖线是**方向性**的展宽，圆对称拟合只能给出一个
    等效变宽的 FWHM；二阶矩能把它拆成 λ1（沿拖线）与 λ2（垂直拖线），
    再配合参考帧（最锐帧）的 λ1 就能反推拖线本身的长度。

    口径固定为 TRAIL_AP×σ_psf：逐帧用同一个口径，跨帧的 λ1 才可比。
    方向用 (mx−my, 2·mxy) 的矢量和——它就是 2×PA 的方向，避免角度折叠。
    """
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(xx, yy)
    ap = rr <= r_ap
    border = rr >= max(r_ap, half - 2)
    l1s: List[float] = []
    vx = vy = 0.0
    for xi, yi in stars:
        cut = kut(xi, yi)
        if cut is None:
            continue
        b = np.asarray(cut, dtype=np.float64)
        bg = float(np.median(b[border])) if bool(border.any()) else float(np.median(b))
        b = np.maximum(np.where(ap, b - bg, 0.0), 0.0)
        tot = float(b.sum())
        if tot <= 0:
            continue
        mx = float((b * xx ** 2).sum()) / tot
        my = float((b * yy ** 2).sum()) / tot
        mxy = float((b * xx * yy).sum()) / tot
        tr, det = mx + my, mx * my - mxy * mxy
        disc = max(tr * tr / 4.0 - det, 0.0)
        l1s.append(tr / 2.0 + float(np.sqrt(disc)))
        vx += mx - my
        vy += 2.0 * mxy
    if not l1s:
        return float('nan'), float('nan')
    return float(np.median(l1s)), float(0.5 * np.arctan2(vy, vx))


def measure_trail(img: np.ndarray, stars: List[Tuple[int, int]],
                  sigma_psf: float) -> Tuple[float, float]:
    """整帧版（内存后端起用）"""
    r_ap = TRAIL_AP * max(float(sigma_psf), 0.5)
    half = int(np.ceil(r_ap)) + 3
    h, w = img.shape

    def kut(xi: int, yi: int):
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            return None
        return img[yi - half:yi + half + 1, xi - half:xi + half + 1]

    return _trail_each(kut, stars, r_ap, half)


def measure_trail_frame(mm: np.ndarray, i: int, off: np.ndarray,
                        stars: List[Tuple[int, int]],
                        sigma_psf: float) -> Tuple[float, float]:
    """memmap 后端：只取星点小窗（与 measure_fwhm_frame 同思路）"""
    offi = np.float32(off[i])
    r_ap = TRAIL_AP * max(float(sigma_psf), 0.5)
    half = int(np.ceil(r_ap)) + 3
    _n, h, w = mm.shape

    def kut(xi: int, yi: int):
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            return None
        return mm[i, yi - half:yi + half + 1, xi - half:xi + half + 1] - offi

    return _trail_each(kut, stars, r_ap, half)


# ---------------------------------------------------------------------------
# 阶段 B：R_r（分区权重，逐像素连续场）
# ---------------------------------------------------------------------------

def _r_envelope(sref: np.ndarray, sigma_psf: float, env_win_mult: float = 2.0,
                env_eps: float = 3.0, env_p: float = 1.0,
                r_blur: float = 0.0, r_mode: str = 'max',
                r_win: float = 0.0, r_k: float = 3.0) -> np.ndarray:
    """R = **亮度归一包络**：E = (I−sky) / local_max(I−sky, 窗≈4σ_psf)

    包络规格由以下约束唯一确定 —— **R 必须在 ≲4σ_psf 内从 1 落到 0**：
        · 无亮度平台：过渡落在 2~4σ_psf 内则平台降幅回到基线，落到 5σ_psf 外就出现平台
        · 无噪声环：ν 要在 r_bright 前回到 1（r_bright = 4.6σ_psf / 4.1σ_psf）
        · 保留锐化：R 在核心必须真的到 1
      以"对比度的绝对阈值"定义 R 无法满足这一点：T ∝ 峰值/σ_psf ÷ 天空梯度，
      同一个阈值在不同目标上落到的 σ_psf 倍数会漂 → 换目标就得重标定，甚至锐化整个失效。

    E 的关键性质：对高斯星点，窗内极大就是峰值 I0，于是
        E(r) = I(r)/I0 = exp(−r²/2σ²)（r 小于半个窗时）
        更一般地 E(r) = I(r)/I(max(0, r−2σ)) = exp(−2r/σ + 2)
      **只含 r/σ_psf，与峰值亮度无关** → 亮星暗星得到同一条径向曲线 → 换目标不漂。
      数值上：核心 1.0、2σ_psf 处 0.135、3σ_psf 处 0.011 → 过渡在 3σ_psf 内完成 ✓
      空白天区：I−sky ≈ 噪声，local_max ≈ 3~4σ 的噪声极大 → E ≈ 0.2~0.3，
        再被 env_eps·σ 项压到接近 0 → 朦胧区严格吃满信噪比 ✓
      平滑亮星云：local_max ≈ 自身 → E ≈ 1 → 该区退化为"全局清晰度加权"（经典做法）。
        这不是缺陷：那里 R 空间上近乎常数 → 无 λ' → 不产生平台/环，代价只是 N_eff。

    参数全部无量纲：env_win_mult（窗 = 该倍数×2×σ_psf）、env_eps（天空抑制，以 σ 为单位）、
    env_p（形状幂）。
    """
    d = sref.astype(np.float64)
    sky = float(np.median(d))
    sig = 1.4826 * float(np.median(np.abs(d - sky)))
    dm = np.maximum(d - sky, 0.0)
    half = int(max(1, round(float(env_win_mult) * max(float(sigma_psf), 0.5))))
    win = 2 * half + 1
    _rb = float(r_blur or 0.0)
    if _rb > 0:
        # 先平滑再取局部极大值：星峰不再垄断窗口，星云自身的结构成为局部极大
        _dms = _gaussf(dm, _rb, 'nearest')
        _loc = _maxf(_dms, win, 'nearest')
        _E = np.clip(_dms / (_loc + float(env_eps) * max(sig, 1e-30)), 0.0, 1.0)
        if env_p and abs(float(env_p) - 1.0) > 1e-9:
            _E = np.power(_E, float(env_p))
        return _E, _loc, max(sig, 1e-30)
    if str(r_mode).lower() == 'ctn':
        # 对比度-噪声（绝对判据）：R = clip((bp − 0.8σ) / ((k−0.8)σ), 0, 1)
        # bp 为 r_win 窗的带通（局部均值扣除）；σ 是 bp 的全图稳健噪声。
        # 与 max/band 的区别：不再用"窗内极大"归一 ⇒ 暗弱纤维不会被旁边的星峰压成 0；
        # k 以"局部噪声倍数"给出，是跨目标可迁移的无量纲判据。
        w2 = int(round(float(r_win))) if float(r_win or 0) > 0 else int(max(9, round(4.0 * max(float(sigma_psf), 0.5))))
        w2 = max(3, w2 | 1)
        bp = dm - _boxf(dm, w2, 'nearest')
        bs = 1.4826 * float(np.median(np.abs(bp - np.median(bp))))
        bs = max(bs, 1e-30)
        lo = 0.8 * bs
        hi = max(float(r_k), 1.05) * bs
        E = np.clip((bp - lo) / max(hi - lo, 1e-30), 0.0, 1.0)
        loc = bp
        if env_p and abs(float(env_p) - 1.0) > 1e-9:
            E = np.power(E, float(env_p))
        return E, np.maximum(loc, 0.0), bs
    if str(r_mode).lower() == 'band':
        # 带通包络（ridge 型）：先减去 win 窗的局部均值，取正部再按窗口内极大归一
        #  · 星点：bp 只有正核（半径 ~1.5σ），负环被裁掉 → R 在 ~2σ 内落到 0，不产生平台
        #  · 纤维/细节：bp 在脊上最大、两侧衰减 → R 精确贴住"有结构"的像素
        #  · 平滑星云/天空：bp≈噪声 → R≈0，仍按信噪比加权
        w2 = int(round(float(r_win))) if float(r_win or 0) > 0 else int(max(9, round(4.0 * max(float(sigma_psf), 0.5))))
        w2 = max(3, w2 | 1)
        bp = dm - _boxf(dm, w2, 'nearest')
        dd = np.maximum(bp, 0.0)
        loc = _maxf(dd, w2, 'nearest')
        E = dd / (loc + float(env_eps) * max(sig, 1e-30))
        E = np.clip(E, 0.0, 1.0)
        if env_p and abs(float(env_p) - 1.0) > 1e-9:
            E = np.power(E, float(env_p))
        return E, loc, max(sig, 1e-30)
    loc = _maxf(dm, win, 'nearest')
    E = dm / (loc + float(env_eps) * max(sig, 1e-30))
    E = np.clip(E, 0.0, 1.0)
    if env_p and abs(float(env_p) - 1.0) > 1e-9:
        E = np.power(E, float(env_p))
    return E, loc, max(sig, 1e-30)


def _grow_support(R: np.ndarray, radius_px: float) -> np.ndarray:
    """把 R 的支撑沿径向向外平移 radius_px（形态学膨胀 + 高斯羽化）

    为什么是"平移"而不是"整体抬升"：R 的等值线（尤其是 0.5）当前贴在参考
    星点上；对高斯星点，用半宽 h 的极大值滤波后 R_new(r) = R(r−h)，0.5 等值线
    正好外移 h 像素，且径向曲线形状不变（不会把整片天空一起抬高）。
    h 之外立刻回到原值，故空白天区的信噪比不受影响；边界再用 σ=h/2 的高斯
    羽化，避免方形膨胀在过渡带留下台阶（星周「亮—暗—亮」环的来源之一）。

    radius_px ≤ 0 原样返回；参数在界面上以 σ_psf 为单位给出。
    """
    h = int(round(float(radius_px)))
    if h <= 0:
        return R
    d = np.maximum(_maxf(R, 2 * h + 1, 'nearest'), 0.0)
    d = _gaussf(d, max(0.8, h * 0.5), 'nearest')
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def _grow_support_adaptive(R: np.ndarray, grow_px: float, peak: np.ndarray,
                           sky_sig: float, levels: int = 4) -> np.ndarray:
    """按星点亮度自适应地外扩 R 的支撑（半径随局部峰值增大）

    为什么需要自适应：固定外扩把**所有**星点（连同噪声峰）的支撑一起推开，
    于是暗星周围也长出一圈"锐帧仅有的"晕——那圈比平均叠暗淡、噪声还更高，
    就是 g=3~4 时看到的星周低信噪/发暗区。物理上需要的只是：**把过渡圈推到
    "这颗星自己已经暗到看不见"的半径**，而那个半径随亮度增大。

    映射：t = clip( (log10(峰高/天空σ) − log10(LO)) / (log10(HI) − log10(LO)), 0, 1 )
        峰值 <LO(=20σ) → t=0；峰值 ≥HI(=1000σ) → t=1。
        外扩量 h = grow_px · (FLOOR + (1−FLOOR)·t)：
          · FLOOR 这部分是**保底**。拖线长度由风（赤道仪抖动）决定、与星点亮度无关，
            所以每颗星都要拿到足以盖住自身拖线的那部分；纯按亮度归零时实测中等亮星
            的伸长率从 0.10 退回 0.15，等于白做。
          · 亮度加成这部分对应"晕能被看到多远"：只有亮星的外晕才延伸到那么远，
            把过渡圈推到那里才不会在暗星周围留下一圈空转的锐帧区。
      用对数而不是线性：光晕影响半径随亮度只按 √log 增长，线性映射会把中等亮星
        过早推满。阈值以**参考像自身的天空 σ** 为单位，故与曝光/增益无关。

    实现：t 连续，但形态学膨胀只能取整半径 → 取 levels+1 个量化半径各做一次
    dilate+羽化，再按 t 在相邻两级之间线性混合。混合避免"级边界"在星周留下
    同心台阶。

    grow_px ≤ 0 原样返回。
    """
    g = float(grow_px)
    if g <= 0:
        return R
    b = np.maximum(peak, 1e-30) / max(float(sky_sig), 1e-30)
    lo, hi = math.log10(STAR_GROW_LO_SNR), math.log10(STAR_GROW_HI_SNR)
    t = np.clip((np.log10(b) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    f = float(np.clip(STAR_GROW_FLOOR, 0.0, 1.0))
    t = f + (1.0 - f) * t
    out = np.zeros(R.shape, dtype=np.float32)
    for k in range(levels + 1):
        w = np.clip(1.0 - np.abs(t * levels - k), 0.0, 1.0).astype(np.float32)
        if not bool(np.any(w > 0)):
            continue
        d = R if k == 0 else _grow_support(R, g * k / levels)
        out += d * w
    return np.clip(out, 0.0, 1.0)


class _HalfFS:
    """把帧集按 2×2 盒式降采样暴露出去（参考像在 1/2 网格重建用）"""

    def __init__(self, fs):
        self._fs = fs
        h, w = fs.shape
        self.shape = (int(h // 2), int(w // 2))

    def _dn(self, a):
        a = np.asarray(a, np.float32)
        h2, w2 = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        a = a[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
        return a.mean(axis=(1, 3))

    def frame(self, i):
        return self._dn(self._fs.frame(int(i)))

    def strip(self, i, j0, j1):
        return self._dn(self._fs.strip(int(i), int(j0) * 2, int(j1) * 2))



# ---------------------------------------------------------------------------
# 邻域原语：优先走 wgpu（R 场里 box/max/gauss 占了大头），失败静默回落 scipy
# ---------------------------------------------------------------------------
_GI = None


def _gpu_img():
    global _GI
    if _GI is None:
        try:
            if os.environ.get('DWT_RF_GPU', '1') == '0':
                _GI = False
            else:
                from . import gpu_wgpu_img as _m
                _GI = _m if _m.available() else False
        except Exception:
            _GI = False
    return _GI


def _maxf(a, size, mode='nearest'):
    gi = _gpu_img()
    if gi:
        try:
            return gi.max_filter(np.asarray(a, np.float32), max(0, int(size) // 2))
        except Exception:
            pass
    return ndimage.maximum_filter(a, size=size, mode=mode)


def _boxf(a, size, mode='nearest'):
    gi = _gpu_img()
    if gi:
        try:
            return gi.box_blur(np.asarray(a, np.float32), max(0, int(size) // 2))
        except Exception:
            pass
    return ndimage.uniform_filter(a, size=size, mode=mode)


def _gaussf(a, sigma, mode='nearest'):
    gi = _gpu_img()
    if gi:
        try:
            return gi.gauss_blur(np.asarray(a, np.float32), float(sigma))
        except Exception:
            pass
    return ndimage.gaussian_filter(a, sigma=sigma, mode=mode)


def build_r_field(sref: np.ndarray, sigma_psf: float,
                  win_mult: float = R_WIN_MULT, r_floor: float = R_FLOOR,
                  env_win_mult: float = ENV_WIN_MULT, env_eps: float = ENV_EPS,
                  env_p: float = ENV_P, star_grow: float = STAR_GROW,
                  r_blur: float = 0.0, r_mode: str = 'max',
                  r_win: float = 0.0, r_k: float = 3.0, dec: int = 1
                  ) -> Tuple[np.ndarray, Dict]:
    """分区权重 R ∈ [0,1]：此处是"细节型"还是"朦胧型"

    R 的构造分两处，均由数据自身确定：
      **映射** = 亮度归一包络（_r_envelope）：径向轮廓只含 r/σ_psf，换目标不漂。
      **判据** = 边缘强度 T = 局部 RMS(|∇Sref|) / 其全图中位，见下方实现。
        高斯星点的 |∇I| 在 r≈σ_psf 处达峰、r≈3σ_psf 降到峰值的 1%，支撑天然贴合星点本身
        （≈1.3×FWHM），噪声台阶因此落在星光里而不是背景上；
        无结构的亮星云 |∇I|≈0 → R≈0，仍符合"朦胧型只需信噪比"的定义。
        （带通 RMS 口径会把星点变成"正核+负环"，负环把支撑撑到星点尺寸两倍以上，
         台阶落到纯背景 → 星周可见噪声环，故不采用。）
      无量纲化：T 一律除以自己的全图中位（天空占多数像素 → 中位就是天空水平，
      T_sky ≡ 1），与目标的绝对亮度、曝光、增益、窄带/宽带全部无关；
      σ_psf 与 win 均由测得的 FWHM 得到 → 换目标不改数值也不改含义。
      T 现在只用于报表诊断（info 里的 T_p50…），不参与 R 的计算。
    """
    # 降采样计算开关：R 是平滑的支撑掩膜（过渡沿 ~σ_psf/2），在 1/dec 网格上算
    # 再双线性放大，边缘量化误差 <1 px，省 4~16× 时间（dec=2 时约 4×）。
    _H0, _W0 = int(sref.shape[0]), int(sref.shape[1])
    if dec and int(dec) > 1 and _H0 > 64 and _W0 > 64:
        _d = int(dec)
        sref = np.ascontiguousarray(sref[::_d, ::_d])
        sigma_psf = float(sigma_psf) / _d
        grow_src = True
    else:
        _d = 0
        grow_src = False
    win = int(max(3, round(float(sigma_psf) * win_mult)))
    win |= 1                                        # 奇数窗，中心对齐
    gy, gx = np.gradient(sref.astype(np.float64))
    g2 = gx * gx + gy * gy
    rms = np.sqrt(np.maximum(_boxf(g2, win, 'nearest'), 0.0))
    sig_hp = float(np.median(rms))
    T = rms / max(sig_hp, 1e-30)

    # 包络映射不经过 T：直接用亮度归一包络，径向轮廓只含 r/σ_psf
    R, loc, sky_sig = _r_envelope(sref, sigma_psf, env_win_mult, env_eps, env_p, r_blur,
                                  r_mode=r_mode, r_win=r_win, r_k=r_k)

    # 星点锐帧区外扩：半径随星点亮度自适应（亮星推到更暗处，暗星不动）
    grow_px = float(star_grow) * max(float(sigma_psf), 0.5)
    grow_mean = 0.0
    if star_grow and float(star_grow) > 0:
        R = _grow_support_adaptive(R, grow_px, loc, sky_sig)
        lo, hi = math.log10(STAR_GROW_LO_SNR), math.log10(STAR_GROW_HI_SNR)
        t = np.clip((np.log10(np.maximum(loc, 1e-30) / sky_sig) - lo) / max(hi - lo, 1e-6),
                    0.0, 1.0)
        t = STAR_GROW_FLOOR + (1.0 - STAR_GROW_FLOOR) * t
        # 报表用：锐帧区（R>0.5）里实际拿到的平均外扩量。抽样 1/16 像素即可
        sub = (slice(None, None, 4), slice(None, None, 4))
        core = R[sub] > 0.5
        grow_mean = float(t[sub][core].mean()) * grow_px if bool(core.any()) else 0.0
        del loc

    # 全局下限：R_eff = floor + (1−floor)·R。见 R_FLOOR 常量处的推导依据。
    if r_floor and r_floor > 0:
        f = min(float(r_floor), 0.95)
        R = f + (1.0 - f) * R
    if _d > 1:
        R = ndimage.zoom(R, _d, order=1)[: _H0, : _W0]
        R = np.clip(R, 0.0, 1.0)
    R = R.astype(np.float32)
    info = {
        'sigma_psf': float(sigma_psf), 'win': int(win), 'sig_hp': float(sig_hp),
        'r_mode': str(r_mode), 'r_win': float(r_win), 'r_k': float(r_k),
        'star_grow': float(star_grow), 'grow_px': float(grow_px),
        'grow_px_mean': float(grow_mean),
        'T_p50': float(np.median(T)), 'T_p90': float(np.percentile(T, 90)),
        'T_p99': float(np.percentile(T, 99)), 'T_max': float(T.max()),
        'R_mean': float(R.mean()),
        'R_gt50': float((R > 0.5).mean()), 'R_gt90': float((R > 0.9).mean()),
    }
    return R, info


# ---------------------------------------------------------------------------
# 阶段 C：融合权重 + 排异 + 唯一一次加权平均
# ---------------------------------------------------------------------------

def strip_weights(C: np.ndarray, S: np.ndarray, R: np.ndarray,
                  A: float, B: float, T=None, fp: float = 0.0,
                  beta: float = 0.0, cap: float = 0.0) -> np.ndarray:
    """W[i,r] = C_i^(A·R_r) · S_i^(B·(1−R_r))，并逐像素归一使 mean_i(W) = 1

    融合式的形状选择依据（式 6、式 7）：
        · 幂律乘法 → 两端行为干净可解释：R=1 只看清晰度，R=0 只看信噪比
        · 指数上线性插值（A·R 与 B·(1−R)）→ R 连续变化时 W 连续变化，无台阶、无拼接痕迹
        · B=2 不是调出来的：加权平均方差最小 ⇔ W ∝ 1/σ² ⇔ S²（S = σ_中位/σ_i）
        · A 是唯一需要标定的数：它决定"细节区愿意用多少信噪比换解析力"
    逐像素归一 mean_i(W)=1 保证加权平均不改变总曝光量（成品背景水平与
               等权叠加一致），也让"权重"只有相对意义、与帧数无关。
    **按条带调用**，不实体化全幅 (n,H,W) 权重场：166 帧全幅时那是
               166×244MB ≈ 40 GiB。条带内算完立刻用掉。
    """
    n = len(C)
    a = (A * R)[None, ...]
    # beta·R：细节区也保留一部分噪声惩罚（beta=0 即原行为：R=1 只看清晰度）
    b = (B * (1.0 - R) + float(beta) * R)[None, ...]
    _Cc = np.asarray(C, np.float64)
    if _Cc.ndim == 1:                      # 帧级标量（旧行为）
        lc = np.log(np.maximum(_Cc, 1e-12)).reshape(n, *([1] * R.ndim))
    else:                                  # 逐瓦片 (n, h, w)：直接对齐到条带
        lc = np.log(np.maximum(_Cc, 1e-12))
    ls = np.log(np.maximum(S, 1e-12)).reshape(n, *([1] * R.ndim))
    if T is not None and fp:
        lt = np.log(np.maximum(np.asarray(T, np.float64), 1e-12)).reshape(
            n, *([1] * R.ndim))
    else:
        lt = 0.0
    # 权重强度 γ：W^γ。γ=1 是原行为（完全按 C/S/T 加权），γ=0 退化为等权。
    # 帧质量相近时，逐帧权重估计本身带噪声，过度加权会降低有效帧数、反而抬高噪声；
    # 把它做成可调量，才谈得上"用多少权重"的实验，而不是写死。
    _gr = float(os.environ.get('DWT_WGAMMA', '') or 1.0)
    # 单帧份额上限：细节区的 C^A 会让某一帧独占绝大部分权重，此时该帧的
    # 缺陷（卫星线/宇宙线）不是"少数派"而是"多数派"，逐像素排异按定义抓不到。
    # 做法：把**总对数权重** ln(W) 截到 ±L，使最大/最小权重比 ≤ ((n-1)·cap/(1-cap))，
    # 从而任一帧的份额 ≤ cap（只截 A·lnC 不够：S/T 仍能把份额推上去）。
    # 注意 L 取**一半**的对数比：实测权重比 ≤ e^(2L)，故 2L = ln((n-1)cap/(1-cap))。
    # cap=0 关闭（保持旧行为）。
    _alc = a * lc
    _lw = _gr * (_alc + b * ls + fp * lt)
    if cap and 0.0 < cap < 1.0 and n > 1:
        _L = 0.5 * float(np.log(max((n - 1) * cap / (1.0 - cap), 1e-9)))
        _lw = np.clip(_lw, -_L, _L)
    W = np.exp(_lw)
    W /= np.maximum(W.mean(axis=0), 1e-30)[None]
    return W.astype(np.float32)


def stack_strip(D: np.ndarray, W: np.ndarray, R: Optional[np.ndarray] = None,
                sky: float = 0.0, rej_k: float = REJ_K,
                rej_max_frac: float = REJ_MAX_FRAC,
                frame_sig: Optional[np.ndarray] = None,
                lsp_acc: Optional[dict] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """排异 + 唯一一次加权平均：S = Σ W·D / Σ W

    核心思想：**豁免余量不靠任何经验常数或代理量，而是逐条带从数据里测出"真实分歧的上界"**：
        m = median_i  P99_{R>0.5 像素}( |D_i−a_r| / (a_r−sky) )
    即每帧在细节像素上"偏差/信号"的 P99（伪迹只占极少数像素，动不了 P99），再跨帧取
    中位数（个别帧上的伪迹/坏像素动不了中位数）。m 即该条带由数据给出的真实视宁度分歧上界
    ——天然包含视场梯度、混星、PSF 形状误差等一切真实效应，换数据自动跟随，零移植成本。
        阈值(r) = κ·1.4826·MAD(跨帧|偏差|) + m·R_r·(a_r−sky)     （a_r = 跨帧中位比较基准）
        · 天空/朦胧区 R≈0 → 纯 κσ clipping（κ 以 σ 为单位，MAD·1.4826=σ 估计）；
          中位数基准对多帧伪迹汇合免疫（个别帧污染一处不拉走基准）。
        · 细节区：偏差 ≤ m·信号 的是真实分歧（按构造豁免，星点零损伤）；
          超出的是伪迹（绝对量，偏差/信号 ≫ m）→ 被排——星核上的伪迹也可排。
        · 剔除比例上限：超额被剔数（扣除噪声期望误排 n·erfc(κ/√2)）> rej_max_frac×n
          → 该像素不排异（多数帧共有结构、配准黑边）；期望修正使上限与 κ 解耦。
    R 未传入时退化为单段 κ·MAD。
      排异掩膜只依赖各帧数值、不依赖权重 → 权重均匀的像素上与无权排异等价。
      被剔的 (帧,像素) 权重置 0，再归一，仍然是同一次加权平均。
    """
    n = D.shape[0]
    if rej_k > 0:
        anchor = np.median(D, axis=0)
        dev = np.abs(D - anchor[None])
        if frame_sig is not None:
            # 月光/薄云下帧间噪声可差 5 倍：把偏差换算到「各帧自己的 σ 单位」再判，
            # 否则 MAD 会被噪声大的帧撑大、干净的帧被误剔（这也是 moon 数据剔除率高的一因）
            _s = np.maximum(np.asarray(frame_sig, np.float32).reshape(n, 1, 1), 1e-12)
            dev = dev / _s
        mad = 1.4826 * np.median(dev, axis=0).astype(np.float32)
        if R is None:
            thr = rej_k * mad
        else:
            sig = np.maximum(anchor - sky, 0.0)
            core = (R > 0.5) & (sig > 0)
            if bool(core.any()):
                # 真实分歧上界：每帧细节像素 |偏差|/信号 的 P99，再跨帧取中位数
                ratio = np.percentile(dev[:, core] / sig[core][None], 99, axis=1)
                m = float(np.median(ratio))
            else:
                m = 0.0
            thr = rej_k * mad + (m * (R * sig) / _s[:, 0, 0].reshape(-1, 1, 1)
                                  if frame_sig is not None else m * (R * sig))
        bad = dev > thr   # thr 可能是 (h,w) 或逐帧 (n,h,w)，两者都能正确广播
        # ---- 天空区单侧压低伪迹（卫星线/月光斜纹）----
        # 卫星线在注册后的帧里常常落在同一批像素上、且只「加亮」。对称的中位/MAD
        # 会被它整体抬高，于是要么留线、要么把干净帧当离群剔掉。天空区（R<0.1）
        # 改用 25% 分位当基准、只剔上侧：干净帧一个不丢，线帧被整段去掉。
        if R is not None:
            _sky = R < 0.1
            if bool(_sky.any()):
                _d = D[:, _sky]
                if frame_sig is not None:
                    _d = _d / _s[:, 0, 0][:, None]
                _a = np.percentile(_d, 25, axis=0)
                _lo = _d <= np.median(_d, axis=0)[None]
                _sig = 1.4826 * np.median(np.abs(_d - _a[None]), axis=0)
                _sig = np.maximum(_sig, 1e-12)
                _hi = (_d - _a[None]) / _sig[None]
                _b = _hi > rej_k
                _nb = _b.sum(axis=0)
                _okb = _nb <= 0.8 * n
                _b &= _okb[None]
                bad[:, _sky] = _b
                del _d, _a, _lo, _sig, _hi, _b
        nbad = bad.sum(axis=0)
        expected = n * math.erfc(rej_k / 2.0 ** 0.5)   # 噪声期望误排数（κ 以 σ 为单位）
        _ok = (nbad - expected <= max(1.0, rej_max_frac * n))
        # 单侧豁免：卫星线/流星这类「只往上偏」的伪迹常出现在 >1/3 的帧里，
        # 会被上面的比例上限整片挡住（这就是"怎么设都排不掉"的原因）。
        # 判据：被剔样本几乎全在中位上侧（≥95%），且下侧几乎没有 → 判为单侧伪迹，
        # 允许剔到 70%；两侧都有明显离群时仍按原上限，避免误伤共有结构。
        if R is None:
            _hi = (dev > thr) & (D > anchor[None])
        else:
            _hi = (dev > thr) & (D > anchor[None])
        _nhi = _hi.sum(axis=0)
        _nlo = nbad - _nhi
        _one = (_nlo <= 0.05 * n) & (_nhi - expected <= 0.7 * n)
        bad &= (_ok | _one)[None] if _ok.ndim == 0 else (_ok | _one)
        if lsp_acc is not None:
            # 大尺度排异（LSPR）所需的「半阈值可疑样本」累加：只加一次原子/一次乘，
            # 事后若该像素落在细长结构里，就用代数把可疑部分整体减掉（无需第二遍读盘）
            _elev = dev > (_lsp_half * thr)
            lsp_acc['we'] += (W * _elev).sum(axis=0, dtype=np.float32)
            lsp_acc['wde'] += (W * D * _elev).sum(axis=0, dtype=np.float32)
            lsp_acc['cnte'] += _elev.sum(axis=0)
        rej_rate = float(bad.mean())
        W = np.where(bad, np.float32(0.0), W)
    else:
        rej_rate = 0.0
    sw = W.sum(axis=0, dtype=np.float32)
    safe = np.where(sw <= 0, np.float32(1e-30), sw)
    S = (W * D).sum(axis=0, dtype=np.float32) / safe
    sw2 = (W.astype(np.float64) ** 2).sum(axis=0)
    neff = np.where(sw > 0, (sw.astype(np.float64) ** 2) / np.maximum(sw2, 1e-30), 0.0)
    return S.astype(np.float32), neff.astype(np.float32), np.float32(rej_rate)


def _avail_bytes() -> int:
    """当前可用物理内存（拿不到就按 8 GiB 保守处理）

    DWT_AVAIL_MB 可覆盖（单位 MiB）：一是给用户「我这台机器请按更保守的预算跑」，
    二是让我们在大内存机器上模拟 8 GB / 4 GB 机器做兼容性回归。
    """
    import os as _os
    _ov = _os.environ.get('DWT_AVAIL_MB', '').strip()
    if _ov:
        try:
            _v = int(float(_ov) * 2 ** 20)
            if _v > 64 * 2 ** 20:
                return _v
        except Exception:
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 2 ** 30


def plan_strip(n_frames: int, width: int, frames_bytes: int,
               workers: int = 1, logf=log) -> int:
    """按"扣掉已驻留帧之后的可用内存"反推条带高度

    一条带内的临时量：W、D、dev、|dev| 各 (n,strip,w) float32/float64，
               外加 bad(bool) 与几个同尺寸中间量 → 按每元素 24 字节 × n 估（宁可估大）。
               帧数据本身已在内存（frames_bytes），必须先从可用量里扣掉，否则 166 帧全幅
               会把条带算得过大、直接耗尽内存。
    workers：并行时每个工作进程各持**一条**条带，故预算要按进程数均分，
               否则 N 条带同时在算 = N 倍内存。
    """
    avail = _avail_bytes()
    budget = max(1 * 2 ** 30, int(avail - frames_bytes) // 2)
    per_row = max(1.0, n_frames * width * 24.0)
    rows = int(budget // max(1, workers) // per_row)
    strip = max(1, min(rows, 512))
    logf(f'[并行] 帧已占 {frames_bytes / 2 ** 30:.2f} GiB，可用 {avail / 2 ** 30:.1f} GiB '
         f'→ 预算 {budget / 2 ** 30:.2f} GiB → {workers} 进程 × 条带 {strip} 行')
    return strip


# ---------------------------------------------------------------------------
# 并行执行：叠加按**条带**切，测星点按**帧**切，都是多进程
# ---------------------------------------------------------------------------
# 为什么这两处可以并行而不改变结果：两者的每一块都只看自己的输入，互不引用。
#   · 叠加   一条带只读那几行帧数据，结果写回 Stk/neff 互不重叠的行段；
#   · 测星点 第 i 帧的 FWHM 只由第 i 帧决定。
# 故并行只改"谁在算"，不改"算什么"，输出文件与串行版逐位一致。
# 顺序也必须保住：Stk/neff 的行段、fwhms 的下标、以及排异率与最大权重
#   这两个浮点累加，都按提交顺序落位/相加 —— 用 map（保序）而不是 as_completed。
#
# 帧数据不复制：工作进程由 initializer 各自**只读打开**同一个临时帧文件，
#   真正占内存的只是"一条带"的临时量。故进程数由内存预算反推（见 plan_strip、
#   plan_workers），而不是照抄 CPU 核数。
# 设 DWT_NO_PARALLEL=1 可强制串行（诊断用）


def _no_parallel() -> bool:
    """串行开关：**每次现读**环境变量 —— 需支持运行中途指定，故不做成常量"""
    return bool(os.environ.get('DWT_NO_PARALLEL'))


_MEM: Dict[str, object] = {}      # 工作进程内：帧后端访问，由 _worker_open* 建立


def plan_workers(per_worker_bytes: int, what: str = '', cap: int = CAP_WORKERS) -> int:
    """进程数 = min(核数−1, cap, 可用内存 ÷ 每进程需求)

    cap 按阶段给（见 CAP_* 常数）：内存型阶段（叠加）受内存约束、cap 小；
               算力型阶段（测星点，只取星点小窗）几乎不占内存，cap 可放到核数级别。
               内存不够就自动退回更少进程，退到 1 即串行。
    """
    avail = _avail_bytes()
    per = max(1, int(per_worker_bytes))
    nw = max(1, min((os.cpu_count() or 4) - 1, cap, int(avail // per)))
    if what:
        log(f'[并行] {what} {nw} 进程（可用 {avail / 2 ** 30:.1f} GiB，'
            f'每进程约 {per / 2 ** 20:.0f} MiB）')
    return nw


def _worker_open(path: str, n: int, h: int, w: int, off: np.ndarray) -> None:
    """只读打开临时帧文件（每个进程只开一次，之后各自切片）"""
    _MEM['mm'] = np.memmap(path, dtype=np.float32, mode='r', shape=(n, h, w))
    _MEM['off'] = off


def _task_fwhm_cfa(job) -> Tuple[int, float, int, int]:
    """真 CFA：子进程里拼 G 平面再量 FWHM —— 与串行的 fs.frame() 逐元素等价。

    串行版这段是 60 帧 × (整帧拆色 + 69 次二维高斯拟合)，实测 ~45 s；
    分到多进程后拟合按核数摊薄。**背景梯度必须一起带过来**：只减常数偏置会
    让星点附近的天空略微倾斜，C 会偏 2%（试过，FWHM 差 0.46 px），所以这里
    照 fs.frame() 的做法减掉 bggrad 的平面模型。
    """
    i, stars, pat, bgcoef, bgdeg, bgshape, bgrcoef = job
    n0, c0 = _FIT_STAT['n'], _FIT_STAT['conv']
    img = np.asarray(_MEM['mm'][i], np.float32) - np.float32(_MEM['off'][i])
    if bgcoef is not None and bgdeg > 0:
        try:
            from . import bggrad as _bgg
        except ImportError:                     # 子进程里没有包上下文
            import bggrad as _bgg
        m = _bgg.model_strip(bgcoef, bgdeg, 0, bgshape[0], bgshape[1],
                             bgshape[0], rcoef=bgrcoef)
        if m is not None:
            img -= np.asarray(m, np.float32)
    g = _cfa.plane(img, pat, 'G')
    v = measure_fwhm(g, stars)
    return int(i), v, 0.0, _FIT_STAT['n'] - n0, _FIT_STAT['conv'] - c0


def _task_fwhm(job) -> Tuple[int, float, int, int]:
    """一帧的星点 FWHM 中位（帧号一并返回，父进程按下标落位）

    只从帧后端取星点小窗，不物化整帧 —— 一帧 244 MiB 而小窗合计 ~0.5 MiB，
            这一步因此从"每帧重读一整帧"变成"几乎免费"。
    """
    i, stars, srch = job
    # 拟合计数发生在子进程里，父进程看不见 → 用增量带回（并发时父进程累加）
    n0, c0 = _FIT_STAT['n'], _FIT_STAT['conv']
    v, fx = measure_fwhm_frame(_MEM['mm'], i, _MEM['off'], stars, with_flux=True,
                                search=int(srch or 0))
    return int(i), v, fx, _FIT_STAT['n'] - n0, _FIT_STAT['conv'] - c0


def _scat(sh) -> float:
    """逐星残差中位；全 NaN（帧太少/星点全丢）时给 nan，别让 nanmedian 喊 All-NaN"""
    a = np.asarray(sh.get('scatter', []), np.float64)
    if a.size and np.isfinite(a).any():
        return float(np.nanmedian(a))
    return float('nan')


def _task_trail(job) -> Tuple[int, float, float]:
    """一帧的长轴方差中位与长轴方向（估拖线长度；同样只取星点小窗）"""
    i, stars, sigma_psf = job
    l1, pa = measure_trail_frame(_MEM['mm'], i, _MEM['off'], stars, sigma_psf)
    return int(i), l1, pa


def _task_strip(job):
    """一条带的权重 + 排异 + 加权平均（与串行的三段调用完全同序同式）"""
    y0, y1, Rr, C, S, A, B, rej_k, sky0, rej_gate, T, fp, beta, sig, wcap = job
    W = strip_weights(C, S, Rr, A, B, T=T, fp=fp, beta=float(beta), cap=float(wcap))
    mm, off = _MEM['mm'], _MEM['off']
    D = np.stack([mm[i, y0:y1] - np.float32(off[i]) for i in range(len(C))])
    s, ne, rr = stack_strip(D, W, Rr, sky0, rej_k=rej_k, rej_max_frac=rej_gate,
                            frame_sig=sig)
    return s, ne, rr, float(W.max(axis=0).mean())


def _worker_open_rw(path: str, n: int, h: int, w: int) -> None:
    """读帧阶段的工作进程：以 r+ 打开临时帧文件

    这里的进程要**写**帧数据。各进程映射的是同一个文件，
            映射共享同一份页缓存，故互不重叠地各写各的帧号是安全的；
            但**不在工作进程里 flush** —— 刷盘与关闭统一由父进程在全部读完后做。
    """
    _MEM['mmw'] = np.memmap(path, dtype=np.float32, mode='r+', shape=(n, h, w))


def _strip_rowfpn(d: np.ndarray, win: int = 101) -> np.ndarray:
    """去掉逐帧的行方向固定纹（原地）。

    行中位的高频部分（减去 ~win 行的中值趋势）≈ 传感器行固定纹：
      · 去趋势窗口要足够长，才不会把天光梯度/渐晕一起当缺陷减掉；
      · 列方向隔 4 取样，纯为省时间（行内像素够多，抽样不影响中位）。
    没装 scipy 时原样返回（不影响其它流程）。
    """
    try:
        from scipy.ndimage import median_filter as _mf
    except Exception:
        return d
    prof = np.median(d[:, ::4], axis=1).astype(np.float64)
    hp = prof - _mf(prof, size=int(win))
    d -= hp.astype(np.float32)[:, None]
    return d


def _noise_sigma(d: np.ndarray) -> float:
    """逐像素噪声（对背景梯度免疫）。

    用相邻行差分的稳健散布估计：大尺度渐变在 d[1:]-d[:-1] 里变成常数，
    被 median 吸收，只剩真实噪声。数值上与整帧 MAD 同量纲（σ）。
    """
    # 全幅 diff 会再开一份 99.5 MB 临时数组；噪声是逐像素统计量，1/4 抽样足够，
    # 而且能显著降低全幅 8 进程时的峰值内存（用户实测 B 通道 MemoryError 的来源之一）。
    _s = d[::2, ::2]
    g = np.diff(_s, axis=0)
    del _s
    if g.size == 0:
        return float(1.4826 * np.median(np.abs(d - np.median(d))))
    return float(1.4826 * np.median(np.abs(g - np.median(g))) / np.sqrt(2.0))


def _task_read_frame(job):
    """读一帧 → 裁剪 → float32 → 曝光归一 → 帧级天空中位与噪声尺度

    与串行循环里那一段逐字同式；结果写进自己那一格，标量与元数据字典回传由父进程按下标落位。
    """
    i, fname, exptime_ref, y0, x0, h, w, bg_deg, cfa_ch, cfa_pat, fw_stars, fw_guess = job
    with contextlib.redirect_stdout(io.StringIO()):
        data, meta = read_frame(Path(fname))
    d = np.ascontiguousarray(data[y0:y0 + h, x0:x0 + w]).astype(np.float32)
    del data
    d *= np.float32(exptime_ref / float(meta['exptime']))
    if cfa_ch and _cfa is not None:
        d = _cfa.plane(d, cfa_pat, cfa_ch)
    if _ROWFPN:
        _strip_rowfpn(d)
    # np.median 内部 flatten→partition 会整帧复制一份（全幅 99.5 MB，8 进程 ×4 = 3.2 GB），
    # 这是用户那台机器在 B 通道爆 MemoryError 的直接原因。天空中位数用 1/16 抽样足够
    # （误差 <0.1 ADU），代价 6 MB。
    _smp = d[::4, ::4]
    med = float(np.median(_smp))
    del _smp
    sig = _noise_sigma(d)
    bgm = bggrad.block_medians(d, bggrad.BG_BLOCK) if bg_deg else None
    # 顺手量本帧星点 FWHM（C 权重）：帧已经解压在内存里，比之后从 memmap
    #   一帧帧捞小窗再拟合便宜；每进程多 ~0.2s，与读盘等待重叠。
    fw = float('nan')
    if fw_stars:
        try:
            _v = measure_fwhm_each(d, fw_stars, fwhm_guess=fw_guess)
            _ok = np.isfinite(_v)
            if _ok.sum() >= 5:
                fw = float(np.median(_v[_ok]))
        except Exception:
            fw = float('nan')
    _MEM['mmw'][i] = d
    del d
    return int(i), med, sig, bgm, meta, fw


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

# 元数据透传的排除清单：成品自己写的关键字 + PI 逐帧测量关键字（只对单帧成立）
_META_SKIP_KW = {'TOTALEXP', 'NCOMBINE', 'STACKMODE'}
_META_SKIP_KW_PREFIX = ('PSF', 'NOISE')


def _aggregate_fits_keywords(metas: List[Dict]) -> List[Dict]:
    """跨帧聚合 FITS 关键字：全帧同值 → 透传；DATE-OBS → 最早、DATE-END → 最晚一帧结束；
    逐帧测量（PSF*/NOISE*，PI 口径只对单帧成立）与成品自身关键字不透传。"""
    n = len(metas)
    vals: Dict[str, List[str]] = {}
    order: List[str] = []
    for m in metas:
        for kw in m.get('fits_all', []):
            name = str(kw['name']).upper()
            # xisf_io 惯例：字符串值带单引号存储 → 比较与解析前剥离
            if name not in vals:
                vals[name] = []
                order.append(name)
            vals[name].append(str(kw['value']).strip("'\""))
    out: List[Dict] = []
    for name in order:
        v = vals[name]
        if len(v) != n or name in _META_SKIP_KW or name.startswith(_META_SKIP_KW_PREFIX):
            continue
        if name == 'DATE-OBS':
            from datetime import datetime, timedelta
            ts = sorted(datetime.fromisoformat(s) for s in v)
            out.append({'name': 'DATE-OBS', 'value': f"'{ts[0].isoformat()}'",
                        'comment': 'earliest frame start (UTC)'})
            out.append({'name': 'DATE-END',
                        'value': f"'{(ts[-1] + timedelta(seconds=float(np.median(
                            [float(m['exptime']) for m in metas])))).isoformat()}'",
                        'comment': 'last frame end (UTC)'})
            continue
        if len(set(v)) == 1:
            kw0 = next(k for k in metas[0]['fits_all'] if str(k['name']).upper() == name)
            out.append({'name': name, 'value': kw0['value'], 'comment': kw0.get('comment', '')})
    return out


def _aggregate_properties(metas: List[Dict]) -> List[Dict]:
    """跨帧聚合 XISF Property：全帧同值 → 原样透传（base64 向量等不变形）；
    数值型逐帧略异 → 取中位数；Observation:Time:Start/End → 最早/最晚。
    PCL:/PixInsight:（逐帧处理签名与测量，只对单帧成立）不透传。"""
    n = len(metas)
    by_id: Dict[str, List[Dict]] = {}
    for m in metas:
        for p in m.get('props_raw', []):
            by_id.setdefault(p['id'], []).append(p)
    raws: List[Dict] = []
    for pid, ps in by_id.items():
        if len(ps) != n or pid.startswith(('PCL:', 'PixInsight:')):
            continue
        same = len({(p['type'],
                     tuple(sorted((k, v) for k, v in p['attributes'].items() if k != 'id')),
                     p['text']) for p in ps}) == 1
        p0 = ps[0]
        if same:
            raws.append(p0)
            continue
        vt = p0['type']
        vv = [p['attributes'].get('value') for p in ps]
        if vt in ('Float32', 'Float64', 'Int8', 'Int16', 'Int32', 'Int64',
                  'UInt8', 'UInt16', 'UInt32', 'UInt64'):
            try:
                med = float(np.median([float(x) for x in vv]))
                raws.append({'id': pid, 'type': vt,
                             'attributes': {'id': pid, 'type': vt, 'value': repr(med)},
                             'text': None})
                continue
            except (TypeError, ValueError):
                pass
        if vt == 'TimePoint':
            pick = min(vv) if pid.endswith(':Start') else (max(vv) if pid.endswith(':End') else None)
            if pick is not None:
                raws.append({'id': pid, 'type': vt,
                             'attributes': {'id': pid, 'type': vt, 'value': pick},
                             'text': None})
            continue
        # 其余逐帧不同的（字符串等）：跳过
    return raws


def _pws_props(diag: Dict) -> List[Dict]:
    """引擎诊断量 → PWS: 命名空间标量 Property（成品自身的星点/噪声/参数读数）。
    String 类型走 text 内容，数值走 value 属性（与 XISF 1.0 规范一致）。"""
    items: List[Tuple[str, str, object]] = [('PWS:Software', 'String',
                                             'DWT · Partition-Weighted Stacking')]
    for k, t in (('A', 'Float64'), ('B', 'Float64'), ('RejK', 'Float64'),
                 ('RejGate', 'Float64'), ('ScaleOut', 'Float64'), ('RejRate', 'Float64'),
                 ('StarCount', 'Int32'), ('FwhmFrameMedian', 'Float64'),
                 ('FwhmOutput', 'Float64'), ('NeffMedian', 'Float64'),
                 ('SigmaFrameMedian', 'Float64'), ('SigmaSky', 'Float64')):
        if diag.get(k) is not None:
            items.append((f'PWS:{k}', t, diag[k]))
    out: List[Dict] = []
    for pid, t, v in items:
        if t == 'String':
            out.append({'id': pid, 'type': t, 'attributes': {'id': pid, 'type': t}, 'text': str(v)})
        else:
            out.append({'id': pid, 'type': t,
                        'attributes': {'id': pid, 'type': t, 'value': str(v)}, 'text': None})
    return out


def save_linear_xisf(path: Path, arr: np.ndarray, metas: List[Dict], mode_text: str,
                     diag: Optional[Dict] = None) -> None:
    """导出线性态成品：全程只有线性运算（曝光归一常数、量纲回乘常数、
    天空加性平移、Σ W·D/Σ W 加权平均），不做任何拉伸/归一/截断，
    导出前仅 float32 连续化。
    元数据：逐帧 FITS 关键字与 XISF Property 跨帧聚合透传（观测/仪器信息
      不因叠加丢失），并写入 PWS: 命名空间的成品诊断量（星点数、逐帧/成品
      FWHM、σ、参数、排异率、量纲回乘）。
    """
    from xisf_io import XISFImage, write_xisf
    a = np.ascontiguousarray(arr, dtype=np.float32)
    img = XISFImage()
    img.data = a
    img.geometry = (a.shape[1], a.shape[0], 1)      # XISF geometry = 宽:高:通道
    img.sample_format = 'Float32'
    img.color_space = 'Gray'
    # bounds 写真实数据范围：写死 0:1 会让 PixInsight 把 ADU 量级的成品
    # （天空几百、星点几万）当成「已归一化」的图像，超出 1 的全按饱和显示，
    # 一打开就是一片白。
    # 用 nanmin/nanmax 而不是 a[np.isfinite(a)]：后者会给全幅再复制一份（417 MB）。
    if np.isfinite(a).any():
        _b0 = float(np.nanmin(a)); _b1 = float(np.nanmax(a))
    else:
        _b0, _b1 = 0.0, 1.0
    img.bounds = (_b0, _b1) if _b1 > _b0 else (0.0, 1.0)
    img.pixel_storage = 'Planar'
    img.fits_keywords = [
        {'name': 'TOTALEXP', 'value': round(sum(m['exptime'] for m in metas), 1),
         'comment': 'total exposure seconds of combined frames'},
        {'name': 'NCOMBINE', 'value': len(metas), 'comment': 'frames combined'},
        {'name': 'STACKMODE', 'value': mode_text,
         'comment': 'partition-weighted stacking (linear output, no stretch)'},
    ] + _aggregate_fits_keywords(metas)
    prop_raws = _aggregate_properties(metas) + _pws_props(diag or {})
    if prop_raws:
        # 有 Property → 走 xisf_io 的原始元素回写路径（base64 向量等逐字节保真）
        img.original_property_info = prop_raws
    else:
        # 素材无任何 Property（纯 FITS 素材）→ PWS 诊断量走标量字典路径
        img.other_properties = {
            p['id']: (p['text'] if p['text'] is not None else p['attributes'].get('value'))
            for p in _pws_props(diag or {})
        }
    with contextlib.redirect_stdout(io.StringIO()):
        write_xisf(str(path), img, compression=None)
    log(f'[输出] 线性态 XISF {path.name}  {a.shape[1]}×{a.shape[0]}  '
        f'值域 {float(np.nanmin(a)):.3e}~{float(np.nanmax(a)):.3e}  '
        f'元数据：关键字 {len(img.fits_keywords)}  Property {len(prop_raws)}')


def _ca_shift_axis(a: np.ndarray, s: float, axis: int) -> np.ndarray:
    """沿 axis 做 Lanczos-3 亚像素平移（s 可为负）。"""
    if abs(s) < 1e-3:
        return a
    n = 6
    i0 = int(np.floor(s)) - n // 2 + 1
    taps = np.arange(i0, i0 + n)
    w = np.sinc(taps - s) * np.sinc((taps - s) / 3.0)
    w = w / w.sum()
    out = np.zeros_like(a, dtype=np.float32)
    for k, wt in zip(taps, w):
        if wt == 0.0:
            continue
        if axis == 0:
            src = np.roll(a, -int(k), axis=0)
            if k > 0:
                src[:int(k)] = src[int(k)]
            elif k < 0:
                src[int(k):] = src[int(k) - 1]
        else:
            src = np.roll(a, -int(k), axis=1)
            if k > 0:
                src[:, :int(k)] = src[:, int(k):int(k) + 1]
            elif k < 0:
                src[:, int(k):] = src[:, int(k) - 1:int(k)]
        out += wt * src
    return out


def _ca_estimate(G: np.ndarray, R: np.ndarray, B: np.ndarray, log, n_star: int = 60):
    """用亮星重心估 (R−G)、(B−G) 的亚像素偏移，返回 ((dyR,dxR),(dyB,dxB))。"""
    try:
        from scipy import ndimage
    except Exception:
        return (0.0, 0.0), (0.0, 0.0)
    h, w = G.shape
    v = G[np.isfinite(G)]
    thr = float(np.median(v) + 30.0 * np.std(v))
    sm = _boxf(np.nan_to_num(G), 3, 'nearest')
    pk = (sm == _maxf(sm, 9, 'nearest')) & (G > thr)
    ys, xs = np.nonzero(pk)
    sel = (ys > 40) & (ys < h - 40) & (xs > 40) & (xs < w - 40)
    ys, xs = ys[sel], xs[sel]
    if len(ys) > n_star:
        step = max(1, len(ys) // n_star)
        ys, xs = ys[::step][:n_star], xs[::step][:n_star]

    def ctr(img, y, x, r=6):
        win = img[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
        win = win - np.median(win)
        np.clip(win, 0, None, out=win)
        s = win.sum()
        if not np.isfinite(s) or s <= 0:
            return None
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return y + (win * yy).sum() / s, x + (win * xx).sum() / s

    dr, db = [], []
    for y, x in zip(ys, xs):
        cg = ctr(G, y, x)
        cr = ctr(R, y, x)
        cb = ctr(B, y, x)
        if cg is None or cr is None or cb is None:
            continue
        if abs(cr[1] - x) > 3 or abs(cb[1] - x) > 3:
            continue
        dr.append((cr[0] - cg[0], cr[1] - cg[1]))
        db.append((cb[0] - cg[0], cb[1] - cg[1]))
    if len(dr) < 8:
        log('[CA] 星点太少（%d），跳过通道对齐' % len(dr))
        return (0.0, 0.0), (0.0, 0.0)
    dr = np.array(dr); db = np.array(db)
    oR = (float(np.median(dr[:, 0])), float(np.median(dr[:, 1])))
    oB = (float(np.median(db[:, 0])), float(np.median(db[:, 1])))
    log('[CA] 通道偏移（相对 G）：R dy=%+.3f dx=%+.3f  B dy=%+.3f dx=%+.3f px（%d 颗星）'
        % (oR[0], oR[1], oB[0], oB[1], len(dr)))
    return oR, oB


def _find_cfa_ref(photos: Path) -> Optional[Path]:
    """自动找 WBPP 的 debayered 目录：素材目录的兄弟（或它自己的子目录）。"""
    cands = []
    for base in (photos, photos.parent, photos.parent.parent):
        for name in ('debayered', 'Debayered', 'debayered_delete'):
            c = base / name
            if c.is_dir():
                cands.append(c)
        # 也允许 debayered 下再套一层滤镜目录
        if (base / 'debayered').is_dir():
            for sub in (base / 'debayered').iterdir():
                if sub.is_dir():
                    cands.append(sub)
    for c in cands:
        try:
            hit = next(c.rglob('*_c_d_R.xisf'), None)
        except Exception:
            hit = None
        if hit is not None:
            return c
    return None


def _ca_from_fs(fs, pattern: str, log, n_star: int = 50):
    """在帧组首帧上估 (R−G)、(B−G) 偏移：判断这套系统有没有明显色差。"""
    try:
        _src = np.asarray(fs.frame(0), np.float32)
        R = _cfa.plane(_src, pattern, 'R')
        G = _cfa.plane(_src, pattern, 'G')
        B = _cfa.plane(_src, pattern, 'B')
    except Exception as _e:
        log('[CA] 估色差失败（%s: %s），按无明显色差处理' % (type(_e).__name__, _e))
        return (0.0, 0.0), (0.0, 0.0)
    return _ca_estimate(G, R, B, log, n_star=n_star)



def _cfa_weights_auto(frames, pattern: str, log, k: int = 6, n_star: int = 40):
    """在前 k 帧上分别量 R/G/B 的逐帧 FWHM，返回 (相关最小值, {ch: fwhm})."""
    try:
        from . import cfa as _c
    except Exception:
        try:
            import cfa as _c
        except Exception:
            return None, {}
    n = len(frames)
    if n < 4:
        return None, {}
    idx = sorted(set(int(round(i * (n - 1) / float(k - 1))) for i in range(k))) if k > 1 else [0]
    try:
        from DWT_DetailCore import read_frame  # noqa: E402
    except Exception as _e:
        log('[CFA] 权重 auto：取不到 read_frame（%s），跳过探测' % _e)
        return None, {}
    stars = None
    out = {}
    for ch in ('R', 'G', 'B'):
        vals = []
        for i in idx:
            try:
                _d = np.asarray(read_frame(frames[i])[0], np.float32)
                _p = _c.plane(_d, pattern, ch)
            except Exception:
                continue
            if stars is None:
                stars = build_star_table(_p, n_star=max(24, int(n_star)))
                if len(stars) < 8:
                    return None, {}
            vals.append(float(measure_fwhm(_p, stars[:n_star] if len(stars) > n_star else stars)))
        if vals:
            out[ch] = np.asarray(vals, np.float64)
    if not all(c in out for c in ('R', 'G', 'B')):
        return None, out
    m = min(len(v) for v in out.values())
    if m < 3:
        return None, out
    cr = float(np.corrcoef(out['R'][:m], out['G'][:m])[0, 1])
    cb = float(np.corrcoef(out['B'][:m], out['G'][:m])[0, 1])
    return min(cr, cb), out



def _ca_from_ref(ref: Path, log, n_frames: int = 5):
    """用参考目录里的 R/G/B 帧（各取前 n 帧平均）估通道偏移。"""
    import glob
    try:
        from xisf_io import read_xisf
    except Exception:
        return None
    out = {}
    for ch in ('R', 'G', 'B'):
        fs = sorted(glob.glob(str(ref / '**' / ('*_c_d_%s.xisf' % ch)), recursive=True))
        if not fs:
            return None
        acc = None
        for f in fs[:n_frames]:
            try:
                d = np.asarray(read_xisf(f)[0].data, np.float32)
            except Exception:
                continue
            if d.ndim == 3:
                d = d[:, :, 0]
            acc = d if acc is None else acc + d
        if acc is None:
            return None
        out[ch] = acc / max(1.0, float(len(fs[:n_frames])))
    oR, oB = _ca_estimate(out['G'], out['R'], out['B'], lambda *a: None)
    log('[参考] 用 WBPP debayered 帧估色差：R dy=%+.3f dx=%+.3f  B dy=%+.3f dx=%+.3f'
        % (oR[0], oR[1], oB[0], oB[1]))
    return oR, oB


class _SkipMerge(Exception):
    pass


def _merge_cfa_rgb(out_dir, stem: str, log) -> None:
    """CFA 三通道成品合成一张 3 通道 XISF（H×W×3，Planar）。

    三个单通道文件照旧保留：需要单独调某个通道（或 LRGB 里当 L）时直接拿它们。
    """
    from xisf_io import XISFImage, read_xisf, write_xisf
    planes = []
    src = Path(out_dir)
    for ch in ('R', 'G', 'B'):
        f = src / f'{stem}_{ch}.xisf'
        if not f.exists():
            log(f'[CFA] 缺 {f.name}，跳过 RGB 合成')
            return
        with contextlib.redirect_stdout(io.StringIO()):
            im = read_xisf(str(f))[0]
        d = np.asarray(im.data, dtype=np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        planes.append(d)
    # ---- 通道间亚像素对齐（色差校正）----
    if (not getattr(_merge_cfa_rgb, '_skip_ca', False)) and             str(os.environ.get('DWT_CFA_CA', '1')).lower() not in ('0', 'off', 'false', 'no') and len(planes) == 3:
        try:
            # 以「成品上量的偏移」为准：参考目录的帧是未叠加的，量不到堆叠引入的
            # 那一部分（实测 R 在成品上还多出 +0.38px 的 x 偏移）。参考目录只作对照。
            _oR, _oB = _ca_estimate(planes[1], planes[0], planes[2], log)
            _rdir = getattr(_merge_cfa_rgb, '_ref_dir', None)
            if _rdir is not None:
                _got = _ca_from_ref(_rdir, log)
                if _got:
                    log('[参考] 参考帧偏移 R dy=%+.3f dx=%+.3f；与成品量到的差 '
                        '(dy=%+.3f, dx=%+.3f) 视为堆叠引入' %
                        (_got[0][0], _got[0][1], _oR[0] - _got[0][0], _oR[1] - _got[0][1]))
            _okR = np.isfinite(np.asarray(_oR, np.float64)).all()
            _okB = np.isfinite(np.asarray(_oB, np.float64)).all()
            if not (_okR and _okB):
                # 星点太少时 _ca_estimate 会给 NaN；拿 NaN 去做插值平移会把整层
                # 变成 NaN（x43 那次就是 R/B 全空、只剩 G）。这里直接跳过。
                log('[CA] 偏移估计无效（R %s / B %s），跳过平移' % (_oR, _oB))
            else:
                if max(abs(_oR[0]), abs(_oR[1])) > 0.05:
                    planes[0] = _ca_shift_axis(_ca_shift_axis(planes[0], _oR[0], 0), _oR[1], 1)
                if max(abs(_oB[0]), abs(_oB[1])) > 0.05:
                    planes[2] = _ca_shift_axis(_ca_shift_axis(planes[2], _oB[0], 0), _oB[1], 1)
                log('[CA] 已按偏移平移 R/B（Lanczos-3 亚像素）')
        except Exception as _e:
            log('[CA] 通道对齐失败（跳过）：%s' % _e)
    # 通道平衡：三个通道是各自归一化叠加出来的，天空水平天然不一致，
    # 直接合并就是一片惨绿/惨红。按天空（25 百分位，避开星点）把 R/B 拉到 G，
    # 相当于 PI 里的 BackgroundNeutralization；真正的色彩还原仍交给用户的 CC。
    # 用 nanpercentile：单通道成品里只要有一个 NaN 像素（排异把某像素的样本
    # 全剔掉就会出现），np.percentile/median 就会整体返回 NaN，接着
    # R×NaN 把整层 R/B 抹成 NaN（x43 那次三通道只剩 G，就是这个）。
    _sky = [float(np.nanpercentile(p, 25)) for p in planes]
    _bal = str(getattr(_merge_cfa_rgb, '_balance', 'off') or 'off').lower()
    if _bal == 'mul' and all(np.isfinite(_sky)) and _sky[1] > 1e-12:
        # 乘性平衡：把 R/B 的天空拉到 G。注意这会同时放大星点——
        # 若三个通道的饱和星本就都在 1.0 附近，放大后 R/B 会超过 G，
        # 不同半径先后被显示裁切，星点就出现彩色边（logtest9 那次）。
        _fac = [_sky[1] / max(_sky[i], 1e-12) if i != 1 else 1.0 for i in range(3)]
        planes = [np.asarray(p * _fac[i], dtype=np.float32) for i, p in enumerate(planes)]
        log(f'[CFA] 通道平衡（乘性，天空对齐 G）：R×{_fac[0]:.3f}  B×{_fac[2]:.3f}'
            f'（原始天空 R {_sky[0]:.3e} G {_sky[1]:.3e} B {_sky[2]:.3e}）')
    elif _bal == 'add' and all(np.isfinite(_sky)):
        _g = _sky[1]
        planes = [np.asarray(p - _sky[i] + _g, dtype=np.float32) for i, p in enumerate(planes)]
        log(f'[CFA] 通道平衡（加性，天空对齐 G）：R{_g - _sky[0]:+.3e}  B{_g - _sky[2]:+.3e}'
            f'（原始天空 R {_sky[0]:.3e} G {_sky[1]:.3e} B {_sky[2]:.3e}）')
    else:
        log(f'[CFA] 通道平衡关闭（保持各通道原始亮度，与 WBPP 的 ChannelCombination 一致；'
            f'天空 R {_sky[0]:.3e} G {_sky[1]:.3e} B {_sky[2]:.3e}）')
    # 共同有效区：真 CFA 下 R/B 的覆盖天生比 G 稀，边界上容易「G 有、B 没有」，
    # 直接合并就是一条绿边。三个通道的 _ncov.npy 都在，取交集，谁没覆盖到
    # 就三个通道一起置 NaN（后续自动裁边会把它裁掉）。
    try:
        _covs = []
        for ch in ('R', 'G', 'B'):
            _cf = src / f'{stem}_{ch}_ncov.npy'
            if _cf.exists():
                _covs.append(np.load(str(_cf), mmap_mode='r'))
        if len(_covs) == 3:
            _shp = [(int(c.shape[0]), int(c.shape[1])) for c in _covs]
            if len(set(_shp)) == 1:
                _h0 = min(p.shape[0] for p in planes)
                _w0 = min(p.shape[1] for p in planes)
                _ok = np.ones((_h0, _w0), bool)
                for _ci, _c in enumerate(_covs):
                    _v = np.asarray(_c[:_h0, :_w0], np.float32)
                    _cname = 'RGB'[_ci]
                    # 相对阈值：每通道覆盖率不同（G≈2×R/B），用「本通道中位的一半」
                    # 当有效线，边缘的薄覆盖自然被剔掉，不会留下单通道色边。
                    # 只用「几乎没有任何样本」判定无效：0.5 = 半个样本。
                    # 曾经用过「本通道 ncov 中位的一半」当阈值，结果星点核心区
                    # （饱和 + 排届让 R/B 的 ncov 掉到 1~15，而中位 13 → 阈值 6.6）
                    # 整片被判成无数据，合成 RGB 的核心就成了黑洞。
                    _th = 0.5
                    _ok &= np.isfinite(_v) & (_v >= _th)
                    _med = float(np.nanmedian(_v)) if _v.size else 0.0
                    log('[CFA] %s 覆盖率：中位 %.2f，P1 %.2f，低于 %.1f 的像素 %.3f%%'
                        % (_cname, _med, float(np.nanpercentile(_v, 1)), _th,
                           100.0 * float((~np.isfinite(_v) | (_v < _th)).mean())))
                if not _ok.all():
                    _bad = int((~_ok).sum())
                    planes = [np.where(_ok, p[:_h0, :_w0], np.nan).astype(np.float32)
                              for p in planes]
                    log(f'[CFA] 三通道共同覆盖：裁掉 {_bad} 个「某通道无数据」的像素'
                        f'（{100.0 * _bad / _ok.size:.2f}%）')
    except Exception as _em:
        log(f'[CFA] 共同覆盖区计算失败（{type(_em).__name__}: {_em}），按原样合并')
    _shapes = sorted({(int(p.shape[0]), int(p.shape[1])) for p in planes})
    if len(_shapes) > 1:
        log(f'[CFA] 注意：三个通道画布不一致 {_shapes}，按最小画布合并且不做平移；'
            '正常情况下不应该出现（三通道共用几何后必然一致）')
    h = min(p.shape[0] for p in planes)
    w = min(p.shape[1] for p in planes)
    a = np.stack([np.ascontiguousarray(p[:h, :w], dtype=np.float32) for p in planes], axis=-1)
    # ---- 高光保护：任一通道接近饱和时，把该点三通道拉平（保持星点白色）----
    # 三通道各自归一化后再叠加，亮星的「饱和半径」不同：R/B 可能先顶到 1.0，
    # 合成后核心就成了彩色（红/蓝边）。这里把接近饱和的像素取三通道均值，
    # 只影响少数核心像素（实测 0.002% 量级），让星点核心保持中性。
    try:
        _hot = np.nanmax(a, axis=2) >= 0.985
        _n_hot = int(_hot.sum())
        if _n_hot:
            _m = np.nanmean(a, axis=2)
            for _c in range(3):
                a[..., _c] = np.where(_hot, _m, a[..., _c])
            log('[CFA] 高光保护：%d 个像素（%.4f%%）三通道已接近饱和，取均值保持星点中性色'
                % (_n_hot, 100.0 * _n_hot / float(_hot.size)))
    except Exception:
        pass
    # ---- 零值孔洞填补：极少像素没有任何样本（drizzle 落在像素缝里）----
    try:
        _zero = (a == 0.0).all(axis=2) & np.isfinite(a).all(axis=2)
        _nz = int(_zero.sum())
        if _nz:
            from scipy.ndimage import distance_transform_edt as _edt
            _idx = _edt(_zero, return_distances=False, return_indices=True)
            a = a[tuple(_idx)]
            log('[CFA] 补洞：%d 个无样本像素（%.5f%%）用最近邻填充' % (_nz, 100.0 * _nz / float(_zero.size)))
    except Exception as _ez:
        log('[CFA] 补洞跳过（%s）' % _ez)
    img = XISFImage()
    img.data = a
    img.geometry = (w, h, 3)                 # XISF geometry = 宽:高:通道
    img.sample_format = 'Float32'
    img.color_space = 'RGB'
    # bounds 写真实数据范围：写死 0:1 会让 PixInsight 把 ADU 量级的成品
    # （天空几百、星点几万）当成「已归一化」的图像，超出 1 的全按饱和显示，
    # 一打开就是一片白。
    _bv = a[np.isfinite(a)]
    if _bv.size:
        _b0, _b1 = float(_bv.min()), float(_bv.max())
    else:
        _b0, _b1 = 0.0, 1.0
    img.bounds = (_b0, _b1) if _b1 > _b0 else (0.0, 1.0)
    # write_xisf 直接写 data.tobytes()（行主序 HWC 交错）。这里若标 Planar，
    # 读方会把交错字节当三段连续平面解析，三通道全错位——PI 打开就是"坏图"。
    img.pixel_storage = 'Normal'
    out = src / f'{stem}_RGB.xisf'
    with contextlib.redirect_stdout(io.StringIO()):
        write_xisf(str(out), img, compression=None)
    log(f'[CFA] 合成彩色 {out.name}  {w}×{h}×3（三个单通道文件也保留）')
    # ---- 自动预览：和「重新合成 RGB」用同一套定标（中位 ± MAD、asinh）----
    if str(os.environ.get('DWT_RGB_PREVIEW', '1')).strip().lower() not in (
            '0', 'off', 'false', 'no'):
        try:
            from maps_page import _to_png
            _lo = np.zeros(3, np.float32)
            _hi = np.zeros(3, np.float32)
            for _c in range(3):
                _v = a[..., _c]
                _m = float(np.nanmedian(_v))
                _d = float(np.nanmedian(np.abs(_v - _m))) * 1.4826
                _lo[_c] = max(0.0, _m - 2.8 * _d)
                _hi[_c] = _m + 60.0 * _d
            _x = np.clip((a - _lo) / np.maximum(_hi - _lo, 1e-12), 0.0, 1.0)
            _x = np.arcsinh(_x * 18.0) / np.arcsinh(18.0)
            _u8 = np.nan_to_num(_x * 255.0).astype(np.uint8)
            _sg = max(1, int(np.ceil(max(h, w) / 1200.0)))
            _u8 = _u8[::_sg, ::_sg]
            _pv = src / f'{stem}_RGB_preview.png'
            _pv.write_bytes(_to_png(_u8))
            log('[CFA] 预览图 %s  %d×%d（抽样 %d）'
                % (_pv.name, _u8.shape[1], _u8.shape[0], _sg))
        except Exception as _ep:
            log('[CFA] 预览图生成失败（不影响成品）：%s' % _ep)


def _clean_stale_frames(max_age_h: float = 2.0) -> int:
    """删掉 core/_DWT_frames 里超过 max_age_h 小时的旧文件（崩溃残留）。

    正常结束时 FrameSet.close() 会自己清理；但中途被杀掉或断电就会留下
    几十 GB 的 .dat，跑几次就把盘塞满，所以启动时兜一次底。
    """
    import time
    try:
        bases = [Path(__file__).resolve().parent / "_DWT_frames"]
        try:
            for ln in _frames_state_file().read_text(encoding='utf-8').splitlines():
                ln = ln.strip()
                if ln:
                    bases.append(Path(ln))
        except Exception:
            pass
        base = bases[0]
        cutoff = time.time() - max_age_h * 3600.0
        n = 0
        for base in bases:
            if not base.is_dir():
                continue
            for f in base.iterdir():
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink()
                        n += 1
                except Exception:
                    continue
        try:
            if not any(base.iterdir()):
                base.rmdir()
        except Exception:
            pass
        return n
    except Exception:
        return 0


def _detail_beta(p) -> float:
    """满月帧补偿（--detail-snr）只在真 CFA 时生效，黑白流程必须为零。"""
    if not _CFA.get("pattern"):
        return 0.0
    return float(getattr(p, "detail_snr", 0.0) or 0.0)


def _clear_cfa_env() -> None:
    """离开彩机流程时清掉可能残留的环境变量（GUI 连跑多组时尤其重要）。"""
    for _k in ("DWT_DETAIL_SNR", "DWT_CFA_CHANNELS", "DWT_CFA_K", "DWT_CFA_RAW"):
        os.environ.pop(_k, None)


def _check_disk_space(frames, out_dir, scale: float, log,
                      n_ch: int = 1, px: float = 0.0,
                      n_frames: int = 0, rgb: bool = False) -> None:
    """跑之前先看盘，按「实际启用通道数」估。

    组成：memmap 帧（n×单帧）+ 每个输出平面一张 float32 画布
    （单通道文件 × n_ch + 彩机 RGB 合成 3 张 + 约 20% 辅助数组）。
    """
    try:
        import shutil
        f0 = frames[0]
        if isinstance(f0, (tuple, list)):
            f0 = f0[0]
        one = float(Path(str(f0)).stat().st_size)
        n = int(n_frames) if n_frames else len(frames)
        if px <= 0:
            px = max(1.0, one / 4.0)          # 估不到就按 float32 反推
        s2 = max(1.0, float(scale)) ** 2
        plane = px * s2 * 4.0                 # 一张输出平面（float32）
        n_planes = max(1, int(n_ch)) + (3 if rgb else 0)
        need = n * one + n_planes * plane * 1.2
        free = float(shutil.disk_usage(str(out_dir)).free)
        if free < need:
            raise SystemExit(
                "磁盘空间不足：%s 还剩 %.1f GB，这次大约需要 %.1f GB（%d 帧、%d 个平面）。"
                % (out_dir, free / 2 ** 30, need / 2 ** 30, n, n_planes)
                + chr(10) + "请清理后再跑，或把输出目录换到空间更大的盘。")
        log("[磁盘] %s 可用 %.1f GB，预计需要 %.1f GB（%d 帧、%d 个平面）"
            % (out_dir, free / 2 ** 30, need / 2 ** 30, n, n_planes))
    except SystemExit:
        raise
    except Exception as _e:
        log("[磁盘] 空间检查跳过（%s: %s）" % (type(_e).__name__, _e))


_ENV_LOGGED = False


def _log_env_once() -> None:
    """开机一行环境摘要：CPU 核数、可用内存、GPU 后端可用性。

    兼容性工单第一步就要这三样；打在日志开头比让用户去查系统信息快得多。
    """
    global _ENV_LOGGED
    if _ENV_LOGGED:
        return
    _ENV_LOGGED = True
    try:
        import os
        ncpu = os.cpu_count() or 0
        av = _avail_bytes() / 2 ** 30
        bits = [f'[环境] CPU {ncpu} 核', f'可用内存 {av:.1f} GiB']
        def _mod(name):
            """包内/脚本两种导入方式都要能拿到（CLI 常以顶层模块加载）。"""
            try:
                return __import__(f'{__package__}.{name}' if __package__ else name,
                                  fromlist=[name])
            except Exception:
                return __import__(name, fromlist=[name])

        try:
            gpu_wgpu = _mod('gpu_wgpu')
            ok = False
            why = '未检测'
            try:
                ok = bool(gpu_wgpu.available())
            except Exception as e:
                why = f'{type(e).__name__}: {e}'
            bits.append('wgpu 可用' if ok else f'wgpu 不可用（{why}）')
        except Exception:
            bits.append('wgpu 不可用')
        if os.environ.get('DWT_AVAIL_MB'):
            bits.append(f'内存预算被 DWT_AVAIL_MB={os.environ["DWT_AVAIL_MB"]} 覆盖')
        log(' | '.join(bits))
    except Exception:
        pass


def run(p: PwsParams, on_log=None, on_progress=None, should_cancel=None) -> Dict:
    """PWS 引擎的对外唯一入口（界面与命令行都调它）

    三个钩子都可选：不传就是纯命令行行为（日志走 stdout、无进度、不可终止）。
        跑完自动解绑，避免界面回调被后续调用意外触发。
    """
    set_hooks(on_log, on_progress, should_cancel)
    try:
        _log_env_once()
        return _run(p)
    finally:
        set_hooks()


def _run(p: PwsParams) -> Dict:
    """引擎主流程：展开素材 → 按帧头滤镜名自动分组 → 逐组独立叠加。

    分组（group_frames_by_filter）：不同滤镜的星点亮度/天空水平差异大，星表、σ、
    权重必须组内自洽，故逐组各走一遍完整流程（读帧→星表→R 场→叠加→导出），
    组间不共用任何中间量。只有一种滤镜时只有一组。
    """
    photos = (p.photos or '').strip()
    if not photos:
        raise SystemExit('请选择素材（目录或单张帧）')
    frames, sources = collect_frames(photos, allow_aligned=bool(int(getattr(p, 'aligned', 0) or 0)))
    if not (p.out or '').strip():
        raise SystemExit('请选择输出目录（不放在素材目录里，避免成品被当成帧）')
    out_dir = Path(p.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit('读帧', 0.0, '读头分组')
    groups = group_frames_by_filter(frames)
    want = [x.strip() for x in (getattr(p, 'pick', '') or '')
            .replace(',', ';').split(';') if x.strip()]
    if want:
        low = {k.lower(): k for k in groups}
        chosen: List[str] = []
        for w in want:
            if w in ('无滤镜信息', 'none', '(none)'):
                w = ''
            k = low.get(w.lower())
            if k is None:
                raise SystemExit('通道 %r 不存在；素材里有：%s' % (
                    w, '、'.join((k2 or '无滤镜信息') for k2 in groups)))
            if k not in chosen:
                chosen.append(k)
        groups = {k: groups[k] for k in chosen}
        log('[分组] 只叠所选通道：'
            + '、'.join((k or '无滤镜信息') + f'×{len(v)}' for k, v in groups.items()))
    multi = len(groups) > 1
    # ---- CFA（彩机）检测：--cfa auto 从帧头读拜耳阵列，也可手动指定 ----
    cfa_pat = None
    _cfa_opt = str(getattr(p, 'cfa', 'off') or 'off').strip()
    if _cfa_opt.lower() not in ('', 'off', '0', 'no', 'none') and _cfa is not None:
        want_pat = _cfa_opt.upper()
        probe_file = next(iter(groups.values()))[0]
        try:
            probe_img, probe_meta = read_frame(probe_file)
            probe_shape = tuple(getattr(probe_img, 'shape', ()))
            del probe_img
        except Exception:
            probe_meta, probe_shape = {}, ()
        if len(probe_shape) == 3:
            raise SystemExit('素材已经是 3 通道彩图（解过马赛克），不需要 --cfa')
        if _INPUT_FLAGS.get('per_channel'):
            log('[CFA] 输入已按通道拆分（每帧是解过马赛克的单色帧）：各通道单独叠加，'
                '不再按拜耳马赛克取样（否则 R/B 只能拿到 1/4 样本，出现网格与横纹）')
            cfa_pat = ''
            want_pat = ''
        elif want_pat in _cfa.PATTERNS:
            cfa_pat = want_pat
        else:
            cfa_pat = _cfa.detect_pattern(probe_meta)
            if not cfa_pat:
                log('[CFA] 帧头没有拜耳信息（BAYERPAT / CFASourcePattern），按单色处理；'
                    '如果这确实是彩机，请在高级区手动选 RGGB / BGGR / GRBG / GBRG')
        if cfa_pat:
            log(f'[CFA] 彩机素材，拜耳阵列 {cfa_pat}：R/G/B 三通道各自拆色后叠加'
                f'（G 先跑并记录配准与画布，R/B 复用同一套几何，保证三通道像素级对齐）')
    _wshare = str(getattr(p, 'cfa_weights', 'auto') or 'auto').lower() in ('shared', 'auto')
    # 逐通道独立配准（PI 的做法）：只关掉「几何复用」，权重仍可共用
    _align_own = str(getattr(p, 'cfa_align', 'shared') or 'shared').strip().lower() == 'own'
    # 逐通道画布：关掉「画布强制统一」，各通道用自己的配准结果定画布
    _grid_own = str(getattr(p, 'cfa_grid', 'shared') or 'shared').strip().lower() == 'own'
    _gshare = bool(_wshare) and not _align_own
    _share = _wshare
    # shared：G 先跑（绿像素最多、星点测量最稳），R/B 复用它量到的 C/S/T
    if cfa_pat and str(getattr(p, 'cfa_debayer', 'bilinear') or 'bilinear').lower() == 'superpixel':
        log('[CFA] 注意：2×2 合并尚未接线，本次按双线性全分辨率拆色（画幅不变）')
    if cfa_pat:
        # 调参只关心亮度口径（G），跑 12 组候选时不必三通道全叠：
        # 环境变量 DWT_CFA_CHANNELS 缺省 RGB，tune 子进程里设成 G。
        _want = str(os.environ.get('DWT_CFA_CHANNELS', 'RGB') or 'RGB').upper()
        chans = tuple(c for c in (('G', 'R', 'B') if _share else ('R', 'G', 'B'))
                      if c in _want) or (('G',) if _share else ('R',))
    else:
        chans = (None,)
    if (cfa_pat and str(getattr(p, 'cfa_weights', 'shared') or 'shared').strip().lower() == 'auto'
            and not _CFA.get('wauto_done')):
        _CFA['wauto_done'] = True
        # 实测结论（2026-11-19，M81 60 帧 + 12/16 帧探针）：真 CFA 下逐帧 FWHM
        # 的逐通道估计被量测噪声主导——12 帧 60 颗星时 R/G、B/G 的相关系数只有
        # -0.10 / 0.32，16 帧 80 颗星也只到 0.32；而拿 debayered 帧逐通道正常
        # 叠出来的排序相关是 0.986~0.995。也就是说「逐通道权重」在这里引入的
        # 主要是噪声，不是信息，还要多花两遍 C 测量（60 帧约 26 s）。
        # 因此 auto = 共用；真要逐通道，手动填 --cfa-weights own。
        log('[CFA] 权重 auto：真 CFA 下逐通道 FWHM 估计被噪声主导（实测排序相关≈0.3），'
            '逐通道权重只会放大噪声且多花两遍测量 → 使用共用权重；要强制逐通道请填 own')
    _stale = _clean_stale_frames()
    if _stale:
        log('[清理] 删掉 %d 个过期的临时帧缓存（>2 小时）' % _stale)
    # ---- 磁盘预检：cfa_pat 已定，知道要出几个平面了 ----
    try:
        _p0 = next(iter(groups.values()))[0]
        _img0, _meta0 = read_frame(_p0)
        _shp0 = tuple(getattr(_img0, 'shape', ()))[:2]
        _px0 = float(_shp0[0] * _shp0[1]) if len(_shp0) == 2 else 0.0
        del _img0
    except Exception:
        _px0 = 0.0
    _n_ch = 3 if cfa_pat else max(1, len(groups))
    _lim = int(getattr(p, 'limit', 0) or 0)
    _n_fr = sum((min(len(v), _lim) if _lim > 0 else len(v))
                for v in groups.values())
    _check_disk_space(frames, out_dir,
                      float(getattr(p, 'drizzle_scale', 1.0) or 1.0), log,
                      n_ch=_n_ch, px=_px0, n_frames=_n_fr,
                      rgb=bool(cfa_pat))
    _CFA['share'] = None
    if not cfa_pat:
        _clear_cfa_env()
    _CFA['want_share'] = bool(cfa_pat and _gshare)
    _CFA['want_wshare'] = bool(cfa_pat and _wshare)
    _CFA['want_wshare'] = bool(cfa_pat and _wshare)
    _CFA['geo'] = None
    _CFA['auto_own'] = None
    t_start = time.perf_counter()
    res: Dict = {}
    summaries: List[Dict] = []
    for gi, (fname, gframes) in enumerate(groups.items()):
        stem = _out_stem(p.tag, fname, multi)
        # 多组时每组日志都带前缀区分（无滤镜信息的组用「未知」）
        reset_fit_stat()
        _set_group(gi, len(groups), (fname or '未知') if multi else '')
        log(f'—— 组 {gi + 1}/{len(groups)}：{fname or "无滤镜信息"}，'
            f'{len(gframes)} 帧 ——')
        for ch in chans:
            if cfa_pat and bool(getattr(p, 'cfa_drizzle', False)) and ch == chans[0]:
                # CFA Drizzle：不解马赛克，三平面一次叠完（不再走逐通道循环）
                _CFA['ch'], _CFA['pattern'] = None, None
                from cfadz import run_group as _cfa_dz_run
                res = _cfa_dz_run(gframes, sources, p, out_dir, stem, cfa_pat, log,
                                  emit=_emit)
                res['stem'] = stem
                res['out_dir'] = str(out_dir)
                summaries.append({
                    'filter': (fname if not cfa_pat else 'RGB'),
                    'n': int(res['n']), 'shape': res['shape'],
                    'fwhm_med': float(res['fwhm_med']),
                    'fwhm_out': float(res['fwhm_out']),
                    'rej_rate': float(res['rej_rate']),
                    'stem': stem, 'file': str(out_dir / f'{stem}_RGB.xisf'),
                })
                break
            _CFA['ch'], _CFA['pattern'] = ch, cfa_pat
            _env_raw = str(os.environ.get('DWT_CFA_RAW', '') or '').strip().lower()
            _p_raw = str(getattr(p, 'cfa_raw', 'auto') or 'auto').strip().lower()
            if _env_raw in ('0', '1'):
                _CFA['raw'] = (_env_raw == '1')
            elif _p_raw in ('hybrid', 'mix', 'mixed'):
                # 混合：亮度（G）走真 CFA 保锐度，色度（R/B）走插值保信噪比。
                # 这正是彩色相机的「色度子采样」思路：人眼对色度分辨率不敏感。
                _CFA['raw'] = (ch == 'G')
            elif _p_raw in ('on', '1', 'true'):
                _CFA['raw'] = True
            elif _p_raw in ('off', '0', 'false'):
                _CFA['raw'] = False
            else:   # auto
                _CFA['raw'] = bool(cfa_pat) and bool(getattr(p, 'drizzle', False))
            if cfa_pat:
                _tag = '真 CFA（不插值，按颜色投样本）' if _CFA['raw'] else '插值拆色（双线性/2×2）'
                if _p_raw in ('hybrid', 'mix', 'mixed') and ch != 'G':
                    _tag += '（混合模式：色度通道）'
                log('[CFA] 采样模式：%s' % _tag)
            if cfa_pat:
                log(f'—— CFA 通道 {ch} ——')
            stem_ch = stem if ch is None else f'{stem}_{ch}'
            res = _run_group(gframes, sources, p, out_dir, stem_ch)
            # 评价页按 stem 回读 _R.npy / _neff.npy（res 不持有 R 数组）
            res['stem'] = stem_ch
            res['out_dir'] = str(out_dir)
            summaries.append({
                'filter': (fname if not cfa_pat else f'{ch}'),
                'n': int(res['n']), 'shape': res['shape'],
                'fwhm_med': float(res['fwhm_med']), 'fwhm_out': float(res['fwhm_out']),
                'rej_rate': float(res['rej_rate']),
                'stem': stem_ch, 'file': str(out_dir / f'{stem_ch}.xisf'),
            })
            if multi or cfa_pat:
                res['stack'] = None      # 成品已落盘；别把每组成品都攒在内存里
        if cfa_pat or _INPUT_FLAGS.get('per_channel'):
            try:
                # 按通道拆分的输入：每组只叠出一个通道，等三张都在了再合成
                _pc = bool(_INPUT_FLAGS.get('per_channel'))
                if _pc:
                    _mstem = stem[:-2] if stem[-2:] in ('_R', '_G', '_B') else stem
                    _missing = [c for c in 'RGB'
                                if not (Path(out_dir) / ('%s_%s.xisf' % (_mstem, c))).exists()]
                    if _missing:
                        log('[CFA] 还缺通道 %s，本组先不合成 RGB' % ''.join(_missing))
                        raise _SkipMerge()
                else:
                    _mstem = stem
                # 彩机参考目录（WBPP debayered）：有就用它估色差，比在成品上估更干净
                _ref = None
                try:
                    _man = str(getattr(p, 'cfa_ref', '') or '').strip()
                    if _man:
                        _ref = Path(_man)
                        if not _ref.is_dir():
                            log(f'[参考] 指定的彩机参考目录不存在：{_man}（改用自动扫描）')
                            _ref = None
                    if _ref is None:
                        _src = Path(str(getattr(p, 'photos', '') or ''))
                        if _src.is_dir():
                            _ref = _find_cfa_ref(_src)
                    if _ref is not None:
                        _merge_cfa_rgb._ref_dir = _ref
                        log(f'[参考] 彩机参考目录：{_ref}')
                    else:
                        _merge_cfa_rgb._ref_dir = None
                except Exception as _e:
                    _merge_cfa_rgb._ref_dir = None
                    log(f'[参考] 扫描彩机参考目录失败（跳过）：{_e}')
                # 无论画布模式如何都尝试合成：逐通道模式下三张画布可能差一两个像素，
                # 合并函数会按最小画布裁齐并给出提示（实测差 1~2 px，裁掉即可，
                # 不合成反而让用户以为功能没生效）。
                if _grid_own:
                    log('[CFA] 逐通道画布：按最小画布裁齐后合成 RGB'
                        '（像素级对齐仍以各通道自己的配准为准）')
                _merge_cfa_rgb._skip_ca = bool(_pc)
                _merge_cfa_rgb._balance = str(getattr(p, 'rgb_balance', 'off') or 'off')
                if _pc:
                    log('[CFA] 输入已按通道拆分：合成时跳过色差平移'
                        '（WBPP 已对齐，再平移会糊边）')
                _merge_cfa_rgb(out_dir, _mstem, log)
            except _SkipMerge:
                pass
            except Exception as e:
                log(f'[CFA] RGB 合成失败（不影响三个单通道成品）：{e}')
        _CFA['ch'], _CFA['pattern'] = None, None
        # 三通道共用缓存到这里才真正释放（含删 memmap 临时文件）
        for _f in list(_FS_CACHE.values()):
            try:
                _f._shared = False
                _f.close(keep=keep_frames)
            except Exception:
                pass
        _FS_CACHE.clear()
    _set_group(0, 1, '')
    _fit_line = fit_warn_line()
    if _fit_line:
        log(_fit_line)
    if multi:
        log(f'[时] 全部 {len(groups)} 组总计 {time.perf_counter() - t_start:.1f}s')
        res['groups'] = summaries
    return res


def _ln_normalize(fs, ref_idx: int, block: int = 32,
                  lo: float = 0.8, hi: float = 1.25, log=print) -> None:
    """逐帧局部归一化（LN）：按 block×block 块中位比，把每帧的缓变大尺度响应
    乘回到参考帧上（星云/渐变/残留平场差异），不改小尺度细节。

    只在 memmap 后端生效（原地写回，避免再占一份内存）。参考帧取 S 最高的那帧。
    """
    mm = getattr(fs, 'mm', None)
    if mm is None or getattr(fs, 'resident', True):
        log('[LN] 跳过：仅 memmap 后端支持原地归一化')
        return
    h, w = fs.shape
    gh, gw = h // block, w // block
    if gh < 4 or gw < 4:
        log('[LN] 跳过：图像太小')
        return
    hs, ws = gh * block, gw * block
    ref = np.asarray(mm[ref_idx][:hs, :ws], dtype=np.float32)
    rb = np.median(ref.reshape(gh, block, gw, block), axis=(1, 3))
    rb = np.maximum(rb, 1e-8)
    nfix = 0
    cmin, cmax = 9e9, -9e9
    for i in range(len(fs)):
        if i == ref_idx:
            continue
        f = np.asarray(mm[i][:hs, :ws], dtype=np.float32)
        fb = np.median(f.reshape(gh, block, gw, block), axis=(1, 3))
        corr = rb / np.maximum(fb, 1e-8)
        corr = np.clip(corr, lo, hi)
        corr = _boxf(corr, 3, 'nearest')
        if not np.isfinite(corr).all():
            continue
        z = ndimage.zoom(corr, (h / gh, w / gw), order=1, mode='nearest')
        z = np.clip(z, lo, hi).astype(np.float32)
        mm[i] *= z
        nfix += 1
        cmin = min(cmin, float(corr.min())); cmax = max(cmax, float(corr.max()))
    log(f'[LN] 局部归一化（参考帧 #{ref_idx}，{block}px 块）：{nfix} 帧已校正，'
        f'系数 {cmin:.3f}~{cmax:.3f}')


def _local_sharp_maps(fs, idx, plan, blk: int = 64, log=None):
    """逐帧、逐块的局部锐度（高频能量）相对量，返回 (n, nby, nbx) 乘性权重。

    只看「同一块里这一帧相对其它帧更锐还是更糊」，绝对值无意义，
    所以每块都对中位数归一，再 clamp 到 [0.5, 2.0] 防止个别块暴走。
    """
    import numpy as _np
    try:
        from scipy import ndimage as _nd
    except Exception:
        return None
    H, W = int(plan['H']), int(plan['W'])
    nby, nbx = int(_np.ceil(H / blk)), int(_np.ceil(W / blk))
    n = len(fs)
    E = _np.full((n, nby, nbx), _np.nan, _np.float32)
    for k, i in enumerate(idx):
        try:
            a = _np.asarray(fs.frame(int(i)), _np.float32)
        except Exception:
            continue
        if a.ndim != 2:
            continue
        # 目标网格上抽样：与 drizzle 的 plan 对齐（最近邻即可，块统计对亚像素不敏感）
        ys = _np.clip(((_np.arange(int(a.shape[0])) * plan['s'] + plan['oy'])), 0, H - 1).astype(_np.int64)
        xs = _np.clip(((_np.arange(int(a.shape[1])) * plan['s'] + plan['ox'])), 0, W - 1).astype(_np.int64)
        # 高频：减去 3x3 均值
        hp = a - _boxf(a, 3, 'nearest')
        # 按输出块聚合（用采集到的坐标索引）
        b = _np.zeros((nby, nbx), _np.float64)
        w = _np.zeros((nby, nbx), _np.float64)
        by = _np.broadcast_to((ys[:, None] // blk), a.shape).astype(_np.int64)
        bx = _np.broadcast_to((xs[None, :] // blk), a.shape).astype(_np.int64)
        v = (hp * hp).astype(_np.float64)
        _np.add.at(b, (by.ravel(), bx.ravel()), v.ravel())
        _np.add.at(w, (by.ravel(), bx.ravel()), 1.0)
        ok = w > 0
        b[ok] /= w[ok]
        E[k] = _np.where(ok, b, _np.nan)
    with _np.errstate(invalid='ignore'):
        med = _np.nanmedian(E, axis=0)
    r = E / _np.where(med > 0, med, _np.nan)
    r = _np.clip(_np.nan_to_num(r, nan=1.0), 0.5, 2.0)
    if log:
        log('[两遍] 局部锐度场：%d 帧 × %d×%d 块（%dpx），相对权重 %.2f~%.2f'
            % (n, nby, nbx, blk, float(_np.min(r)), float(_np.max(r))))
    return r.astype(_np.float32)

def _flag(v) -> bool:
    """命令行/界面传进来的布尔：'0'/'false' 也算关（argparse 里 bool 会退化成字符串）"""
    if isinstance(v, str):
        return v.strip().lower() not in ('', '0', 'false', 'no', 'off')
    return bool(v)


def _run_group(frames: List[Path], sources: List[Path], p: PwsParams,
               out_dir: Path, stem: str) -> Dict:
    """单滤镜组的叠加本体：展开素材与建输出目录由 _run 完成，此处只做该组的
    完整流程（读帧→星表→R 场→叠加→导出），成品文件主名由 stem 给出。"""
    # 参数别名：算法体统一从 PwsParams 取值
    A, B = float(p.A), float(p.B)
    # A 完全按用户填的走（彩机/黑白都不再自动改），选项栏下方给了建议值。
    # 但真 CFA 下 A>6 会让「最锐的那一帧」在逐像素上独占权重：实测用户 78 帧 B、
    # A=10 时单帧最大权重是均值的 409 倍，成品在小星周围出现 8.5%~28.6% 的
    # 低于天空 −2σ 的像素（A=6 时 0.5%，A=4 时 0.3%）。这里对真 CFA 设上限 6，
    # 并在日志里说明；要保留原值请设 DWT_A_KEEP=1。
    if _CFA.get('pattern') and A > 6.0 and os.environ.get('DWT_A_KEEP', '') != '1':
        log(f'[CFA] 清晰度 A 上限 6：你填的 {A:g} 会让最锐帧在逐像素上独占权重'
            '（实测 A=10 时小星周围低于天空 -2σ 的像素 8.5%~28.6%，A=6 为 0.5%），'
            '已按 6 计算；要强制使用请设 DWT_A_KEEP=1')
        A = 6.0
    # 黑白/单色：A_SHARP=20 是设计上限（注释在 :80），但以前只是写在注释里、
    # 没有真正约束——填 40 会照单全收。A=40 时 C^40 在 C∈[0.3,1.7] 上跨 30 个
    # 数量级，结构区等于「逐像素挑一帧」，局部噪声退化为单帧。这里补上钳制。
    if (not _CFA.get('pattern')) and A > A_SHARP and os.environ.get('DWT_A_KEEP', '') != '1':
        log(f'[A] 清晰度 A 上限 {A_SHARP:g}：你填的 {A:g} 会让结构区由单帧独占'
            f'（C^A 跨约 2×A 个数量级），已按 {A_SHARP:g} 计算；要强制使用请设 DWT_A_KEEP=1')
        A = A_SHARP
    r_floor = float(p.r_floor)
    env_win_mult, env_eps, env_p = float(p.env_win_mult), float(p.env_eps), float(p.env_p)
    star_grow_pct = float(p.star_grow_pct)
    rej_k, n_star = float(p.rej_k), int(p.n_star)
    _c_star = int(getattr(p, 'c_star', 64) or 0)
    rej_gate = float(p.rej_gate)                  # 剔除比例上限（默认 1/3）
    limit = int(p.limit) or None
    crop, keep_frames, frames_dir = int(p.crop), bool(p.keep_frames), p.frames_dir
    save_xisf = True
    t_start = time.perf_counter()

    _emit('读帧', 0.0, '准备')

    # ---- 阶段 0：读帧 + 帧级归一（自动选驻留 / memmap）----
    # --frame 模式下画布就是参考栅格，--crop 只是保存时的数组切片：帧必须整幅
    # 参与叠加（只裁输入的话，画布中心那块根本没有数据，切出来是空的），
    # 所以输入帧不裁，裁剪框在写盘前按画布坐标算。
    _frame_mode = bool(str(getattr(p, 'frame', '') or '').strip())
    _crop_in = int(crop or 0)
    if _frame_mode and _crop_in > 0:
        log(f'[裁剪] --frame 模式：输入帧不裁，{_crop_in}×{_crop_in} 在保存时按画布坐标切')
        _crop_in = 0
    center = resolve_center(frames, _crop_in, parse_center(p.center))
    # 三通道共用几何时，读帧 / 减背景 / 量天空这十几秒只需做一次。
    # 同一个 run 内按（帧列表、裁剪、中心、上限、背景模型）缓存；帧数据只读
    # （拆色在 _PlaneFS 里按需做），复用不会污染任何通道。
    _ck = (tuple(str(f) for f in frames), int(_crop_in), tuple(center or ()),
           int(limit or 0), int(getattr(p, 'bgdeg', 1) or 0),
           int(getattr(p, 'bg_r', 0) or 0), bool(keep_frames))
    _hit = _FS_CACHE.get(_ck)
    if _hit is not None and _hit.arrays is not None or (
            _hit is not None and _hit.mm is not None):
        fs = _hit
        fs._shared = True
        log('[读帧] 复用本组已读入的 %d 帧（三通道共用同一批帧，免重复读盘）' % fs.n)
        try:
            _emit('读帧', 1.0, '%d/%d 帧（复用）' % (fs.n, fs.n))
        except Exception:
            pass
    else:
        fs = prepare_frames(frames, ';'.join(str(s) for s in sources), _crop_in, center,
                            limit, frames_dir, keep_frames=keep_frames,
                            bgdeg=int(getattr(p, 'bgdeg', 1) or 0),
                            bg_r=int(getattr(p, 'bg_r', 0) or 0))
        if _CFA.get('pattern'):
            fs._shared = True
        _FS_CACHE[_ck] = fs
    n = len(fs)
    # ---- 彩机对齐方式 auto：在 G 通道这遍（第一个通道）量一次色差 ----
    if (_CFA.get('pattern') and _CFA.get('ch') == 'G'
            and str(getattr(p, 'cfa_align', 'auto') or 'auto').strip().lower() == 'auto'
            and _CFA.get('auto_own') is None):
        try:
            (_dyR, _dxR), (_dyB, _dxB) = _ca_from_fs(fs, _CFA['pattern'], log)
            _ca_max = max(abs(_dyR), abs(_dxR), abs(_dyB), abs(_dxB))
            _CFA['auto_own'] = bool(_ca_max > 0.15)
            log('[CFA] 对齐方式 auto：首帧色差最大 %.3f px（阈值 0.15）→ %s'
                % (_ca_max, '逐通道独立配准' if _CFA['auto_own'] else '共用几何'))
        except Exception as _e:
            _CFA['auto_own'] = False
            log('[CFA] auto 判色差失败（%s: %s），退回共用几何' % (type(_e).__name__, _e))
    if _CFA.get('auto_own'):
        _CFA['want_share'] = False
    metas = fs.metas
    skies = fs.skies
    # 量纲回乘：内部一切计算都在 EXPTIME_REF 基准上（权重/排异全是比值，不受影响），
    #   但成品若留在该量纲，数值 = 原始×(基准/曝光)，跨软件对比必踩量纲坑
    #   （900s 素材的读数只有原始的 1/3）。故导出前把成品乘回"曝光中位/基准"：
    #   全部同曝光 → 逐位回到输入帧量纲；混合曝光 → 中位曝光量纲（约定明确）。
    #   σ 的展示/落盘同步该缩放：均匀缩放不改 S/σ 比值，权重数学零变化。
    t_med = float(np.median([float(m['exptime']) for m in metas]))
    scale_out = t_med / EXPTIME_REF

    # S_i = σ_中位 / σ_i（归一化后测 → 曝光差异已被吸收，可直接跨帧比）
    sig = fs.sigma
    S = np.median(sig) / np.maximum(sig, 1e-30)
    # 帧间噪声离群保护：σ 极差大时（薄云、月光、不同增益），S² 会让 1~2 张
    # 异常干净的帧独占全部权重——79 帧叠出来 N_eff 只有 2.2，成品几乎等于单帧。
    # 这里把 S 限在 [1/K, K]：保留逆方差排序，但单帧最多 K² 倍权重。
    _k_clip = float(getattr(p, 'snr_clip', 3.0) or 0.0)
    if _k_clip > 1.0:
        _n_clip = int(np.count_nonzero((S < 1.0 / _k_clip) | (S > _k_clip)))
        S = np.clip(S, 1.0 / _k_clip, _k_clip)
        if _n_clip:
            _r = (float(np.max(sig)) / max(float(np.min(sig)), 1e-30))
            log(f'[S] 帧间噪声极差 {_r:.1f}×，S 已限幅到 [{1.0 / _k_clip:.3f}, '
                f'{_k_clip:g}]（{_n_clip} 帧被限；避免单帧独占权重，'
                f'要关掉填 --snr-clip 0）')
    # ---- 局部噪声权重（实验，--local-snr 1）----
    # 月光/光害/薄云在视场里不均匀：同一帧在一角被抬高、另一角还干净。
    # 用每帧自己的背景梯度模型估每格天空，再按散粒噪声把 σ 缩放到该格，
    # 得到 S_i(cell) = median_i σ_i(cell)/σ_i(cell)：亮格自动少用被污染的帧。
    _S_field = None
    sig_out = sig * scale_out          # 展示与落盘量纲 = 成品量纲（内部仍用 sig 算比值）
    log(f'[S] 单帧信噪比权重（σ_中位/σ_i）{S.min():.3f}~{S.max():.3f}  中位 {np.median(S):.3f}')
    log(f'[σ] 逐帧噪声 {sig_out.min():.4e}~{sig_out.max():.4e}'
        f'（极差 {sig_out.max() / sig_out.min():.2f}×，量纲=素材）')

    # ---- 阶段 A：C_i（星点 FWHM，开源 fit_fwhm）----
    # 先用最干净的一帧建固定星表（星表位置对全部帧共用）
    # 参考帧不能只看「天空最干净」：雾帧 σ 低、星也少，拿它当参考会让后面每帧
    # 的全星验证都压在很低的命中率上。在 S 最高的前 20 帧里挑检出星最多的一帧。
    from drizzle import _detect_stars as _ds
    from drizzle import _rot_hist as _rh
    _cand = np.argsort(-S)[:20]
    seed = int(_cand[0])
    _best_n = -1
    _pts = {}
    for _c in _cand:
        try:
            _pts[int(_c)] = _ds(fs.frame(int(_c)))
        except Exception:
            _pts[int(_c)] = None
    for _c in _cand:
        _pc = _pts.get(int(_c))
        if _pc is None or len(_pc) < 8:
            continue
        if len(_pc) <= _best_n:
            continue
        # 当参考帧还得「测得动别人」：拿 S 最高的几帧试角度直方图，
        # 一帧都对不上（长度谱不对称时会发生）就换下一帧当参考。
        _probe = [int(_k) for _k in _cand if int(_k) != int(_c)][:3]
        _okp = 0
        for _k in _probe:
            _pk = _pts.get(_k)
            if _pk is None or len(_pk) < 8:
                continue
            try:
                if _rh(_pc, _pk) is not None:
                    _okp += 1
            except Exception:
                pass
        if _okp == 0 and _probe:
            continue
        seed, _best_n = int(_c), len(_pc)
        if _best_n >= 400:
            break
    if _best_n > 0:
        log(f'[星表] 候选参考帧中 #{seed} 可用且检出星最多（{_best_n} 颗）')
    if bool(getattr(p, 'ln', False)):
        _ln_normalize(fs, seed, block=32, log=log)
    t1 = time.perf_counter()
    _emit('星表', 0.1, '检测星点')
    _reg = (_PlaneFS(fs, _CFA['pattern'], _CFA['ch'])
            if (_CFA.get('raw') and _CFA['ch']) else fs)
    stars = build_star_table(_reg.frame(seed), n_star=n_star)
    log(f'[星表] 帧 #{seed}（S 最高）上 DAOStarFinder 选出 {len(stars)} 颗'
        f'（未饱和、孤立、最亮）  {time.perf_counter() - t1:.1f}s')
    if len(stars) < 5:
        raise SystemExit('星表不足 5 颗，无法测 C_i')
    _emit('星表', 1.0, f'{len(stars)} 颗')

    # 留一份参考帧副本：后面 fs 会 close，SNR 对比还要用它
    try:
        _snr_seed_img = np.array(_reg.frame(seed), np.float32, copy=True)
    except Exception:
        _snr_seed_img = None
        log('[SNR] 单张基准量不出：%s' % _e0)

    # 逐帧独立、互不引用 → 可分给多进程；结果按下标落位，与串行逐位一致。
    #   只取星点小窗 → 每进程几乎不占内存，进程数可以放开（CAP_FWHM）
    fwhms = np.empty(n, dtype=np.float64)
    # C 只要中位：星表按亮度排序，等间隔抽样即可覆盖亮暗
    _cs = stars
    if _c_star and len(stars) > _c_star:
        _step = max(1, len(stars) // _c_star)
        _cs = stars[::_step][:_c_star]
    # ---- 卫星拖尾：整帧剔除（放在 fmed/C 统计之前，让权重表也不含它）----
    # 为什么不是降权：strip_weights 逐像素归一 mean_i(W)=1，某个画布像素只有这一帧
    # 覆盖时（素材抖动到 ±2900 px，这种像素很多），分母只剩它自己，权重被整体约掉，
    # 拖尾照样原样印进成品。实测把 idx25 的 C 从 1.37993 压到 0.896952（份额 0.656%
    # → 0.0096%，68×），沿线幅度只从 +2.97e-5 降到 +2.49e-5（−16%）；而整帧不参与
    # 叠加时同一条线降到 +1.03e-6。所以这里按"帧"剔除，不做标量惩罚。
    _tdrop = np.zeros(n, bool)
    _trail_info = {}
    _td = bool(getattr(p, 'trail_drop', True))
    _tp = float(getattr(p, 'trail_penalty', 1.0) or 0.0)
    if _td:
        t1c = time.perf_counter()
        for i in range(n):
            _check_cancel()
            try:
                _px, _ln = trail_pixels(fs.frame(i))
            except Exception:
                _px, _ln = 0, 0.0
            if _ln > 0:
                _tdrop[i] = True
                _trail_info[i] = (_px, _ln)
            _emit('查拖尾', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
        if _tdrop.any():
            log('[C] 卫星拖尾：剔除帧 '
                + ' '.join(f'#{int(i)}（长 {_trail_info[i][1]:.0f}px）'
                           for i in sorted(_trail_info))
                + f'  参与 {int(n - _tdrop.sum())}/{n}  {time.perf_counter() - t1c:.1f}s')
        else:
            log(f'[C] 未检出卫星拖尾（{n} 帧全干净）  {time.perf_counter() - t1c:.1f}s')
    ini, ini_args = fs.reader
    nw = 0 if (not fs.parallelable or _no_parallel()) \
        else min(plan_workers(32 * 2 ** 20, '测星点', cap=CAP_FWHM), n)
    if _CFA.get('raw') and _CFA.get('pattern'):
        # 真 CFA：帧存储是马赛克，必须在 G 平面上量（否则拟合全废、C 恒 1）。
        # 而且这个量与最终跑哪个通道无关（永远在 G 平面上量），所以三通道共用几何时
        # R/B 直接复用 G 的结果——省掉每通道一次 60 帧 × 全幅拆色 + 120 次高斯拟合。
        _fw_cache = _CFA.get('fwhm_cache')
        if _fw_cache is not None and len(_fw_cache) == n:
            log('[C] 复用 G 通道逐帧 FWHM（同一批帧、同在 G 平面上量，与重算等价）')
            fwhms = np.array(_fw_cache, np.float64, copy=True)
            for i in range(n):
                _emit('测星点', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
            _reuse_fw = True
        else:
            _reuse_fw = False
        # 每进程峰值 ≈ 6 个整帧临时数组（拆色/卷积/掩膜），全幅约 600 MiB；
        # 按 900 MiB 估会开太多进程，用户 16 GiB 机器上实测 OOM。
        _bgs0 = tuple(getattr(fs, 'bgshape', fs.shape))
        _per_c = max(900 * 2 ** 20, _bgs0[0] * _bgs0[1] * 4 * 6 + (256 << 20))
        if not _reuse_fw and fs.parallelable and not _no_parallel():
            _nw_c = min(plan_workers(_per_c, '测星点', cap=CAP_FWHM), n)
            # 再按「可用内存的一半」压一次：父进程还有 memmap 页缓存与成品画布，
            # 按满额分配会在真机上把提交内存打穿（用户 16 GiB + 全幅 CFA 实测 OOM）。
            _nw_c = max(1, min(_nw_c, int((_avail_bytes() * 0.5) // _per_c)))
        else:
            _nw_c = 1
        _oom_fallback = False
        if _nw_c > 1 and not _reuse_fw:
            log('[C] 真 CFA：在 G 平面上逐帧量 FWHM（%d 进程并行）' % _nw_c)
            _ini, _iargs = fs.reader
            _bgc = getattr(fs, 'bgcoef', None)
            _bgd = int(getattr(fs, 'bgdeg', 0) or 0)
            _bgs = tuple(getattr(fs, 'bgshape', fs.shape))
            _bgrc = getattr(fs, 'bgrcoef', None)
            ex = ProcessPoolExecutor(max_workers=_nw_c,
                                     initializer=_ini, initargs=_iargs)
            try:
                _jobs = ((i, _cs, _CFA['pattern'],
                          None if _bgc is None else _bgc[i], _bgd, _bgs,
                          None if _bgrc is None else _bgrc[i]) for i in range(n))
                for _i, _v, _fx, _dn, _dc in ex.map(_task_fwhm_cfa, _jobs, chunksize=1):
                    _check_cancel()
                    fwhms[_i] = _v
                    _FIT_STAT['n'] += _dn
                    _FIT_STAT['conv'] += _dc
                    _emit('测星点', (_i + 1) / max(n, 1), f'{_i + 1}/{n} 帧')
            except MemoryError as _eom:
                # 并行拆色把内存打穿时不要把整轮叠加判死：退回串行重算这一段。
                _oom_fallback = True
                log(f'[C] 并行测 FWHM 内存不足（{_eom}），改串行重算这一段')
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
        log('[C] 真 CFA：在 G 平面上逐帧量 FWHM（串行）') if (not _reuse_fw and (_nw_c <= 1 or _oom_fallback)) else None
        for i in ([] if (_reuse_fw or (_nw_c > 1 and not _oom_fallback)) else range(n)):
            _check_cancel()
            try:
                _g = _cfa.plane(np.asarray(fs.frame(i), np.float32),
                                _CFA['pattern'], 'G')
                fwhms[i] = measure_fwhm(_g, _cs)
            except Exception as _ec:
                fwhms[i] = np.nan
                if i == 0:
                    log(f'[C] G 平面量 FWHM 失败（{type(_ec).__name__}: {_ec}），C 退回 1.0')
            _emit('测星点', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
        if not _reuse_fw:
            _CFA['fwhm_cache'] = np.array(fwhms, np.float64, copy=True)
    elif nw > 1:
        _srch = int(getattr(p, 'fwhm_search', 0) or 0)
        ex = ProcessPoolExecutor(max_workers=nw, initializer=ini, initargs=ini_args)
        try:
            for i, v, fx, dn, dc in ex.map(_task_fwhm, ((i, _cs, _srch) for i in range(n)),
                                           chunksize=1):
                _check_cancel()
                fwhms[i] = v
                _FIT_STAT['n'] += dn
                _FIT_STAT['conv'] += dc
                _emit('测星点', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    else:
        for i in range(n):
            _check_cancel()
            fwhms[i] = measure_fwhm_frame(fs.mm, i, fs.off, _cs,
                                         search=int(getattr(p, 'fwhm_search', 0) or 0))
            _emit('测星点', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
    if _tdrop.any():
        fwhms[_tdrop] = np.nan          # 拖尾帧不进 fmed/C，也不进权重表
    ok = np.isfinite(fwhms)
    if not bool(ok.any()):
        log('[C] 所有帧的 FWHM 都没量出来（星点太少或拟合失败）→ 清晰度权重按 1.0 处理')
        fwhms = np.ones(n, np.float64)
        ok = np.isfinite(fwhms)
    fmed = float(np.median(fwhms[ok]))
    C = np.where(ok, fmed / np.maximum(fwhms, 1e-9), 1.0)
    log(f'[C] 逐帧星点 FWHM（开源 fit_fwhm，{len(_cs)} 颗中位）'
        f' {np.nanmin(fwhms):.2f}~{np.nanmax(fwhms):.2f}px  中位 {fmed:.2f}px  '
        f'{time.perf_counter() - t1:.1f}s')
    log(f'[C] 清晰度权重（FWHM_中位/FWHM_i）{C.min():.3f}~{C.max():.3f}')
    order = np.argsort(np.where(np.isfinite(fwhms), fwhms, np.inf))
    log('[C] 最锐 3 帧 ' + '  '.join(
        f'#{int(i)}({fwhms[int(i)]:.2f}px)' for i in order[:3]))
    log('[C] 最糊 3 帧 ' + '  '.join(
        f'#{int(i)}({fwhms[int(i)]:.2f}px)' for i in order[-3:]))

    # ---- 卫星拖尾惩罚（退路，默认关）：trail_drop=False 且 trail_penalty<1 时才走 ----
    # 保留是因为有些素材里拖尾只擦过边缘、整帧剔除代价太大；但要知道它的局限：
    # 某像素只有这一帧覆盖时，逐像素归一 mean_i(W)=1 会把标量权重整体约掉，
    # 沿线照样留痕（实测只能压掉 16%）。能剔除就剔除。
    if (not _td) and 0.0 < _tp < 0.999:
        t1c = time.perf_counter()
        _hit = []
        for i in range(n):
            _check_cancel()
            try:
                _px, _ln = trail_pixels(fs.frame(i))
            except Exception:
                _px, _ln = 0, 0.0
            if _ln > 0:
                _hit.append((i, _px, _ln))
            _emit('查拖尾', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
        if _hit:
            _mul = np.ones(n, np.float64)
            for _i, _px, _ln in _hit:
                _mul[_i] = _tp
            C = C * _mul
            log(f'[C] 卫星拖尾惩罚 {len(_hit)} 帧 ×{_tp:.2f}（检出长直线结构，未剔除）  '
                f'{time.perf_counter() - t1c:.1f}s')
            for _i, _px, _ln in sorted(_hit, key=lambda r: -r[2])[:3]:
                log(f'[C] 拖尾帧 #{int(_i)} 长 {_ln:.0f}px / {_px} 像素  '
                    f'FWHM {fwhms[int(_i)]:.2f}px  C→{C[int(_i)]:.3f}')
        else:
            log(f'[C] 未检出卫星拖尾（{n} 帧全干净）  {time.perf_counter() - t1c:.1f}s')

    # ---- 逐瓦片清晰度（可选）：每帧在 K×K 瓦片里各量一个 FWHM ----
    C_tiles = None
    _kt = int(getattr(p, 'sharp_tiles', 0) or 0)
    if _kt >= 2:
        t1b = time.perf_counter()
        _ft = np.full((n, _kt, _kt), np.nan, np.float32)
        _h0, _w0 = int(fs.mm.shape[1]), int(fs.mm.shape[2])
        for i in range(n):
            _check_cancel()
            try:
                if _CFA.get('pattern'):
                    _g = _cfa.plane(np.asarray(fs.frame(i), np.float32), _CFA['pattern'], 'G')
                else:
                    _g = np.asarray(fs.frame(i), np.float32)
                _ev = measure_fwhm_each(_g, _cs)
                _ft[i] = tile_fwhm_map(_ev, _cs, _g.shape[0], _g.shape[1], _kt).astype(np.float32)
            except Exception:
                pass
            _emit('测瓦片', (i + 1) / max(n, 1), f'{i + 1}/{n} 帧')
        _gmed = float(np.nanmedian(_ft))
        if np.isfinite(_gmed) and _gmed > 0:
            C_tiles = (_gmed / np.maximum(_ft, 1e-9)).astype(np.float32)
            # 星太少的瓦片回落到该帧的全局 C，避免出现 0/NaN 权重
            _bad = ~np.isfinite(C_tiles)
            if _bad.any():
                for i in range(n):
                    C_tiles[i][_bad[i]] = float(C[i])
            # 轻微平滑（σ≈0.6 瓦片）：压掉插值残留的折角，让权重场更接近真实视宁
            try:
                from scipy.ndimage import gaussian_filter as _gf
                C_tiles = np.stack([_gf(C_tiles[i], 0.6, mode='nearest')
                                    for i in range(n)]).astype(np.float32)
            except Exception:
                pass
            _sp = []
            for a in range(_kt):
                for b in range(_kt):
                    v = C_tiles[:, a, b]
                    v = v[np.isfinite(v)]
                    if len(v) > 2:
                        _sp.append(float(np.percentile(v, 90) / max(np.percentile(v, 10), 1e-9)))
            log(f'[C] 逐瓦片清晰度：{_kt}×{_kt} 瓦片，瓦片内 C 的 p90/p10 中位 {np.median(_sp):.2f}×，'
                f'{time.perf_counter() - t1b:.1f}s')
        else:
            log('[C] 逐瓦片清晰度：瓦片 FWHM 全为空，退回全帧标量权重')

    sigma_psf = fmed / 2.3548

    # ---- 拖线长度：逐帧二阶矩，供 star_grow 自动取值 ----
    #   帧级 λ1（沿长轴的方差）减去参考帧（最锐 k 帧）的 λ1，剩下的就是拖线本身
    #   贡献的方差；均匀拖线长度 L 满足 λ1−λ1_ref = L²/12。
    star_grow = 0.0
    t2 = time.perf_counter()
    if star_grow_pct > 0:
        lam1 = np.full(n, np.nan)
        pas = np.full(n, np.nan)
        nw = 0 if (not fs.parallelable or _no_parallel()) \
            else min(plan_workers(32 * 2 ** 20, '测拖线', cap=CAP_FWHM), n)
        if nw > 1:
            ex = ProcessPoolExecutor(max_workers=nw, initializer=ini, initargs=ini_args)
            try:
                for i, l1, pa in ex.map(_task_trail,
                                        ((i, stars, sigma_psf) for i in range(n)),
                                        chunksize=1):
                    _check_cancel()
                    lam1[i], pas[i] = l1, pa
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
        else:
            for i in range(n):
                _check_cancel()
                lam1[i], pas[i] = fs.trail(i, stars, sigma_psf)

    # 绝对量外扩：R=1 覆盖整个星点（含半高位置），背景仍等权
    _sgpx = float(getattr(p, 'star_grow_px', 0.0) or 0.0)
    if _sgpx > 0:
        star_grow = float(star_grow) + _sgpx / max(float(sigma_psf), 0.5)
        log(f'[R] 星点锐帧区按 {_sgpx:g}px 外扩（star_grow={star_grow:.2f}σ）')

    # ---- 阶段 B：R 场 ----
    _emit('R 场', 0.0, '建参考像')
    # 参考像帧数：G 用 8 就够；R/B 只有 1/4 的采样，参考像本身噪声大，
    # 量出来的 σ_psf 会被抬高（→ R 场过度平滑 → 成品发虚）。给它们加一倍帧数，
    # 实测 60 帧全幅 B：k=8 → 1.630px，k=16 → 1.600px（WBPP 1.609）。
    _refn = int(getattr(p, 'ref_k', 0) or 0)
    if _refn <= 0:
        _ch = str(_CFA.get('ch') or '')
        _refn = 8 if _ch in ('', 'G') else 16
    k = max(1, min(_refn, n))
    idx = np.argsort(np.nan_to_num(fwhms, nan=1e9))[:k]

    if star_grow_pct > 0:
        ref1 = float(np.nanmedian(lam1[idx]))
        d1 = np.maximum(lam1 - ref1, 0.0)
        L = np.sqrt(12.0 * d1)
        okL = np.isfinite(L)
        Lv = L[okL]
        if len(Lv):
            grow_auto = float(np.percentile(Lv, TRAIL_QUANTILE))
            star_grow = grow_auto / max(sigma_psf, 0.5) * star_grow_pct / 100.0
            w = np.where(okL, d1, 0.0)
            cv = float(np.nansum(w * np.cos(2 * pas)))
            sv = float(np.nansum(w * np.sin(2 * pas)))
            coh = float(np.hypot(cv, sv) / max(float(w.sum()), 1e-30))
            pa_mean = float(0.5 * np.degrees(np.arctan2(sv, cv))) % 180.0
            log(f'[拖线] 逐帧拖线长度 P50 {np.percentile(Lv, 50):.1f}px  '
                f'P{TRAIL_QUANTILE:.0f} {grow_auto:.1f}px  最大 {Lv.max():.1f}px  '
                f'{time.perf_counter() - t2:.1f}s')
            log(f'[拖线] 方向一致性 {coh:.2f}（1=同向）  平均 PA {pa_mean:.0f}°')
            log(f'[拖线] 自动外扩 {grow_auto:.1f}px ÷ σ_psf {sigma_psf:.2f}px = '
                f'{grow_auto / max(sigma_psf, 0.5):.2f}σ_psf；'
                f'按 {star_grow_pct:g}% 取 {star_grow:.2f}σ_psf')
        del lam1, pas, d1, L
    else:
        log('[拖线] 星点锐帧扩展已关闭（0%）')

    # ---- 参考像与 R 场 ----
    #   常规：最锐 k 帧在本机网格上取等权中位。
    #   drizzle：同样这 k 帧先各自按自己的位移落点、再取中位，所以它们
    #            即便彼此还没对齐，参考像也仍然是锐的。
    def _drizzle_workers() -> int:
        # 分带多进程：帧在磁盘 memmap 上，子进程按路径重建视图（不复制数据）
        try:
            import multiprocessing as _mp
            # 进程数按物理并发（线程数的一半）自适应：12 核 → 6（实测最优点），
            # 16c32t → 16，不再写死上限 8。内存带宽是真正的天花板，
            # 用户嫌慢时可用 --jobs 手动加，或 DWT_JOBS 覆盖。
            _cpu = int(_mp.cpu_count() or os.cpu_count() or 4)
            _n = max(2, min(CAP_DRIZZLE, max(2, _cpu // 2)))
            try:
                _pj = int(getattr(p, 'jobs', 0) or 0)
                if _pj > 0:
                    _n = max(1, min(32, _pj))
            except Exception:
                pass
            _jenv = os.environ.get("DWT_JOBS", "").strip()
            if _jenv:
                try:
                    _n = max(1, min(CAP_DRIZZLE, int(_jenv)))
                except Exception:
                    pass
        except Exception:
            _n = 1
        _env = os.environ.get('DWT_DRIZZLE_WORKERS', '')
        if _env.strip():
            try:
                _n = max(1, int(_env))
            except Exception:
                pass
        return _n

    drz_on = bool(getattr(p, 'drizzle', False))
    _aligned_in = bool(int(getattr(p, 'aligned', 0) or 0))
    # --frame / --crop 的参考栅格信息：在这里先初始化。drizzle 可能被下面的
    # “已对齐且 ×1”分支关掉，那时写盘前的切片判断若再引用就会 NameError。
    _frame_ref = str(getattr(p, 'frame', '') or '').strip()
    _ref_wh = None
    _inv_grid = 1.0
    if _aligned_in:
        # 旧行为：无条件关掉 drizzle → --drizzle-scale 被忽略，成品恒为 1×（bug）。
        # 现在保留用户要求的倍率：已对齐的帧位移≈0，走 drizzle 只是把样本投到 s× 网格；
        # s=1 时下面「已基本对齐 → 跳过重采样」会自动生效，等价于旧行为。
        _s_al = float(getattr(p, 'drizzle_scale', 1.0) or 1.0)
        if drz_on and _s_al > 1.001:
            log('[对齐] 输入已对齐模式：跳过配准修正（位移≈0），仍按 ×%g 输出网格做 drizzle'
                % _s_al)
        else:
            # ×1 且已对齐：直接加权叠加最锐（实测 2.55px vs drizzle 3.17px）
            if drz_on:
                log('[对齐] 输入已对齐且 ×1：不做 drizzle，直接加权叠加（最锐）')
            drz_on = False
    # 非 drizzle 路径也要有 T/fp：默认等价于关闭吞吐权重
    fp = float(getattr(p, 'flux_pow', FLUX_POW) or 0.0)
    T = None
    cov_map = None
    drz_s = float(getattr(p, 'drizzle_scale', 2.0) or 1.0)
    drz_p = float(getattr(p, 'drizzle_pix', 1.0) or 1.0)
    # 2× 网格 + pixfrac 0.8：液滴边长 1.6 输出像素，盖不满 2 像素的格子，会留下
    #   0.4px 的规则缝隙；帧数多、抖得散时被填平，帧数少（<10）时就显成周期网点。
    #   默认值就是 0.8，用户多半没动过 → 在 2× 上自动提到 1.0，并说明原因。
    if (drz_s >= 1.95 and drz_p < 1.0 and drz_p >= 0.7
            and not os.environ.get('DWT_PIXFRAC_KEEP')):
        log(f'[drizzle] 2× 输出：pixfrac {drz_p:g} → 1.0（{drz_p:g} 的液滴在 2× 网格上'
            f'留缝隙，帧数少时会出现周期性网点；想保留请设 DWT_PIXFRAC_KEEP=1）')
        drz_p = 1.0
        # CFA 的 wgpu 分支直接从 p.drizzle_pix 读，这里同步回去，避免「日志说 1.0、
        #   实际仍按 0.8 投液滴」的两套口径。
        try:
            setattr(p, 'drizzle_pix', drz_p)
        except Exception:
            pass
    # 取消 drizzle 的本意是「不出 2× 网格」，不是「不做配准」。旧代码直接走
    # 未配准的等权叠加（帧间位移没补），20 帧实测成品 FWHM 14.4px（单帧 4.3）
    # —— 等于把整晚的漂移糊进去了。这里改成：1× 网格 + pixfrac 1，仍做配准。
    if (not drz_on) and (not bool(int(getattr(p, 'aligned', 0) or 0))):
        log('[叠加] 未启用 drizzle 放大：按 1× 网格（pixfrac 1）配准后叠加')
        drz_on = True
        drz_s = 1.0
        drz_p = 1.0
    # 彩机 CFA：R/B 只有 1/4 像素有真值、G 只有 1/2。Siril 官方建议 CFA drizzle
    # 用 scale=1、pixfrac=1；再上采样等于要 4 倍的帧数，否则信噪比直接掉一半。
    # 实测（M81 12 帧、G 通道）：scale1/pix1 → 原生 FWHM 3.04px、N_eff 7.5；
    # scale2/pix1 → 3.02px、N_eff 4.7（分辨率没涨、等效帧数砍掉 37%）。
    if _CFA.get('pattern'):
        if drz_s > 1.0 + 1e-9:
            # 只提示、不再擅自改 pixfrac：pixfrac*scale 才是液滴边长，缩到 1/scale
            # 等于液滴只有 1 个输出像素，CFA 下会留下 3/4 的空洞（覆盖率掉、信噪比塌）。
            log('[CFA] 彩机 drizzle 提示：R/B 每 4 个像素才 1 个真值，'
                '2× 上采样约需 4 倍帧数；pixfrac 保持你填的 {0:g}'.format(drz_p))
    if _CFA.get('pattern') and not drz_on:
        # 彩机素材不走 drizzle 时，四个 CFA 相位只能按整像素落点，抖动信息全部丢失，
        # 星点周围会留下低于天空的坑（实测用户 78 帧 B：28.6% 的小星有 −2σ 像素，
        # 同样参数开 drizzle 后降到 3%）。这里直接强制开启，并说明原因。
        drz_on = True
        log('[CFA] 彩机素材必须做 drizzle：已自动启用（1× 网格，pixfrac 按你填的）——'
            '不 drizzle 时四个 CFA 相位只能按整像素落点，星点周围会出现低于天空的坑')
    drz_plan = None
    if drz_on:
        from drizzle import measure_shifts, plan_output, stack_sharp_reference
        t_sh = time.perf_counter()
        _emit('R 场', 0.10, '量帧间位移')
        _xz = str(getattr(p, 'xdrz', '') or '').strip()
        _gg = None
        if _xz:
            # 用 WBPP 写下的几何（.xdrz）当**初值**：PI 的配准只存变换不插值，
            # 但实测它的矩阵与它自己的 registered 图在翻转帧上就差 4~5px，
            # 直接落盘会整幅拖影，所以后面仍用本帧星点精修。
            from xdrz import transforms_for
            _gg = transforms_for(frames, _xz, log=log)
            _gok = np.asarray(_gg['ok'], bool)
            for _k in ('dx', 'dy', 'theta', 'scale'):
                _a = np.asarray(_gg[_k], np.float64).copy()
                _a[~_gok] = np.nan
                _gg[_k] = _a
            if _gok.any():
                _gt = np.asarray(_gg['theta'], np.float64)[_gok]
                _gk = np.asarray(_gg['scale'], np.float64)[_gok]
                log(f'[drizzle] 几何初值来自 WBPP .xdrz（逐帧用本帧星点精修）：'
                    f'θ {np.min(_gt):+.2f}°~{np.max(_gt):+.2f}°  '
                    f'k {np.min(_gk):.4f}~{np.max(_gk):.4f}')
        # ---- 配准结果缓存（跨运行复用）----
        # 读帧/量 FWHM/测位移与 A、排异 k、吞吐权重、drizzle 倍率都无关，
        # 同素材反复试参数时这几步纯属重复劳动。指纹含逐帧路径+大小+mtime、
        # 裁剪/中心/限帧/参考帧/星点数/对齐模式/xdrz，任何一项变了就重算。
        _ck = None
        sh = None
        _cv = getattr(p, 'cache', True)
        if isinstance(_cv, str):
            _cache_on = _cv.strip().lower() not in ('0', 'false', 'off', 'no', '')
        else:
            _cache_on = bool(_cv)
        if os.environ.get('DWT_CACHE_OFF', '') == '1':
            _cache_on = False
        if _cache_on and not _xz:
            try:
                from cache import frames_key, load_align, save_align, describe
                _ck_extra = {'crop': int(crop), 'center': tuple(center or ()),
                             'limit': int(limit or 0), 'seed': int(seed),
                             'n_star': int(n_star), 'aligned': 0}
                _ck = frames_key(frames, _ck_extra)
                sh = load_align(p, _ck)
                if sh is not None:
                    # 缓存只存了数组；下游还要用这几个派生量，就地补上
                    _fk = np.isfinite(np.asarray(sh['dx'], float))
                    _dxa = np.asarray(sh['dx'], float)[_fk]
                    _dya = np.asarray(sh['dy'], float)[_fk]
                    _tha = np.asarray(sh['theta'], float)[_fk]
                    sh['dx_range'] = (float(_dxa.min()), float(_dxa.max())) if _dxa.size else (0.0, 0.0)
                    sh['dy_range'] = (float(_dya.min()), float(_dya.max())) if _dya.size else (0.0, 0.0)
                    sh['theta_range'] = (float(_tha.min()), float(_tha.max())) if _tha.size else (0.0, 0.0)
                    sh['dist'] = np.zeros(len(sh['dx']), np.float64)
                    sh['n_bad'] = 0
                    sh['n_dist'] = 0
                    sh['scatter'] = []
                    log(f'[缓存] 命中上次的配准结果（{int(sh["n_keep"])}/{len(sh["dx"])} 帧，'
                        f'{describe(p)}）；跳过测位移。要重测请取消「复用配准缓存」'
                        f'或设 DWT_CACHE_OFF=1')
            except Exception as _e:
                sh = None
                log(f'[缓存] 读取失败，重测（不影响本次）：{_e}')
        if sh is None:
            sh = measure_shifts(_reg, stars, ref=int(seed), log=log, guess=_gg)
            # 参考帧回退：自动挑的参考帧若是异类（星少/最糊/取景偏），会把整组都判成配准失败。
            # 这里只在首轮参与率 < 50% 时，换星点最多的几个候选各试一次，取参与最多的那一次。
            _ref_fallback = None
            try:
                _n_all = int(len(sh['dx']))
                _kr = float(sh.get('n_keep', 0)) / max(1, _n_all)
                if _kr < 0.5 and len(stars) > 2:
                    _cnt = np.array([len(_s) for _s in stars], np.float64)
                    _order = [int(i) for i in np.argsort(-_cnt) if int(i) != int(seed)]
                    _best, _best_sh = int(sh.get('n_keep', 0)), sh
                    for _cand in _order[:3]:
                        try:
                            _sh2 = measure_shifts(_reg, stars, ref=_cand, log=None, guess=_gg)
                        except Exception:
                            continue
                        if int(_sh2.get('n_keep', 0)) > _best:
                            _best, _best_sh = int(_sh2.get('n_keep', 0)), _sh2
                            _ref_fallback = _cand
                    if _ref_fallback is not None:
                        log(f'[配准] 参考帧 #{int(seed)} 只对上 {int(sh["n_keep"])}/{_n_all} 帧，'
                            f'改参考帧 #{_ref_fallback}（对上 {_best} 帧）')
                        sh = _best_sh
            except Exception as _e:
                log(f'[配准] 参考帧回退失败（不影响本次）：{_e}')
            if _ck:
                try:
                    if save_align(p, _ck, sh):
                        log(f'[缓存] 已保存配准结果（{int(sh.get("n_keep", 0))}/{len(sh["dx"])} 帧，'
                            f'{describe(p)}），下次同素材可直接复用')
                    else:
                        log('[缓存] 保存失败（不影响本次叠加，下次仍会重测）')
                except Exception as _e:
                    log(f'[缓存] 写入失败（不影响本次）：{_e}')
        # ---- 卫星拖尾帧整帧剔除：直接从配准结果里划掉（keep / dx / dy / n_keep 一起改）----
        if _tdrop.any():
            _kp = np.asarray(sh['keep'], bool).copy()
            _ncut = 0
            for _i in np.nonzero(_tdrop)[0]:
                if _i < len(_kp) and bool(_kp[_i]):
                    _kp[_i] = False
                    _ncut += 1
                try:
                    sh['dx'][_i] = np.nan
                    sh['dy'][_i] = np.nan
                except Exception:
                    pass
                if 'theta' in sh:
                    try:
                        sh['theta'][_i] = np.nan
                    except Exception:
                        pass
            sh['keep'] = _kp
            sh['n_keep'] = int(np.count_nonzero(_kp))
            log(f'[筛选] 卫星拖尾帧不参与叠加：{_ncut}/{n} 帧  '
                f'（#{" #".join(str(int(i)) for i in np.nonzero(_tdrop)[0])}）')
        _drop = [int(i) for i in np.nonzero(~np.asarray(sh['keep'], bool))[0]]
        if _drop:
            log(f'  [叠加] 未参与的帧索引：{_drop[:40]}'
                + (f' 等 {len(_drop)} 帧' if len(_drop) > 40 else ''))
        log(f'[叠加] 参与 {sh["n_keep"]}/{len(sh['dx'])} 帧'
            + (f'（插值补 {sh["n_fill"]} 帧）' if sh.get('n_fill') else ''))
        dx = np.asarray(sh['dx'], np.float64)
        dy = np.asarray(sh['dy'], np.float64)
        th = np.asarray(sh['theta'], np.float64)
        kk = np.asarray(sh.get('scale', np.ones(len(dx))), np.float64)
        # 参考帧本身配准失败（星点太少/散度太大）时 dx[seed] 是 NaN：星表坐标
        # 没法投到输出网格，R 场、吞吐都会跟着塌。换一帧能配准的当参考帧，
        # 重建星表（星表位置本来就要求所有帧共用，换一次是安全的）。
        if not (np.isfinite(dx[int(seed)]) and np.isfinite(dy[int(seed)])):
            _ok = [i for i in range(n) if np.isfinite(dx[i]) and np.isfinite(dy[i])]
            if not _ok:
                raise SystemExit('所有帧的位移都没测出来，无法叠加（星点太少或素材不是同一片天）')
            _new = int(min(_ok, key=lambda i: (fwhms[i] if np.isfinite(fwhms[i]) else np.inf)))
            log(f'[位移] 参考帧 #{int(seed)} 未能配准，改用 #{_new} 当参考帧并重建星表')
            seed = _new
            stars = build_star_table(_reg.frame(seed), n_star=n_star)
            log(f'[星表] 帧 #{seed} 上重新选出 {len(stars)} 颗')
        _reg_ok = np.isfinite(dx) & np.isfinite(dy)   # 记下筛选前的配准结果
        _fm = float(getattr(p, 'fwhm_max', 0.0) or 0.0)
        if _fm > 0:
            _cut = np.nonzero(~(fwhms <= _fm))[0]
            for _i in _cut:
                dx[_i] = np.nan
                dy[_i] = np.nan
            if len(_cut):
                log(f'[筛选] 剔除 {len(_cut)}/{n} 帧（单帧 FWHM > {_fm:.2f}px），'
                    f'剩余 {n - len(_cut)} 帧参与叠加')
        # ---- 被排除的帧：给界面一份清单（谁被剔了、为什么）----
        _excluded_rows = []
        for _i in range(n):
            _why = []
            if _tdrop[_i]:
                _why.append('卫星拖尾（长 %.0fpx）' % _trail_info.get(_i, (0, 0.0))[1])
            if not _reg_ok[_i]:
                _why.append('配准失败（星点对不上或全星验证未过）')
            if _fm > 0 and np.isfinite(fwhms[_i]) and not (fwhms[_i] <= _fm):
                _why.append('单帧 FWHM %.2fpx > 上限 %.2fpx' % (fwhms[_i], _fm))
            if _why:
                _excluded_rows.append({'idx': int(_i),
                                       'file': (fs.names[_i] if getattr(fs, 'names', None)
                                                else '#%d' % _i),
                                       'reason': '；'.join(_why),
                                       'fwhm': (None if not np.isfinite(fwhms[_i])
                                                else round(float(fwhms[_i]), 3))})
        if _excluded_rows:
            try:
                _nu = int(np.count_nonzero(np.asarray(sh.get('keep', np.isfinite(dx)), bool)))
            except Exception:
                _nu = int(np.count_nonzero(np.isfinite(dx) & np.isfinite(dy)))
            _payload = {'n_used': _nu, 'n_total': int(n), 'items': _excluded_rows}
            try:
                log('[EXCLUDED] ' + json.dumps(_payload, ensure_ascii=False))
            except Exception:
                pass
            try:
                _csv = out_dir / (stem + '_excluded.csv')
                with open(_csv, 'w', encoding='utf-8-sig', newline='') as _fo:
                    _w = csv.writer(_fo)
                    _w.writerow(['序号', '文件名', '排除原因', '单帧FWHM(px)'])
                    for _r in _excluded_rows:
                        _w.writerow([_r['idx'], _r['file'], _r['reason'],
                                     '' if _r['fwhm'] is None else _r['fwhm']])
                log(f'[筛选] 被排除的帧已写入 {_csv.name}')
            except Exception as e:
                log(f'[筛选] 写排除清单失败：{e}')
        else:
            try:
                log('[EXCLUDED] ' + json.dumps({'n_used': int(n), 'n_total': int(n),
                                                'items': []}, ensure_ascii=False))
            except Exception:
                pass

        log(f'[drizzle] 帧间位移 dx {sh["dx_range"][0]:+.2f}~{sh["dx_range"][1]:+.2f}px  '
            f'dy {sh["dy_range"][0]:+.2f}~{sh["dy_range"][1]:+.2f}px  '
            f'视场旋转 {sh["theta_range"][0]:+.2f}°~{sh["theta_range"][1]:+.2f}°  '
            f'逐星残差 {_scat(sh):.3f}px  '
            f'{time.perf_counter() - t_sh:.1f}s')

        # ---- 输入本身已经对齐（例如 WBPP 的 registered 帧）就别再插值一次 ----
        #   实测同一批 registered 帧：走 drizzle 1x = 2.66px，直接加权叠加 = 2.47px，
        #   差的 0.19px 全是重采样核的代价；DWT 原版叠 registered 帧也不插值。
        if _aligned_in:
            # 输入已由 WBPP 注册：几何就是单位变换。不再用我们自己的星点匹配结果
            # （对已对齐帧只会引入几十分之一像素的抖动，叠加后等于多插值一次 ⇒ 更糊）。
            dx = np.zeros_like(np.asarray(dx, np.float64))
            dy = np.zeros_like(np.asarray(dy, np.float64))
            th = np.zeros_like(np.asarray(th, np.float64))
            kk = np.ones_like(np.asarray(kk, np.float64))
            log('[对齐] 已对齐输入：位移/旋转/尺度置为 0/1，不再二次配准'
                '（×1 时自动跳过重采样；>1 时只把样本投到 s× 网格）')
        # 画布必须先算出来，下面的「已对齐→跳过 drizzle」判据要用它的尺寸
        try:
            drz_plan = plan_output(fs.shape, dx, dy, drz_s, theta=th, k=kk)
        except Exception:
            drz_plan = None
        # --frame / --crop 的参考栅格信息：必须在 drizzle 分支之外初始化。
        # 否则 `--aligned 1` 且 ×1 时 drizzle 被关掉，这段赋值被跳过，
        # 写盘前的切片判断会 NameError: _frame_ref（2.7.119-D 的回归）。
        _frame_ref = str(getattr(p, 'frame', '') or '').strip()
        _inv_grid = 1.0
        _ref_wh = None
        _fin = np.isfinite(dx) & np.isfinite(dy)
        if drz_on and _fin.any():
            _mx = float(np.nanmax(np.abs(dx[_fin])))
            _my = float(np.nanmax(np.abs(dy[_fin])))
            _mt = float(np.nanmax(np.abs(th[_fin])))
            _mk = float(np.nanmax(np.abs(kk[_fin] - 1.0)))
            _cfa_now = bool(_CFA.get('raw') and _CFA.get('ch'))
            _fh, _fw = int(fs.shape[0]), int(fs.shape[1])
            _canvas_is_frame = (drz_plan is not None
                                and int(drz_plan.get('H', 0)) == _fh
                                and int(drz_plan.get('W', 0)) == _fw)
            if (_mx < 0.75 and _my < 0.75 and _mt < 0.05 and _mk < 0.002
                    and not _cfa_now and _canvas_is_frame):
                drz_on = False
                drz_plan = None
                log('[drizzle] 输入帧已基本对齐（最大位移 %.3f/%.3fpx，旋转 %.4f°，'
                    '尺度差 %.4f），跳过重采样直接加权叠加（省一次插值、更锐）'
                    % (_mx, _my, _mt, _mk))
            elif _cfa_now and _mx < 0.75 and _my < 0.75:
                log('[CFA] 帧间已对齐，但仍走 drizzle：真 CFA 必须按通道掩膜投样本')
            elif (not _cfa_now) and (not _canvas_is_frame) and _mx < 0.75 and _my < 0.75:
                log('[drizzle] 帧间已对齐，但输出画布 %dx%d ≠ 帧 %dx%d（亚像素余量），'
                    '仍走 drizzle 以保证 R 场与带内数据同一网格'
                    % (int(drz_plan.get('W', 0)), int(drz_plan.get('H', 0)), _fw, _fh))

        # ---- 吞吐 T_i：在配准后的真实星点位置上测，跨帧同一颗星才有可比性 ----
        #   形状（C）和噪声（S）都不回答"这帧收了多少光"：通透的夜晚信号高、
        #   薄云的夜晚信号低，而两者的噪声可能差不多。PI 的 PSF Signal Weight
        #   正是以这个量为核心。
        fp = float(getattr(p, 'flux_pow', FLUX_POW) or 0.0)
        T = np.ones(n, dtype=np.float64)
        if fp > 0:
            t_t = time.perf_counter()
            _emit('吞吐', 0.5, '测逐帧星点流量')
            jobs = [(i, stars, float(th[i]), float(kk[i]),
                     float(dx[i]) if np.isfinite(dx[i]) else 0.0,
                     float(dy[i]) if np.isfinite(dy[i]) else 0.0) for i in range(n)]
            fluxes = np.full((n, len(stars)), np.nan, dtype=np.float64)
            nwf = 0 if (not fs.parallelable or _no_parallel()) \
                else min(plan_workers(32 * 2 ** 20, '测吞吐', cap=CAP_FWHM), n)
            if nwf > 1:
                exf = ProcessPoolExecutor(max_workers=nwf, initializer=ini, initargs=ini_args)
                try:
                    for i, fl in exf.map(_task_flux, jobs, chunksize=1):
                        _check_cancel()
                        fluxes[i] = fl
                finally:
                    exf.shutdown(wait=True, cancel_futures=True)
            else:
                if 'mm' not in _MEM:            # 常驻内存路径：临时提供适配器
                    _h0, _w0 = fs.shape
                    _n0 = int(len(dx))
                    _MEM['mm'] = _ResidentStack(fs, _n0, _h0, _w0)
                    _MEM['off'] = np.zeros(int(_n0), np.float64)
                for i, fl in (_task_flux(j) for j in jobs):
                    _check_cancel()
                    fluxes[i] = fl
            # 逐星按「该星跨帧中位」归一 → 只留帧与帧之间的吞吐差，
            # 饱和、邻居污染、孔径吃光度这些静态偏差在归一里消掉。
            with np.errstate(invalid='ignore', divide='ignore'):
                ref = np.nanmedian(np.where(fluxes > 0, fluxes, np.nan), axis=0)
                ratio = fluxes / ref[None]
            ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
            good = np.isfinite(ratio).sum(axis=1) >= 5
            med = np.full(n, np.nan)
            med[good] = np.nanmedian(ratio[good], axis=1)
            if np.isfinite(med).sum() >= 3:
                med = med / np.nanmedian(med)
                T[:] = np.where(np.isfinite(med) & (med > 0), med, 1.0)
                log(f'[T] 逐帧星点流量（{FLUX_R:.0f}px 孔径扣平面，逐星归一，配准后位置）'
                    f' {np.nanmin(T):.3f}~{np.nanmax(T):.3f}  中位 1.000  '
                    f'→ 吞吐权重 T^{fp:g}（{np.nanmin(T) ** fp:.2f}~{np.nanmax(T) ** fp:.2f}）'
                    f'  {time.perf_counter() - t_t:.1f}s')
            else:
                fp = 0.0
                log('[T] 可用的星点流量太少，吞吐权重关闭')
        drz_plan = plan_output(fs.shape, dx, dy, drz_s, theta=th, k=kk)
        _frame_ref = str(getattr(p, 'frame', '') or '').strip()
        _inv_grid = 1.0          # 参考栅格相对原生像素的放大倍数（无参考图时为 1）
        _ref_wh = None           # 参考栅格的 (宽,高)；--crop 切片要拿它核对画布
        if _frame_ref and matchref is not None:
            try:
                _refv = np.asarray(
                    __import__('xisf_io').read_xisf(_frame_ref)[0].data, np.float64)
                try:
                    _MISC['ref_median'] = float(np.nanmedian(_refv))
                except Exception:
                    pass
                if _refv.ndim == 3:
                    # WBPP 合过色的 master 是 3 通道；matchref 只吃 2 维，
                    # 直接喂进去会抛异常，被外层兜住，「输出网格」就静默失效。
                    _refv = _refv.mean(axis=2)
                    log('[网格] 参考图是 RGB，按三通道平均折成亮度当网格参考')
                try:
                    _MISC['ref_median'] = float(np.nanmedian(_refv))
                except Exception:
                    pass
                _seed_img = np.asarray(_reg.frame(int(seed)), np.float64)
                _c0 = tuple(int(v) for v in _MISC.get('crop_origin', (0, 0)))
                _seed_align = _seed_img
                if _c0 != (0, 0):
                    # 裁剪图只有整幅的一小块，拿去和整幅 master 配准会解成
                    # 180° 翻转或错位（实测对账 0 对、成品缩在画布左上角）。
                    # 改用整幅 seed 帧配准，再把裁剪原点折算进 um。
                    try:
                        _full, _ = read_frame(frames[int(seed)])
                        _full = np.asarray(_full, np.float64)
                        if _full.ndim == 3:
                            _full = _full.mean(axis=2)
                        elif _CFA.get('ch') and _cfa is not None:
                            _full = _cfa.plane(
                                np.ascontiguousarray(_full, np.float32),
                                _CFA['pattern'], _CFA['ch']).astype(np.float64)
                        _seed_align = _full
                        log(f'[网格] 裁剪模式：用整幅 seed 帧配准，裁剪原点 '
                            f'({_c0[0]},{_c0[1]}) 折算进变换')
                    except Exception as _e_f:
                        log(f'[网格] 整幅 seed 读取失败（{type(_e_f).__name__}: '
                            f'{_e_f}），退回裁剪图配准')
                _, _fi = matchref.align(_seed_align, _refv, log=log)
                if _fi and _fi.get('k'):
                    _km, _thm = float(_fi['k']), float(_fi['theta'])
                    _inv = 1.0 / _km        # matchref 的 k 与 _xy_rows 的 k 互为倒数
                    _inv_grid = float(_inv)
                    _um = np.asarray(_fi['u'], np.float64)
                    if _c0 != (0, 0):
                        # u_master = k·R·c0 + u_crop ；这里 _fi 是整幅给的，
                        # 下面用的是裁剪坐标，先把 c0 的效果扣掉。
                        _rm0 = np.array([[np.cos(np.radians(float(_fi['theta']))),
                                          -np.sin(np.radians(float(_fi['theta'])))],
                                         [np.sin(np.radians(float(_fi['theta']))),
                                          np.cos(np.radians(float(_fi['theta'])))]])
                        _um = _um + (+1.0) * float(_km) * (_rm0 @ np.asarray(_c0, np.float64))
                    _rm = np.array([[np.cos(np.radians(_thm)),
                                     -np.sin(np.radians(_thm))],
                                    [np.sin(np.radians(_thm)),
                                     np.cos(np.radians(_thm))]])
                    _ux = -np.asarray(dx, np.float64)
                    _uy = -np.asarray(dy, np.float64)
                    _t = np.stack([_ux, _uy], -1)                 # 每帧的 u
                    _tc = _inv * (_t @ _rm.T) + _um                # 复合到 master 坐标
                    dx = -_tc[:, 0]
                    dy = -_tc[:, 1]
                    th = np.asarray(th, np.float64) + _thm
                    kk = np.asarray(kk, np.float64) * _inv
                    _h, _w = _refv.shape
                    # 亚像素收尾：seed 星点按最终变换投到 master，扣掉中位偏移
                    try:
                        from scipy.spatial import cKDTree as _KD
                        _sr = matchref.find_stars(_refv)
                        _ss = matchref.find_stars(_seed_img)
                        _si = int(seed)
                        _th2 = math.radians(float(th[_si]))
                        _c2, _s2 = math.cos(_th2), math.sin(_th2)
                        _k2 = float(kk[_si])
                        _ux2, _uy2 = -float(dx[_si]), -float(dy[_si])
                        _px = _k2 * (_c2 * _ss[:, 0] - _s2 * _ss[:, 1]) + _ux2
                        _py = _k2 * (_s2 * _ss[:, 0] + _c2 * _ss[:, 1]) + _uy2
                        _pp = np.stack([_px, _py], -1)
                        _d, _j = _KD(_sr).query(_pp, distance_upper_bound=6.0)
                        _m = np.isfinite(_d)
                        if _m.sum() >= 30:
                            _off = np.median(_sr[_j[_m]] - _pp[_m], axis=0)
                            dx = np.asarray(dx, np.float64) - _off[0]
                            dy = np.asarray(dy, np.float64) - _off[1]
                            log(f'[网格] 星点对账 {int(_m.sum())} 对，整体扣掉 '
                                f'({_off[0]:+.2f}, {_off[1]:+.2f})px')
                    except Exception as _e2:
                        log(f'[网格] 星点对账跳过（{type(_e2).__name__}: {_e2}）')
                    # 输出网格就是参考栅格本身：画布宽高/原点都照参考像素，
                    # 采样倍率固定 1（1 输出像素 = 1 参考像素 = _inv 倍原生像素）。
                    # 以前按 s = 倍率 / _inv 折算，2× 的 master 会被缩回原生 1×，
                    # 画布只有参考的一半，--crop 的切片坐标也就对不上参考像素。
                    _s_req = float(drz_s) if drz_on else 1.0
                    if drz_on:
                        if abs(_s_req - 1.0) > 1e-9:
                            log(f'[网格] --frame 模式下画布固定为参考栅格，'
                                f'--drizzle-scale {_s_req:g} 不生效（参考栅格已是最终采样）')
                        drz_plan = {'s': 1.0, 'ox': 0.0, 'oy': 0.0,
                                    'W': int(_w), 'H': int(_h)}
                        log(f'[网格] 输出画布 = 参考 {Path(_frame_ref).name}'
                            f' {_w}×{_h}（原点对齐参考像素，{_inv:.4f}× 原生像素，'
                            f'旋转 {_thm:+.2f}°）')
                    else:
                        _s_grid = _s_req / _inv if _inv else _s_req
                        _s_grid = max(0.05, min(8.0, _s_grid))
                        drz_plan = {'s': _s_grid, 'ox': 0.0, 'oy': 0.0,
                                    'W': int(round(_w * _s_grid)),
                                    'H': int(round(_h * _s_grid))}
                        log(f'[网格] 输出画布取参考 {Path(_frame_ref).name}'
                            f' {_w}×{_h} × {_s_grid:g} = {drz_plan["W"]}×{drz_plan["H"]}'
                            f'（放大 {_inv:.4f}× 旋转 {_thm:+.2f}°）')
                    _ref_wh = (int(_w), int(_h))
                else:
                    log(f'[网格] 参考图配准失败，退回包围盒画布：{_fi.get("error")}')
            except Exception as _e:
                log(f'[网格] 参考网格不可用（{type(_e).__name__}: {_e}），退回包围盒画布')
        # ---- --frame 与 --crop 同时给出：画布就是参考栅格，裁剪只是写盘前的
        #      数组切片（不换算仿射/原点），切片框在保存时算。
        if _frame_ref and int(crop or 0) > 0:
            log(f'[裁剪] --frame 模式：画布 = 参考栅格 {_ref_wh or "?"}，'
                f'写盘前切 {int(crop)}×{int(crop)}（超出画布会直接报错）')
        # ---- CFA：三通道必须共用同一个变换和同一张画布。各自配准会让画布差
        #      几个像素、通道之间整体错位（实测 R 12702 宽 vs G/B 12700 宽，
        #      合成出来是一堆分开的红绿蓝点）。这里存的必须是**帧块修正之后**的
        #      dx/dy（sh 里是 measure_shifts 的原始值，不含裁剪原点与 master 折算）。
        # ----
        if _CFA.get('pattern') or _CFA.get('want_share') or _CFA.get('auto_own'):
            _geo = _CFA.get('geo')
            if _geo is None:
                try:
                    _CFA['geo'] = {
                        'ch': _CFA['ch'], 'seed': int(seed),
                        'dx': np.array(dx, np.float64),
                        'dy': np.array(dy, np.float64),
                        'theta': np.array(th, np.float64),
                        'scale': np.array(kk, np.float64),
                        'dist': np.array(sh.get('dist', np.zeros(len(dx))), np.float64),
                        'keep': np.array(sh.get('keep', np.ones(len(dx), bool)), bool),
                        'n_bad': int(sh.get('n_bad') or 0),
                        'n_dist': int(sh.get('n_dist') or 0),
                        'scatter': np.asarray(sh.get('scatter', []), np.float64),
                        'plan': dict(drz_plan) if drz_plan else None,
                    }
                    log(f'[CFA] 已记录本通道几何（dx {np.nanmin(dx):+.1f}~'
                        f'{np.nanmax(dx):+.1f}px），后续通道复用')
                except Exception as _eg:
                    log(f'[CFA] 几何记录失败：{_eg}')
            elif not _CFA.get('auto_own'):
                seed = int(_geo['seed'])
                stars = build_star_table(_reg.frame(seed), n_star=n_star)
                dx = np.array(_geo['dx'], np.float64)
                dy = np.array(_geo['dy'], np.float64)
                th = np.array(_geo['theta'], np.float64)
                kk = np.array(_geo['scale'], np.float64)
                _dist = np.array(_geo['dist'], np.float64)
                _kp = np.array(_geo['keep'], bool)
                sh = {'keep': _kp, 'n_keep': int(np.count_nonzero(_kp)),
                      'n_fill': 0, 'n_bad': int(_geo.get('n_bad') or 0),
                      'n_dist': int(_geo.get('n_dist') or 0),
                      'dx': dx, 'dy': dy, 'theta': th, 'scale': kk,
                      'dist': _dist,
                      'scatter': np.asarray(_geo.get('scatter', []), np.float64),
                      'dx_range': (float(np.nanmin(dx)), float(np.nanmax(dx))),
                      'dy_range': (float(np.nanmin(dy)), float(np.nanmax(dy))),
                      'theta_range': (float(np.nanmin(th)), float(np.nanmax(th)))}
                if _geo.get('plan'):
                    drz_plan = dict(_geo['plan'])
                log(f'[CFA] 通道 {_CFA["ch"]} 复用通道 {_geo["ch"]} 的几何：'
                    f'参考帧 #{seed}、画布 {drz_plan["W"]}×{drz_plan["H"]}、'
                    f'{int(np.count_nonzero(_kp))}/{n} 帧参与')

        # ---- 逐通道独立配准（auto 判定有彩差）时，画布仍强制统一 ----
        # 各通道自己配准会得到差 1~2 像素的画布，合并时按最小画布裁就等于把
        # 整条通道平移了 1 像素（RGB 边缘出彩边）。这里保留各自的 dx/dy，但把
        # 画布换成首通道的，并把差异折进平移量：s*(u+ox) 不变。
        if (_CFA.get('pattern') and _CFA.get('auto_own') and not _CFA.get('want_share')
                and str(getattr(p, 'cfa_grid', 'shared') or 'shared').strip().lower() != 'own'
                and _CFA.get('geo') and drz_plan):
            _gp = _CFA['geo'].get('plan') or {}
            if _gp and abs(float(_gp.get('s', 0)) - float(drz_plan['s'])) < 1e-9:
                _sc = float(drz_plan['s']) or 1.0
                _ddx = (float(drz_plan['ox']) - float(_gp['ox'])) / _sc
                _ddy = (float(drz_plan['oy']) - float(_gp['oy'])) / _sc
                dx = np.asarray(dx, np.float64) - _ddx
                dy = np.asarray(dy, np.float64) - _ddy
                drz_plan = {'s': float(_gp['s']), 'ox': float(_gp['ox']),
                            'oy': float(_gp['oy']), 'W': int(_gp['W']),
                            'H': int(_gp['H'])}
                log(f'[CFA] 通道 {_CFA["ch"]} 画布统一到首通道 {_gp["W"]}×{_gp["H"]}'
                    f'（平移折进 dx/dy：{_ddx:+.2f}, {_ddy:+.2f}px）')

        # ---- 画布定稿后，把「用户填的倍率」和「实际画布倍率」统一 ----
        # 勾了输出网格时画布来自参考图（可能 1×），后面所有 σ_psf / 星点映射 /
        # 边缘过渡宽都要按实际画布算，否则会按 2× 折算，R 场被过度平滑。
        _drz_req = float(drz_s)
        if drz_plan:
            _s_real = float(drz_plan['s'])
            if abs(_s_real - _drz_req) > 1e-9:
                log(f'[网格] 实际画布倍率 {_s_real:g}×（你填的是 {_drz_req:g}×），'
                    '内部按实际倍率折算 R 场与星点')
            drz_s = _s_real

        # 星表是在 seed 帧（没有被变换的那一帧）上测的。喂到输出网格之前必须
        # 过一遍 seed 自己的变换：xdrz 模式下 dx/dy 是 PI 相对**它的**参考帧
        # 给的，seed 一般不在原点，甚至可能是 180° 翻转，直接 (x*s+ox) 会把
        # 星点映射到空天上（R 场 σ 会量成 12px 这种荒唐值）。
        def _seed_map(plan, scale):
            _si = int(seed)
            _ux0, _uy0 = -float(dx[_si]), -float(dy[_si])
            _th0, _k0 = float(th[_si]), float(kk[_si])
            _c0 = float(np.cos(np.radians(_th0)))
            _s0 = float(np.sin(np.radians(_th0)))
            out = []
            for (x, y) in stars:
                X = _k0 * (_c0 * x - _s0 * y) + _ux0
                Y = _k0 * (_s0 * x + _c0 * y) + _uy0
                out.append((int(round(X * scale + plan['ox'])),
                            int(round(Y * scale + plan['oy']))))
            return out
        # 输入是否还留着真实的亚像素相位：相位全同 ⇒ 已被对齐器插值过
        kp = np.asarray(sh.get('keep', np.ones(len(dx), bool)), bool)
        if sh.get('n_bad'):
            log(f'[drizzle] 有 {sh["n_bad"]} 帧位移无法直接测出（星点对不上/散度大），'
                '且前后没有可靠帧可插值，已排除、不参与叠加')
        ph = np.concatenate([np.mod(dx[kp], 1.0), np.mod(dy[kp], 1.0)])
        if ph.size >= 3 and float(np.std(ph)) < 0.03:
            log('[drizzle] 注意：各帧位移的亚像素相位几乎全同 ⇒ 输入已被对齐/插值过。'
                '此时 drizzle 只做一次无损失的重采样，采样率不会提高；'
                '真正的采样增益需要未对齐的原始子帧。')
        _dist = np.asarray(sh.get('dist', np.zeros(len(dx))), np.float64)
        if sh.get('n_dist'):
            _dzn = _dist[np.isfinite(dx)]
            if len(_dzn):
                log(f'[drizzle] 径向畸变 d2 {np.min(_dzn):+.4f}~{np.max(_dzn):+.4f}')
        _emit('R 场', 0.35, '重建参考像')
        # 参考像在 1/2 网格上重建：R 场是平滑掩膜（过渡沿 ~5.7px @2× 网格，
        # 半网格仍有 ~2.8px），半网格足以定出支撑边界，散射量降到 1/4。
        _ref_half = int(os.environ.get('DWT_REFHALF', '0') or 1)
        _pl2 = None
        if _ref_half and int(drz_plan['W']) >= 1024 and int(drz_plan['H']) >= 1024:
            _pl2 = dict(drz_plan)
            _pl2['s'] = float(drz_plan['s']) / 2.0
            _pl2['ox'] = float(drz_plan['ox']) / 2.0
            _pl2['oy'] = float(drz_plan['oy']) / 2.0
            _pl2['W'] = max(8, int(drz_plan['W']) // 2)
            _pl2['H'] = max(8, int(drz_plan['H']) // 2)
            # 帧先 2×2 盒式降采样再走同一条参考像代码：液滴边长仍是
            # pixfrac·s'（s'=s/2），对应的天区尺码与全分辨率一致，不会抹平。
            sref = stack_sharp_reference(_HalfFS(_reg), idx, dx / 2.0, dy / 2.0, _pl2, drz_p,
                                         theta=th, scale=kk, dist=_dist)
        else:
            sref = stack_sharp_reference(_reg, idx, dx, dy, drz_plan, drz_p,
                                         theta=th, scale=kk, dist=_dist)
        # R 场的 σ_psf 必须用"参考像自己"的 FWHM：参考像取的是最锐 k 帧，
        # 比全帧中位锐得多（实测 4.23px vs 2.6px）。用中位值会把窗和包络放大
        # 1.6 倍，细节区的 R 被压到 0.07%，分区加权等于没开。
        _px_scale = float(drz_plan.get('s', drz_s)) * float(_inv_grid)
        sigma_rf = float(sigma_psf) * _px_scale
        try:
            _sm = 0.5 if _pl2 is not None else 1.0
            _plm = _pl2 if _pl2 is not None else drz_plan
            _ho = max(16, int(round(16 * _px_scale * _sm)))
            _so = max(29, int(round(29 * _px_scale * _sm)))
            _so = _so | 1
            # 星点位是在参考栅格上的（_seed_map 里已乘过 _inv），换成这张图的
            # 画布像素要用它自己的 s；用 _px_scale*_sm 只有在 _inv=1 时才对。
            _st = _seed_map(_plm, float(_plm['s']))
            _fr = float(measure_fwhm(sref, _st, fwhm_guess=6.0 * _px_scale * _sm,
                                     half=_ho, fit_shape=_so))
        except Exception:
            _fr = float('nan')
        if _pl2 is not None:
            # 半网格上量的 FWHM 要换算回输出网格：×2；σ 再换回半网格给 R 场用：/2
            _fr = _fr * 2.0 if np.isfinite(_fr) else _fr
        if np.isfinite(_fr) and 0.5 * _px_scale < _fr < 40.0 * _px_scale:
            sigma_rf = _fr / 2.3548
            if _pl2 is not None:
                sigma_rf = sigma_rf / 2.0
            log(f'[drizzle] 参考像自量 FWHM {_fr:.2f}px（输出网格）→ R 场 σ_psf '
                f'{sigma_rf:.2f}px（中位帧折算值本应是 {float(sigma_psf) * _px_scale:.2f}px）')
        grow_rf = float(star_grow)
        R, rinfo = build_r_field(sref, sigma_rf, r_floor=r_floor,
                                 env_win_mult=env_win_mult,
                                 env_eps=env_eps, env_p=env_p,
                                 star_grow=grow_rf,
                                 r_blur=float(getattr(p, 'r_blur', 0.0) or 0.0),
                                 r_mode=str(getattr(p, 'r_mode', 'max') or 'max'),
                                 r_win=float(getattr(p, 'r_win', 0.0) or 0.0),
                                 r_k=float(getattr(p, 'r_k', 3.0) or 3.0),
                                 dec=_RDEC)
        log(f'[drizzle] 输出网格 {drz_plan["W"]}×{drz_plan["H"]}'
            f'（本机 ×{drz_s:g}，pixfrac {drz_p:g}）  '
            f'σ_psf {float(sigma_psf):.2f}→{sigma_rf:.2f}px  '
            f'R 的 2σ 过渡宽 {2 * sigma_rf:.1f}px')
    else:
        sref = np.median(np.stack([_reg.frame(int(i)) for i in idx], axis=0),
                         axis=0).astype(np.float32)
        R, rinfo = build_r_field(sref, sigma_psf, r_floor=r_floor,
                                 env_win_mult=env_win_mult,
                                 env_eps=env_eps, env_p=env_p,
                                 star_grow=star_grow, dec=_RDEC)
    if drz_on:
        log(f'[R] 参考像 = 最锐 {k} 帧逐帧落点后取中位（drizzle 网格）')
    else:
        log(f'[R] 参考像 = 最锐 {k} 帧等权中位')
    log(f'[R] 映射=包络(亮度归一)  口径=边缘强度  下限={r_floor:g}  '
        f'σ_psf={rinfo["sigma_psf"]:.2f}px  窗={rinfo["win"]}px  '
        f'T 尺度={rinfo["sig_hp"]:.4e}')
    if star_grow > 0:
        log(f'[R] 星点锐帧区自适应外扩：最亮星 {rinfo["grow_px"]:.1f}px'
            f'（{star_grow:g}σ_psf），全图均值 {rinfo["grow_px_mean"]:.2f}px'
            f'（R>0.5 占比 {rinfo["R_gt50"]:.2%}）'
            + (f'  [输出网格 ×{drz_s:g}]' if drz_on else ''))
    log(f'[R] 对比度 T 中位 {rinfo["T_p50"]:.2f}  P90 {rinfo["T_p90"]:.2f}  '
        f'P99 {rinfo["T_p99"]:.2f}  最大 {rinfo["T_max"]:.1f}')
    log(f'[R] 分区权重 R 均值 {rinfo["R_mean"]:.3f}  '
        f'R>0.5 占 {rinfo["R_gt50"]:.2%}  R>0.9 占 {rinfo["R_gt90"]:.2%}')
    _pl2 = locals().get('_pl2', None)   # 只有 drizzle 分支会定义它；非 drizzle 时在这里补上
    if _pl2 is not None:
        # 半网格的 R 放大回输出网格（双线性；R 是平滑掩膜，边缘量化 <1px）
        R = np.clip(ndimage.zoom(R, 2, order=1)[: int(drz_plan['H']), : int(drz_plan['W'])],
                    0.0, 1.0).astype(np.float32)
        rinfo['R_mean'] = float(R.mean())
        rinfo['R_gt50'] = float((R > 0.5).mean())
        rinfo['R_gt90'] = float((R > 0.9).mean())
        log(f'[R] 参考像在 1/2 网格重建（散射量 1/4），R 已放大回 {R.shape[1]}×{R.shape[0]}')
    del sref
    _emit('R 场', 1.0, f'R>0.5 占 {rinfo["R_gt50"]:.2%}')

    # ---- 阶段 C：统计层（W）× 几何层（a）----
    h, w = R.shape
    # CFA 权重共用：G 通道先跑，把 C/S/T 存下来给 R/B 用（等价于 PI 的「一图一套」）
    if _CFA.get('share') is not None and _CFA.get('ch') not in (None, 'G'):
        _C, _S, _T = _CFA['share']
        C, S = _C, _S
        T = _T
        log('[CFA] 权重共用：C/S/T 取 G 通道的量（自动/共用模式）')
    elif _CFA.get('want_wshare') and _CFA.get('ch') == 'G':
        _CFA['share'] = (np.array(C, copy=True), np.array(S, copy=True),
                         None if T is None else np.array(T, copy=True))
    if drz_on:
        # ---- drizzle：每帧按自己的位移落点，输出网格逐像素排异，分母 ΣW·c ----
        _gm = str(getattr(p, 'gpu', 'auto') or 'auto').lower()
        if _S_field is not None and _gm != 'cpu':
            # 逐格 S 目前只接在 CPU 的逐块加权里；GPU canvas 内核用的是标量 lnS
            log('[局部权重] GPU 内核暂不支持逐格 S，本次 drizzle 改走 CPU 逐块加权')
            _gm = 'cpu'
        if _gm == 'cpu':
            os.environ['DWT_GPU'] = '0'
        elif _gm == 'gpu':
            os.environ['DWT_GPU'] = '1'
        else:
            os.environ.pop('DWT_GPU', None)

        from drizzle import stack_drizzle
        # 分带大小：num/cov 两个 (n, bh, w) float32 缓冲不超过 ~512 MiB
        _nw = max(1, _drizzle_workers())
        # GPU 走显存，缓冲区预算可以大得多；带越宽，旋转帧的斜条带浪费越小
        # （θ≈176° 时一行带要读 band + w·|sinθ| 行，48 行 → 10 倍浪费，256 行 → 2.7 倍）
        _gpu_ok = False
        try:
            import gpu_dz as _g
            _gpu_ok = bool(str(os.environ.get('DWT_GPU', 'auto')).lower() not in ('0', 'off', 'false', 'no')
                           and _g.available())
            if _gpu_ok and str(os.environ.get('DWT_DZ_KERNEL', '')) == 'lanczos':
                _gpu_ok = False
                log('[drizzle] Lanczos 液滴核目前只有 CPU 版，本组走 CPU')
        except Exception:
            _gpu_ok = False
        if _gpu_ok:
            # 显存预算按实际可用量缩放：num/cov 两个 (n,bh,w) float32 缓冲。
            # 写死 3 GB 会让 4 GB 显存的小卡（GTX 1650 Ti 这类）直接 OOM。
            _free = 0
            try:
                _free = int(_g.free_mem())
            except Exception:
                _free = 0
            # 不设固定上限：按可用显存的固定比例（70%）算，大卡自然吃满。
            _budget = int(max(256 * 2**20, _free * 0.70)) if _free > 0 else (4096 * 2**20)
            if _free > 0:
                log(f'[drizzle] 显存可用 {_free // 2**20} MB，本带预算 '
                    f'{_budget // 2**20} MB（可用量的 70%）')
            _rows = _budget / max(1.0, 2.0 * n * w * 4)
            band = int(max(64, min(512, _rows)))
        else:
            # CPU 侧实测：带高 32/48/64/160 行（6 进程、M81 60 帧 1600²×2）=
            # 35.7/38.1/39.5/58.8 s —— 带越宽，每带要来回扫的 (n,bh,w) 缓冲越大，
            # 反而更慢；260 帧全幅时又不允许太宽（每带 2*n*bh*w*4 字节）。
            # 于是按内存公式算上界，再夹到 32~40 行。
            _rows = (2048 * 2**20) / max(1.0, 2.0 * n * w * 4 * _nw)
            band = int(max(32, min(40, _rows)))
            _env_band = str(os.environ.get('DWT_BAND', '') or '').strip()
            if _env_band:
                try:
                    band = max(8, int(_env_band))
                    log(f'[drizzle] 带高由 DWT_BAND 指定：{band} 行（实验用）')
                except Exception:
                    pass
        # 带宽按"单带 num/cov 各约 750MB"取；带太窄时旋转帧的斜条带
        # （横跨宽度 w 时行偏移约 w·sinθ）占比过高：34 行时斜条带被放大
        # 13×，96 行降到 2.7×
        # ---- 位移表补齐到帧数：某些素材里配准阶段会少回一帧，长度不一致
        #      会让 stack_drizzle 按帧索引时越界。补成 NaN 的帧等于不参与。
        try:
            _nd = len(np.asarray(dx))
            if _nd != n:
                log(f'[配准] 位移表 {_nd} 个 / 帧 {n} 帧，已补齐（缺的帧不参与）')
                _dw = np.full(n, np.nan)
                _dw[:_nd] = np.asarray(dx, np.float64)[:_nd]
                dx = _dw
                _dw = np.full(n, np.nan)
                _dw[:_nd] = np.asarray(dy, np.float64)[:_nd]
                dy = _dw
                _tw = np.zeros(n)
                _tw[:_nd] = np.asarray(th, np.float64)[:_nd]
                th = _tw
                _kw = np.ones(n)
                _kw[:_nd] = np.asarray(kk, np.float64)[:_nd]
                kk = _kw
                if _dist is not None and len(np.asarray(_dist)) != n:
                    _dw = np.zeros(n)
                    _dw[:_nd] = np.asarray(_dist, np.float64)[:_nd]
                    _dist = _dw
        except Exception as _e_len:
            log(f'[配准] 位移表补齐失败（{type(_e_len).__name__}: {_e_len}），继续')
        t2 = time.perf_counter()
        _cfa_sel = None
        if _CFA.get('raw') and _CFA['ch']:
            _cfa_sel = (_cfa.mask2(_CFA['pattern'], _CFA['ch']),
                        int(_CFA.get('origin', (0, 0))[1]) % 2,
                        int(_CFA.get('origin', (0, 0))[0]) % 2)
            log('[CFA] 真 CFA drizzle：只投 %s 的原始样本（不插值）' % _CFA['ch'])
        # ---- 后端选择 ----
        #   gpu=auto → 优先 wgpu（跨平台：Vulkan / Metal / D3D12），用不了退回 CPU
        #   gpu=wgpu → 强制 wgpu（失败报错，不静默退回 CPU）
        #   gpu=cpu  → 纯 CPU
        #   兼容：旧配置里的 gpu=gpu（原 CUDA/CuPy 路径）按 wgpu 处理，CuPy 已于 2.7.73 移除
        #   环境变量 DWT_GPU_WGPU=1/0 可强制开/关（调试用）
        _gpu_sel = str(getattr(p, 'gpu', 'auto') or 'auto').strip().lower()
        if _gpu_sel == 'gpu':
            log('[GPU] 选项「只用 GPU（CUDA）」已随 CuPy 移除，改按 wgpu 处理')
            _gpu_sel = 'wgpu'
        _env_wgpu = os.environ.get('DWT_GPU_WGPU', '').strip()
        _want_wgpu = (_env_wgpu == '1') or (_gpu_sel == 'wgpu') or (
            _gpu_sel == 'auto' and _env_wgpu != '0')
        _wgpu_ok = False
        if _want_wgpu and _env_wgpu != '0':
            try:
                from gpu_wgpu import diagnose as _wg_diag, reason as _wg_reason
                _wgpu_ok, _wg_why = _wg_diag()
                if _wgpu_ok:
                    log('[wgpu] 通用 GPU 后端：%s' % _wg_why)
                elif _gpu_sel == 'wgpu':
                    log('[wgpu] 用不了，本组改走其它后端（CUDA 或 CPU）。原因：%s' % _wg_why)
                    if '导入 wgpu 失败' in str(_wg_why):
                        log('[wgpu] 提示：包里的 gpudeps/ 目录要跟主程序放在一起；'
                            '双击「检测-wgpu.bat」可看完整诊断')
            except Exception as _e_wg:
                _wgpu_ok = False
                if _gpu_sel == 'wgpu':
                    log('[wgpu] 后端加载失败（%s: %s），本组改走其它后端'
                        % (type(_e_wg).__name__, _e_wg))
        # 真 CFA：wgpu 走「最近同色样本」（等价于 pixfrac*scale≈1 的 1×1 液滴，
        # 实测与 CPU 的 scatter drizzle 差别 <5%）；想要关掉可设 DWT_WGPU_CFA_OFF=1。
        _cfa_off = os.environ.get('DWT_WGPU_CFA_OFF', '').strip() == '1'
        if _wgpu_ok and _cfa_sel is not None and _cfa_off:
            log('[wgpu] 真 CFA 已按 DWT_WGPU_CFA_OFF=1 关闭，本组走 CUDA/CPU')
            _wgpu_ok = False
        elif _wgpu_ok and _cfa_sel is not None:
            _pf = float(getattr(p, 'drizzle_pix', 1.0) or 1.0)
            # CFA：同色格点间隔 2 源像素，2× 输出时样本间距 4 输出像素；
            # 液滴边长 = pixfrac*s*2，pixfrac<1 必然在网格相位上留缝（构造性问题，
            # 不是「帧少才显」）。所以真 CFA 下无条件抬到 1.0。
            if _pf < 1.0 and not os.environ.get('DWT_CFA_PIX_STRICT'):
                log(f'[CFA] 液滴边长按同色格点铺满：pixfrac {_pf:g} → 1.0'
                    f'（低于 1 会在 2× 网格上留空洞；想强制保留设 DWT_CFA_PIX_STRICT=1）')
                _pf = 1.0
            _sc = float(drz_plan.get('s', 1.0)) if drz_plan else 1.0
            _near = os.environ.get('DWT_WGPU_CFA_NEAR', '').strip() == '1'
            if _near and abs(_pf * _sc - 1.0) <= 0.02:
                log('[wgpu] 真 CFA：最近同色样本（DWT_WGPU_CFA_NEAR=1，实验路径）')
            else:
                log('[wgpu] 真 CFA：液滴散射（定点原子加，pixfrac×倍率=%.2f）'
                    % (_pf * _sc))
        _use_wgpu = _wgpu_ok
        # 灰度素材：收集式（stack_pws）与 CPU 差 2~3%，改用液滴散射（float 原子画布），
        # 60 帧实测 corr 0.998、比 CuPy 快 22%。收集式只保留给彩机最近邻路径。
        if _use_wgpu and _cfa_sel is None:
            log('[wgpu] 灰度素材走液滴散射后端（收集式只用于彩机）')
            _use_wgpu = False
        _wmap = None
        _PF = {'mul': 1.0}      # 两遍 drizzle：第二遍把液滴放大，只补未覆盖像素
        _lsp_acc = None
        _lsp = False
        def _do_stack():
            _lsp_acc = None
            _lsp = False
            if _use_wgpu:
                _H2, _W2 = int(drz_plan['H']), int(drz_plan['W'])
                _yy = np.clip(((np.arange(_H2) - float(drz_plan['oy'])) / float(drz_plan['s'])).astype(int), 0, h - 1)
                _xx = np.clip(((np.arange(_W2) - float(drz_plan['ox'])) / float(drz_plan['s'])).astype(int), 0, w - 1)
                # R 已经是输出画布尺寸（build_r_field 在画布网格上构建，stack_drizzle
                # 也按 (H, W) 校验）。早先按「画布→本机」映射去索引它，s≠1 时取到的是
                # 画面另一端的数据，导致 A 权重在过渡区系统性偏错（corr 0.985 vs 0.999）。
                _R2 = R if R is not None else None
                if _cfa_sel is not None:
                    _pf = float(getattr(p, 'drizzle_pix', 1.0) or 1.0)
                    if _pf < 1.0 and not os.environ.get('DWT_CFA_PIX_STRICT'):
                        _pf = 1.0   # CFA：液滴至少等于样本足迹（更小会缺相位）
                    _pf *= float(_PF['mul'])
                    _pd = _pf * float(drz_plan.get('s', 1.0))
                    # 默认全部走散射：最近邻在 P=1 且带旋转时会丢掉液滴的分数覆盖
                    # （20 帧实测 corr 仅 0.92），散射与 CPU 逐像素等价（corr 0.9997）
                    if os.environ.get('DWT_WGPU_CFA_NEAR', '').strip() == '1' and abs(_pd - 1.0) <= 0.02:
                        from gpu_wgpu_pws import stack_pws_cfa as _sgc
                        return _sgc(fs, C, S, dx, dy, th, kk, drz_plan, A, B,
                                    pattern=_CFA['pattern'], channel=_CFA['ch'],
                                    sky=0.0, R=_R2, rej_k=rej_k, log=log, n=n,
                                    pixfrac=_pf)
                    try:
                        # 画布常驻路径已在上面的 DWT_CANVAS 分支处理过；这里只保留
                        # 「关掉画布 + 显式开启」的实验组合，其余情况静默跳过，
                        # 避免日志里出现误导性的「float 原子散射不可用」。
                        if os.environ.get('DWT_CANVAS', '1').strip() != '0'                                 or os.environ.get('DWT_FATOM_ON', '').strip() != '1':
                            raise _Skip('canvas 路径已处理或实验路径未开启')
                        from gpu_wgpu_cfa_fatom import stack_cfa_fatom as _sgf
                        return _sgf(fs, C, S, dx, dy, th, kk, drz_plan, A, B,
                                    rej_k=rej_k, rej_gate=rej_gate, R=_R2,
                                    pixfrac=_pf, log=log, T=T, fp=fp,
                                    pattern=_CFA['pattern'], channel=_CFA['ch'],
                                    Ctiles=C_tiles)
                    except Exception as _e_fa:
                        if os.environ.get('DWT_WGPU_DEBUG') == '1':
                            import traceback as _tb
                            _tb.print_exc()
                        if os.environ.get('DWT_FATOM_ON', '').strip() == '1':
                            log('[wgpu][fallback] reason=fatom:%s: %s'
                                % (type(_e_fa).__name__, _e_fa))
                        if not isinstance(_e_fa, _Skip):
                            log('[wgpu] float 原子散射不可用（%s: %s），退回分桶'
                                    % (type(_e_fa).__name__, _e_fa))
                    try:
                        if os.environ.get('DWT_WGPU_BUCKET_OFF', '').strip() == '1':
                            raise RuntimeError('bucket disabled by env')
                        _canv = os.environ.get('DWT_CANVAS', '1').strip()
                        # 细节区噪声权重 beta（--detail-snr）：GPU 内核从环境读
                        os.environ['DWT_DETAIL_SNR'] = str(_detail_beta(p))
                        if _canv != '0':
                            try:
                                from gpu_wgpu_cfa_fatom import (
                                    stack_cfa_fatom as _sgg, available as _fa_ok)
                                _Hc, _Wc = int(drz_plan['H']), int(drz_plan['W'])
                                _need_mb = _Hc * _Wc * 4.0 * 10 / (1024.0 * 1024.0)
                                # 帧数 × 一帧条带超过单绑定上限时，画布后端要开环形缓冲，
                                # 那条路径在 51 帧全幅上产出周期性重复（实测），先禁用：
                                # 让分桶后端（已验证）接手，宁可慢也不能出错图。
                                _plane_mb = float(fs.shape[0]) * float(fs.shape[1]) * 4.0 / (1024.0 * 1024.0)
                                _ring_need = len(dx) * _plane_mb
                                _limit_mb = 2047.0 * 0.9
                                # 2.7.34-C：环形缓冲已修对（每趟重新上传，见 gpu_wgpu_cfa_fatom.py），
                                # 帧数超过单绑定上限也能走画布后端；DWT_CANVAS_RING=0 可强制回分桶。
                                if _ring_need > _limit_mb and os.environ.get('DWT_CANVAS_RING', '1').strip() == '0':
                                    log('[wgpu] 画布环形缓冲被 DWT_CANVAS_RING=0 关闭（%d 帧 × %.0f MB > %.0f MB），'
                                        '改走分桶后端' % (len(dx), _plane_mb, _limit_mb))
                                    raise _Skip('canvas ring disabled by env')
                                if _fa_ok() and (_canv == '1' or _need_mb <= 6000.0):
                                    return _sgg(fs, C, S, dx, dy, th, kk, drz_plan,
                                                A, B, rej_k=rej_k, rej_gate=rej_gate,
                                                R=_R2, pixfrac=_pf, band_rows=_Hc,
                                                log=log, T=T, fp=fp,
                                                pattern=_CFA['pattern'],
                                                channel=_CFA['ch'],
                                                Ctiles=C_tiles)
                            except Exception as _e_cv:
                                if os.environ.get('DWT_WGPU_DEBUG') == '1':
                                    import traceback as _tb
                                    _tb.print_exc()
                                log('[wgpu][fallback] reason=canvas:%s: %s'
                                    % (type(_e_cv).__name__, _e_cv))
                        from gpu_wgpu_cfa_bucket import stack_cfa_bucket as _sgb
                        return _sgb(fs, C, S, dx, dy, th, kk, drz_plan, A, B,
                                    rej_k=rej_k, rej_gate=rej_gate, R=_R2,
                                    pixfrac=_pf, log=log, T=T, fp=fp,
                                    pattern=_CFA['pattern'], channel=_CFA['ch'])
                    except Exception as _e_bk:
                        _emsg = '%s: %s' % (type(_e_bk).__name__, _e_bk)
                        if os.environ.get('DWT_WGPU_DEBUG') == '1':
                            import traceback as _tb
                            _tb.print_exc()
                        log('[wgpu][fallback] reason=bucket:%s' % _emsg)
                        log('[wgpu] 分桶后端不可用（%s），退回 CPU 重算本组' % _emsg)
                        if os.environ.get('DWT_WGPU_CFA_SCATTER', '').strip() == '1':
                            from gpu_wgpu_cfa_scatter import stack_cfa_scatter as _sgs
                            return _sgs(fs, C, S, dx, dy, th, kk, drz_plan, A, B,
                                        pattern=_CFA['pattern'], channel=_CFA['ch'],
                                        R=_R2, pixfrac=_pf, rej_k=rej_k, log=log, n=n)
                        raise RuntimeError('wgpu 真 CFA 后端不可用：' + _emsg)
                from gpu_wgpu import stack_pws as _sg
                return _sg(fs, C, S, dx, dy, th, kk, drz_plan, A, B,
                           sky=0.0, R=_R2, log=log, n=n)
            # 并行进程数要按画布内存收敛：每个子进程一趟要持有整幅输出（float32），
            # 全幅 2× 的画布是 ~417 MB/进程，8 进程光输出缓冲就 3.3 GB。
            try:
                _cv_mb = float(drz_plan['W'] * drz_plan['H'] * 4) / (1024.0 * 1024.0)
                # 每进程要持有整幅输出（num/cov 两份 + 临时），按 2.2 倍画布估；
                # 上限用「当前可用内存的 60%」，而不是写死 2048 MB——
                # 后者在 32 GB 机器上会把手指数从 8 压到 2，白丢 4 倍速度。
                _bud_mb = min(8192.0, max(512.0, _avail_bytes() / 2**20 * 0.60))
                _cap = max(1, int(_bud_mb // max(1.0, _cv_mb * 2.2)))
                _nw = max(1, min(int(_drizzle_workers()), _cap))
                if _S_field is not None:
                    _nw = 1
                if _nw < int(_drizzle_workers()):
                    log('[drizzle] 画布 %.0f MB，并行进程从 %d 降到 %d（避免内存打爆）'
                        % (_cv_mb, int(_drizzle_workers()), _nw))
            except Exception:
                _nw = int(_drizzle_workers())
            # ---- wgpu 后端：没有 CUDA 时不要直接掉到 CPU（跨平台 Vulkan/Metal/D3D12）----
            if _cfa_sel is None and str(os.environ.get('DWT_GPU', 'auto')).lower() not in (
                    '0', 'off', 'false', 'no'):
                # 2.7.73：wgpu 作为主 GPU 后端（跨平台、60 帧实测比 CuPy 还快 22%），
                # CuPy 只在 wgpu 不可用时兜底；DWT_CUPY_FIRST=1 可恢复旧优先序。
                _cupy_first = os.environ.get('DWT_CUPY_FIRST', '') == '1'
                if not _cupy_first:
                    try:
                        import gpu_wgpu_cfa_fatom as _wmod
                        if _wmod.available():
                            log('[drizzle] 叠加走 GPU（wgpu）：整张画布')
                            _r = _wmod.stack_cfa_fatom(
                                fs, C, S, dx, dy, th, kk, drz_plan, A, B,
                                rej_k=rej_k, rej_gate=rej_gate, R=R,
                                pixfrac=float(drz_p),
                                band_rows=int(max(64, drz_plan.get('H', 1024))),
                                T=T, fp=float(fp), log=log,
                                pattern=None, channel=None,
                                # 黑白素材永远不用满月帧补偿；显式传 0 以杜绝环境变量串味
                                beta=float(_detail_beta(p)),
                                cap=float(getattr(p, 'max_share', 0.0) or 0.0),
                                # 逐瓦片清晰度（可选）：让每个区域各自挑最锐的帧
                                Ctiles=C_tiles,
                                # 帧域验证需要着色器实际用的 2×3 变换（p[0..5]）
                                want_p=True)
                            try:
                                # cropcover.crop_box 要的是「每像素被多少帧覆盖」的计数图；
                                # fatom 返回的 cov 是权重和（< 帧数、边缘更小），直接当计数
                                # 会把自动裁边压成几十像素见方。这里按「有样本=满覆盖」构造计数，
                                # 边缘只 1~2 帧覆盖处也给满值，结果是裁得略宽松（安全方向）。
                                _st = _r.get('stack')
                                _n = float(len(dx))
                                _r['ncov'] = ((np.isfinite(_st) & (_st > 0)).astype(np.float32) * _n)
                                _r.setdefault('rej_rate', 0.0)
                                _r.setdefault('wmax', 0.0)
                            except Exception:
                                pass
                            return _r
                    except Exception as _e_w:
                        log('[drizzle] wgpu 不可用（%s: %s），改走 CPU'
                            % (type(_e_w).__name__, _e_w))
            return stack_drizzle(fs, R, C, S, dx, dy, drz_plan, A, B, rej_k,
                                rej_gate, float(fs.sky0), drz_p, band,
                                theta=th, scale=kk, dist=_dist, workers=_nw,
                                T=T, fp=fp, rej_mode=str(getattr(p, 'rej_mode', 'winsor')),
                                kernel=str(getattr(p, 'kernel', 'box') or 'box'),
                                psf_target=float(getattr(p, 'psf_target', 50.0) or 50.0),
                                fwhms=fwhms,
                                native_shape=fs.shape, log=log, emit=_emit,
                                cancel=_check_cancel,
                                sharp_w=float(getattr(p, 'sharp_w', 0.0) or 0.0),
                                cfa_sel=_cfa_sel, wmap=_wmap,
                                beta=float(getattr(p, 'detail_snr', 0.0) or 0.0))
        try:
            try:
                r_d = _do_stack()
            except Exception as _e_wgpu:
                if not _use_wgpu:
                    raise
                if os.environ.get('DWT_WGPU_DEBUG') == '1':
                    import traceback as _tb
                    _tb.print_exc()
                log('[wgpu] 出错，改用 CPU 重算本组：%s: %s'
                    % (type(_e_wgpu).__name__, _e_wgpu))
                _use_wgpu = False
                r_d = _do_stack()
            try:
                _lsp_pmap = r_d.get('p') if isinstance(r_d, dict) else None
            except Exception:
                _lsp_pmap = None
            if bool(int(getattr(p, 'two_pass', 0) or 0)):
                _wmap = _local_sharp_maps(fs, np.arange(n), drz_plan, 64, log)
                if _wmap is not None:
                    r_d = _do_stack()
            # ---- 自适应第二遍：只补第一遍没有覆盖的像素 ----
            # 第一遍用样本足迹（锐），空洞=抖动没覆盖到的相位；
            # 第二遍把这些像素的液滴放大到 2×间距（仍是同一批真实样本，
            # 只是足迹更宽），把缺的相位补上。有覆盖的像素一律保留第一遍，
            # 所以全图锐度不受影响。
            if _cfa_sel is not None and os.environ.get('DWT_CFA_TWOPASS', '1') != '0':
                _c1 = r_d.get('ncov', r_d.get('cov'))
                if _c1 is not None and _c1.shape == r_d['stack'].shape:
                    _hole = ~(_c1 > 0)
                    _H_, _W_ = _hole.shape
                    _b = 50
                    _inner = _hole.copy()
                    if _H_ > 2 * _b and _W_ > 2 * _b:
                        _inner[: _b, :] = False; _inner[-_b:, :] = False
                        _inner[:, : _b] = False; _inner[:, -_b:] = False
                    _in_frac = float(_inner.sum()) / float(max(1, _inner.size))
                    if _in_frac > 0.0002:
                        log('[CFA] 两遍 drizzle：第一遍内部未覆盖 %.3f%% → 只在空洞处'
                            '用 2× 液滴补一遍（同一批真实样本，不伪造）' % (100 * _in_frac))
                        _PF['mul'] = 2.0
                        try:
                            _r2 = _do_stack()
                            _c2 = _r2.get('ncov', _r2.get('cov'))
                            if _c2 is not None and _c2.shape == _c1.shape:
                                _fix = _hole & (_c2 > 0)
                                _n_fix = int(_fix.sum())
                                if _n_fix:
                                    _s1 = r_d['stack']
                                    _s1[_fix] = _r2['stack'][_fix]
                                    if r_d.get('ncov') is not None:
                                        r_d['ncov'] = np.where(_fix, _c2, _c1)
                                    if r_d.get('cov') is not None:
                                        r_d['cov'] = np.where(_fix, _c2, _c1)
                                log('[CFA] 两遍合并：补上 %d 个像素（内部 %d），'
                                    '其余保持第一遍锐度' % (_n_fix, int((_fix & _inner).sum())))
                        except Exception as _e2:
                            log('[CFA] 第二遍补洞失败（%s），保留第一遍结果' % _e2)
                        finally:
                            _PF['mul'] = 1.0
        finally:
            if not getattr(fs, '_shared', False):
                fs.close(keep=keep_frames)
        globals()['_GPU_WMAX'] = True
        Stk = r_d['stack']
        neff = r_d['neff']
        cov_map = r_d.get('ncov')
        _lspd = r_d.get('lsp')
        if _lspd:
            try:
                import lsp as _lsp_mod
                _cs = _lspd.get('cnte')
                _S0 = Stk if Stk.ndim == 2 else None
                if _S0 is not None and _cs is not None and _cs.size == _S0.size:
                    _ht = _lspd.get('hits')
                    _rate = (_cs / np.maximum(_ht if _ht is not None else 1.0, 1.0)).reshape(_S0.shape)
                    if os.environ.get('DWT_LSP_DUMP'):
                        np.save(os.environ['DWT_LSP_DUMP'], _rate.astype(np.float32))
                    _M = _lsp_mod.detect(_rate, 1, R=R, logf=log)
                    if _M is not None and bool(_M.any()):
                        _cw = (cov_map if cov_map is not None else r_d.get('cov'))
                        _cw = np.asarray(_cw, np.float32)
                        _we = _lspd['we'].reshape(_S0.shape)
                        _wde = _lspd['wde'].reshape(_S0.shape)
                        _den = _cw - _we
                        _ok = _M & (_den > 0.25 * np.maximum(_cw, 1e-12))
                        if bool(_ok.any()):
                            _num = _S0 * _cw - _wde
                            _corr = _num / np.maximum(_den, 1e-12)
                            _S0[_ok] = _corr[_ok].astype(np.float32)
                            log('[LSPR] 大尺度排异：结构 %d 像素（%.3f%%），修正 %d'
                                % (int(_M.sum()), 100.0 * float(_M.mean()), int(_ok.sum())))
                        else:
                            log('[LSPR] 检出结构 %d 像素，可用样本不足，未改动' % int(_M.sum()))
            except Exception as _e_lsp:
                log('[LSPR] 后处理失败（%s: %s），维持原结果' % (type(_e_lsp).__name__, _e_lsp))
        rej_rate = float(r_d['rej_rate'])
        wmax = [float(r_d['wmax'])]
        del r_d
        log(f'[drizzle] 分带 {band} 行 × {int(np.ceil(h / band))} 带  '
            f'排异剔除率 {rej_rate:.3%}')
    else:
        ini, ini_args = fs.reader
        nw = 0 if (not fs.parallelable or _no_parallel()) \
            else plan_workers(512 * 2 ** 20, '叠加', cap=CAP_WORKERS)
        strip = plan_strip(n, w, fs.resident_bytes, workers=max(1, nw), logf=log)
        Stk = np.empty((h, w), np.float32)
        neff = np.empty((h, w), np.float32)
        _lsp = bool(int(os.environ.get('DWT_LSP', '0')))
        _lsp_acc = None
        if _lsp:
            _lsp_acc = {'we': np.zeros((h, w), np.float32),
                        'wde': np.zeros((h, w), np.float32),
                        'cnte': np.zeros((h, w), np.float32),
                        'sw': np.zeros((h, w), np.float32),
                        'swd': np.zeros((h, w), np.float32)}
        t2 = time.perf_counter()
        rej_num = rej_den = 0.0
        wmax = []
        n_strip = int(np.ceil(h / strip))
        _emit('叠加', 0.0, f'0/{n_strip} 条带')
        try:
            if nw > 1:
                # 每条带只看自己那几行、写回互不重叠的行段 → 可分给多进程；
                #   map 保序 → 行段落位与两个浮点累加都与串行同序
                ex = ProcessPoolExecutor(max_workers=nw, initializer=ini, initargs=ini_args)
                try:
                    _beta_st = float(_detail_beta(p))
                    jobs = ((y0, min(h, y0 + strip), R[y0:y0 + strip],
                             band_cfield(C, C_tiles, y0, min(h, y0 + strip), h, w), S, A, B, rej_k,
                             fs.sky0, rej_gate, T, fp, _beta_st,
                             (fs.sigma if bool(getattr(p, 'rej_z', False)) else None),
                             float(getattr(p, 'max_share', 0.0) or 0.0))
                            for y0 in range(0, h, strip))
                    for k, (s, ne, rr, wm) in enumerate(ex.map(_task_strip, jobs, chunksize=1)):
                        _check_cancel()
                        y0 = k * strip
                        y1 = min(h, y0 + strip)
                        Stk[y0:y1] = s
                        neff[y0:y1] = ne
                        cnt = float(s.size)
                        rej_num += float(rr) * cnt
                        rej_den += cnt
                        wmax.append(wm)
                        del s, ne
                        _emit('叠加', (k + 1) / n_strip, f'{k + 1}/{n_strip} 条带')
                finally:
                    ex.shutdown(wait=True, cancel_futures=True)
            else:
                _sf_mod = None
                if _S_field is not None:
                    try:
                        import sigma_field as _sf_mod
                    except Exception:
                        _sf_mod = None
                for k, y0 in enumerate(range(0, h, strip)):
                    _check_cancel()
                    y1 = min(h, y0 + strip)
                    _beta = _detail_beta(p)
                    _rej_z = bool(getattr(p, 'rej_z', False))
                    _wcap = float(getattr(p, 'max_share', 0.0) or 0.0)
                    if _S_field is None or _sf_mod is None:
                        W = strip_weights(band_cfield(C, C_tiles, y0, y1, h, w),
                                          S, R[y0:y1], A, B, T=T, fp=fp, beta=_beta, cap=_wcap)
                        D = fs.stack_strips(y0, y1)
                        s, ne, rr = stack_strip(D, W, R[y0:y1], fs.sky0, rej_k=rej_k,
                                                rej_max_frac=rej_gate,
                                                frame_sig=(fs.sigma if _rej_z else None),
                                                lsp_acc=({'we': _lsp_acc['we'][y0:y1],
                                                          'wde': _lsp_acc['wde'][y0:y1],
                                                          'cnte': _lsp_acc['cnte'][y0:y1]} if _lsp_acc else None))
                        if _lsp_acc is not None:
                            _lsp_acc['sw'][y0:y1] = W.sum(axis=0, dtype=np.float32)
                            _lsp_acc['swd'][y0:y1] = (W * D).sum(axis=0, dtype=np.float32)
                        Stk[y0:y1] = s
                        neff[y0:y1] = ne
                        cnt = float(s.size)
                        rej_num += float(rr) * cnt
                        rej_den += cnt
                        wmax.append(float(W.max(axis=0).mean()))
                        del W, D, s, ne, rr
                    else:
                        # 局部权重：按列块切，每块用该块的 S_i(cell) 标量
                        rows = _sf_mod.cell_for_band(_S_field, y0, y1, h, _S_field.shape[1])
                        nb = int(_S_field.shape[2])
                        D_full = None
                        for cx in range(nb):
                            x0 = cx * w // nb
                            x1 = (cx + 1) * w // nb
                            if x1 <= x0:
                                continue
                            S_c = rows.mean(axis=1)[:, cx]
                            W = strip_weights(
                                band_cfield(C, C_tiles, y0, y1, h, w)[..., x0:x1],
                                S_c, R[y0:y1, x0:x1], A, B, T=T, fp=fp, beta=_beta, cap=_wcap)
                            if D_full is None:
                                D_full = fs.stack_strips(y0, y1)
                            s, ne, rr = stack_strip(D_full[:, :, x0:x1], W,
                                                    R[y0:y1, x0:x1], fs.sky0,
                                                    rej_k=rej_k, rej_max_frac=rej_gate,
                                                    frame_sig=(fs.sigma if _rej_z else None))
                            Stk[y0:y1, x0:x1] = s
                            neff[y0:y1, x0:x1] = ne
                            cnt = float(s.size)
                            rej_num += float(rr) * cnt
                            rej_den += cnt
                            wmax.append(float(W.max(axis=0).mean()))
                            del W, s, ne, rr
                        del D_full
                    _emit('叠加', (k + 1) / n_strip, f'{k + 1}/{n_strip} 条带')
        finally:
            # 正常跑完与中途终止都要清理临时帧文件（三通道共享时由 _run 收尾统一关）
            if not getattr(fs, '_shared', False):
                fs.close(keep=keep_frames)
    if _lsp_acc is not None:
        try:
            import lsp as _lsp_mod
            _M = _lsp_mod.detect(_lsp_acc['cnte'], n, R=R, logf=log)
            if _M is not None and bool(_M.any()):
                _sw = _lsp_acc['sw']; _swd = _lsp_acc['swd']
                _we = _lsp_acc['we']; _wde = _lsp_acc['wde']
                _den = _sw - _we
                _ok = _M & (_den > 0.25 * np.maximum(_sw, 1e-12))
                if bool(_ok.any()):
                    _corr = (_swd - _wde) / np.maximum(_den, 1e-12)
                    Stk[_ok] = _corr[_ok].astype(np.float32)
                    log('[LSPR] 大尺度排异：结构覆盖 %d 像素（%.3f%%），可修正 %d，已扣除可疑样本'
                        % (int(_M.sum()), 100.0 * float(_M.mean()), int(_ok.sum())))
                else:
                    log('[LSPR] 检出结构 %d 像素，但可修正样本不足，未改动' % int(_M.sum()))
        except Exception as _e_lsp:
            log('[LSPR] 后处理失败（%s: %s），维持原结果' % (type(_e_lsp).__name__, _e_lsp))
    # 条带全部落位后一次回乘（串行/并行在此汇合，只乘一次；元素级乘法与切分无关）
    Stk *= np.float32(scale_out)
    log(f'[量纲] 成品回乘 ×{scale_out:g}（曝光中位 {t_med:g}s / 归一基准 {EXPTIME_REF:g}s），'
        f'量纲与素材一致')
    if drz_on:
        log(f'[W] A={A:g} B={B:g}（B=2 为逆方差最优）  drizzle {drz_s:g}× '
            f'pixfrac {drz_p:g}  分母 ΣW·c（式 7 的归一已由它吸收）')
    else:
        rej_rate = rej_num / max(rej_den, 1.0)
        log(f'[W] A={A:g} B={B:g}（B=2 为逆方差最优）  条带 {strip} 行 × '
            f'{int(np.ceil(h / strip))} 条  归一后 mean_i(W)=1')
    if globals().get('_GPU_WMAX', False):
        # GPU 路径返回的 'wmax' 是覆盖和（cov.max()），不是逐帧最大权重
        log(f'[W] 单帧最大权重：GPU 路径不统计该量（这里的 {np.mean(wmax):.1f} 是覆盖和）')
    else:
        log(f'[W] 单帧最大权重（条带均值）{np.mean(wmax):.3f}  '
            f'等权份额 = {1.0:.3f} → 最大独占约 {np.mean(wmax):.1f}× 份额')
    _wc_used = float(getattr(p, 'max_share', 0.0) or 0.0)
    if 0.0 < _wc_used < 1.0 and n > 1:
        _L = 0.5 * float(np.log(max((n - 1) * _wc_used / (1.0 - _wc_used), 1e-9)))
        log(f'[W] 份额上限 {_wc_used:.0%}：对数权重 clamp 到 ±{_L:.3f}'
            f'（最重帧 ≤ 其余帧 {np.exp(2 * _L):.1f} 倍 → 任一帧份额 ≤ {_wc_used:.0%}）')
    log(f'[叠加] 完成 {time.perf_counter() - t2:.1f}s  排异剔除率 {rej_rate:.3%}')
    log(f'[叠加] N_eff 中位 {np.median(neff):.2f}/{n}  '
        f'P10 {np.percentile(neff, 10):.2f}  最小 {neff.min():.2f}')
    # 分区分读：细节区 vs 朦胧区的 N_eff 与噪声
    det = R > 0.5
    amb = R < 0.1
    sky_m = Stk < np.percentile(Stk, 50)
    sky_sd = None
    for name, msk in (('细节区 R>0.5', det), ('朦胧区 R<0.1', amb), ('天空(下半亮度)', sky_m)):
        if msk.sum() == 0:
            continue
        sd = 1.4826 * float(np.median(np.abs(Stk[msk] - np.median(Stk[msk]))))
        if name.startswith('天空'):
            sky_sd = float(sd)
        log(f'[读数] {name:<18} 占 {msk.mean():6.2%}  N_eff 中位 '
            f'{np.median(neff[msk]):5.2f}  σ {sd:.4e}')

    try:
        _stars_o0 = _seed_map(drz_plan, drz_s) if drz_on else stars
        _snr = _snr_report(_snr_seed_img, stars, Stk, neff, scale=float(drz_s if drz_on else 1.0))
        if _snr:
            log('[SNR] 单帧天空噪声 %.4g → 成品 %.4g  噪声降低 %.1f×（= 等效 SNR 提升）'
                % (_snr['n1'], _snr['n2'], _snr['gain']))
            if _snr['gain'] < 1.0:
                log('[SNR] 注意：成品噪声反而比单帧高 —— 多半是帧数太少/覆盖不足/帧间电平没对齐，'
                    '这次的噪声比不代表最终质量（帧数上来后会回到 1 以上）')
            log('[SNR] 等效帧数中位 -> √N_eff = %.1f×（实测 %.1f×，为理论的 %.0f%%；'
                '差额来自 drizzle 重采样把噪声抹成空间相关）'
                % (_snr['sqrt_neff'], _snr['gain'],
                   100.0 * _snr['gain'] / max(_snr['sqrt_neff'], 1e-9)))
    except Exception as _e:
        log('[SNR] 统计失败（不影响成品）：%s' % _e)

    # 成品自身的星点 FWHM（同一星表、同一开源口径）→ 与单帧/参照栈直接可比
    _emit('输出', 0.3, '量成品星点')
    if drz_on:
        # 星表与量星窗口都要换算到输出网格；读数再折回本机口径才与 fmed 同尺度
        stars_o = _seed_map(drz_plan, drz_s)
        # 输出像素/原生像素 的真实比例：带 2× 参考网格时是 2，而不是 drz_s（1）
        _pxr = float(_px_scale if np.isfinite(_px_scale) and _px_scale > 0 else drz_s)
        hw = max(16, int(round(16 * _pxr)))
        fsz = max(29, int(round(29 * _pxr)) | 1)
        f_out_px = measure_fwhm(Stk, stars_o, fwhm_guess=6.0 * _pxr,
                                half=hw, fit_shape=fsz)
        f_out = f_out_px / _pxr
        if np.isfinite(f_out) and np.isfinite(fmed) and fmed > 0:
            log(f'[输出] 成品星点 FWHM {f_out:.2f}px（本机口径；图上 {f_out_px:.2f}px'
                f' @×{_pxr:g}）  单帧中位 {fmed:.2f}px  '
                f'改善 {100 * (1 - f_out / fmed):.1f}%')
        else:
            log('[输出] 成品星点 FWHM 量不出来（该通道样本太少或太暗，'
                '高斯拟合不收敛）；图本身正常，以目视/外部工具为准')
        log(f'[输出] 采样口径：σ_psf {sigma_psf:.2f}→{sigma_psf * _pxr:.2f}px'
            f'（1.5~2 为目标区间）')
        if _CFA.get('pattern') and abs(_pxr - 1.0) < 1e-6 and _CFA.get('ch') == 'R':
            log('[CFA] 提示：1× CFA drizzle 保持输入量纲（与 PI 的 Bayer drizzle 一致），'
                '而 WBPP 成品通常比输入暗约 30%（其 master 天空 ≈0.70×输入）。'
                '在同一拉伸下直接对比会显得我们的星点更早饱和、肩部更陡。'
                '要与 WBPP 对齐量级请用 --match-scale（GUI「按参考图标定亮度」）指向其 master。')
        if f_out > 1.15 * fmed:
            log('[输出] 注意：这个通道量出来比单帧还糊，多半是通道太暗、高斯拟合在'
                '暗星上发散（彩机的蓝通道常见）。图本身没问题，以目视/外部工具为准。')
    else:
        f_out = measure_fwhm(Stk, stars)
        if np.isfinite(f_out) and np.isfinite(fmed) and fmed > 0:
            log(f'[输出] 成品星点 FWHM {f_out:.2f}px（单帧中位 {fmed:.2f}px，'
                f'改善 {100 * (1 - f_out / fmed):.1f}%）')
        else:
            # 帧数很少（如 --limit 2）或通道太暗时高斯拟合会不收敛，
            # 以前这里直接格式化 nan，日志里出现 "nanpx"；改成明确说明。
            log('[输出] 成品星点 FWHM 量不出来（帧数太少或该通道太暗，高斯拟合不收敛）；'
                '图本身正常，以目视/外部工具为准')

    res = {
        'n': n, 'shape': Stk.shape, 'fwhms': fwhms, 'C': C, 'S': S, 'sigma': sig_out,
        'fwhm_med': fmed, 'fwhm_out': f_out, 'sigma_psf': sigma_psf,
        'R_info': rinfo, 'neff': neff, 'rej_rate': float(rej_rate),
        'A': A, 'B': B, 'rej_k': rej_k, 'stars': stars,
        'metas': metas, 'skies': skies, 'scale_out': float(scale_out),
        'drizzle': (None if not drz_on else {
            'scale': drz_s, 'pixfrac': drz_p, 'dx': dx, 'dy': dy,
            'theta': np.asarray(sh.get('theta', np.zeros_like(dx))),
            'plan': drz_plan, 'scatter': sh['scatter']}),
    }

    if save_xisf:
        p_out = out_dir / f'{stem}.xisf'
        _emit('输出', 0.7, '导出成品')
        # 未覆盖像素写 NaN（不是 0）：写 0 会让 PI/EZ Soft Stretch 的直方图被
        # 一大块 0 压成一根针，自动拉伸失效。NaN 在 PI 里显示为纯黑，但直方图
        # 与统计会跳过它。
        if cov_map is not None:
            _cm = np.asarray(cov_map)
            if _cm.shape == Stk.shape:
                _bad = ~(_cm > 0)
                _nan_n = int(np.count_nonzero(_bad))
                if _nan_n:
                    Stk = np.where(_bad, np.nan, Stk).astype(Stk.dtype, copy=False)
                    log(f'[输出] 未覆盖像素置 NaN：{_nan_n} 个（{_nan_n / Stk.size:.3%}）')
        # 两类"空"必须分开处理：
        #   ① 画布上没有任何帧覆盖的像素（cov=0）——这是几何决定的，位于抖动并集之外，
        #      多在画布边缘；写 NaN 表示"无数据"，交给自动裁边，绝不伪造数值。
        #   ② 有样本但被排异剔空的像素（cov>0、分子=0）——已在排异处保证"至少留一个
        #      真实样本"，正常不会再出现；若出现说明逻辑回退，补上并报警。
        _z = np.isfinite(Stk) & (Stk <= 0.0)
        if bool(_z.any()):
            _ct = np.asarray(cov_map) if cov_map is not None else None
            # cov 是「落点面积之和」。画布边缘只有部分液滴压到，cov 很小但 >0；
            # 这些像素属于「没被完整覆盖」，不是「被排异剔空」——用 cov<0.5 区分，
            # 否则每次都会误报几万个「排异剔空」。
            _unc = (_z & (_ct < 0.5)) if _ct is not None and _ct.shape == Stk.shape else _z.copy()
            _rej = _z & ~_unc
            if bool(_unc.any()):
                Stk[_unc] = np.nan
                log('[输出] 无样本像素：%d 个（%.3f%%，画布上没有任何帧覆盖，多为抖动并集之外）'
                    '——写 NaN，不伪造数值；可用「自动裁边」去掉'
                    % (int(_unc.sum()), 100.0 * float(_unc.sum()) / Stk.size))
            if bool(_rej.any()):
                try:
                    from scipy.ndimage import distance_transform_edt as _edt2
                    _ok = ~_z
                    if _ok.any():
                        _idx = _edt2(_z, return_distances=False, return_indices=True)
                        Stk = Stk[tuple(_idx)].astype(Stk.dtype, copy=False)
                        log('[输出] 警告：%d 个像素有样本却被排异剔空（%.3f%%），已用最近邻临时填充；'
                            '请把日志发回（排异逻辑应保证至少保留一个样本）'
                            % (int(_rej.sum()), 100.0 * float(_rej.sum()) / Stk.size))
                except Exception as _e:
                    log('[输出] 排异空像素填充失败（%s），保留原值' % _e)
        # 2×2 去格点（CFA 专用）：即使液滴按同色格点铺满，输出像素的 4 个相位
        # 仍会拿到不同数目的样本（液滴边长 2.3 输出像素 vs 输出格点 1 像素），
        # 表现为每隔一个像素的系统性亮暗差——强拉伸下就是"细密网格"，
        # 星点边缘对比度最高，看起来像"黑边"。实测用户 78 帧 B：相位差 0.9%~3.3%。
        # 这里用天空为主体的 4 个相位中位数估计增益（星点占比 <2%，中位不受影响），
        # 归一到均值后除掉。只改常数增益，不动任何结构，也不引入噪声。
        if _CFA.get('pattern') and os.environ.get('DWT_CFA_DELATTICE', '1') != '0':
            try:
                _dl = np.isfinite(Stk)
                _g = np.ones((2, 2), np.float64)
                for _i in (0, 1):
                    for _j in (0, 1):
                        _sub = Stk[_i::2, _j::2]
                        _sub = _sub[np.isfinite(_sub)]
                        if _sub.size > 1000:
                            _g[_i, _j] = float(np.median(_sub))
                        else:
                            _g[_i, _j] = float('nan')
                if np.isfinite(_g).all() and _g.min() > 0:
                    _gm = _g.mean()
                    _sp = (_g.max() - _g.min()) / _gm * 100.0
                    if _sp > 0.05:
                        for _i in (0, 1):
                            for _j in (0, 1):
                                Stk[_i::2, _j::2] *= np.float32(_gm / _g[_i, _j])
                        log('[CFA] 2×2 去格点：四相位增益 %s（相差 %.2f%%）→ 已归一'
                            % (np.array2string(np.round(_g / _gm, 4), separator=','), _sp))
            except Exception as _e:
                log('[CFA] 去格点失败（%s），跳过' % _e)
        # ── 大尺度结构排异（LSPR 等价）：在叠加域检出细长直线结构并扣除 ──
        # 判据不是“像素离群”而是“该像素属于一个跳很多像素的细长直线”；
        # 只在 |d-中心| ≤ 4w 的带内做加性扣除，带外像素逐位不变。DWT_LSPR=0 关闭。
        _lspn = 0
        if str(os.environ.get('DWT_LSPR', '1')).strip() != '0':
            try:
                import lsp as _lsp_new
                _t_lsp = time.time()
                _lsp_smp = None
                _lsp_nf = 0
                _lsp_sf = None
                try:
                    if _lsp_new._LSP_FRAMES and int(n) > 1 and _lsp_pmap:
                        _pm_l = dict(_lsp_pmap)
                        _DBG_L = str(os.environ.get('DWT_LSPR_DEBUG','0')) in ('1','true','True')
                        _cache_l = {}
                        _pmap = {}
                        try:
                            for _r0, _d0, _fl0 in os.walk(str(getattr(p, 'photos', '') or '')):
                                for _f0 in _fl0:
                                    if _f0.lower().endswith('.xisf'):
                                        _pmap[_f0] = os.path.join(_r0, _f0)
                        except Exception:
                            _pmap = {}

                        def _lsp_smp(j, _fs=fs, _pm=_pm_l, _cc=_cache_l, _pmap=_pmap):
                            j = int(j)
                            if j in _cc:
                                return _cc[j]
                            _p6 = _pm.get(j)
                            if _p6 is None:
                                if _DBG_L:
                                    try: log('[LSPR][debug] 无 p 表项 j=%d（键 %s）' % (j, sorted(_pm.keys())[:5]))
                                    except Exception: pass
                                return None
                            # 父进程里 fs.frame() 在多进程后端下不可用（frame 数据在子进程/临时 memmap），
                            # 直接读 memmap 视图即可：帧域验证只要「带内 vs 带外中位差」的相对量，
                            # 曝光归一/天空常数平移都不影响该差值。
                            _im = None
                            try:
                                _mmv = getattr(_fs, 'mm', None)
                                if _mmv is not None:
                                    _im = _mmv[j]
                                else:
                                    _arr = getattr(_fs, 'arrays', None)
                                    if _arr is not None:
                                        _im = _arr[j]
                                    else:
                                        # 叠加结束时 memmap 已删除、父进程不留帧：
                                        # 帧域验证只读被测的那几帧（≤10），从原始 XISF 取。
                                        _fnm = (list(getattr(_fs, 'names', None) or []))
                                        _pth = _pmap.get(_fnm[j]) if j < len(_fnm) else None
                                        if _pth:
                                            from xisf_io import read_xisf as _rxf
                                            _lst = _rxf(_pth)
                                            _im = _lst[0].data if _lst else None
                            except Exception as _e_fr:
                                if _DBG_L:
                                    try: log('[LSPR][debug] 读帧失败 j=%d %s: %s' % (j, type(_e_fr).__name__, _e_fr))
                                    except Exception: pass
                                _im = None
                            if _im is None:
                                return None
                            # p6 = (p0,p1,p2,p3,p4,p5)，语义见 gpu_wgpu_cfa_fatom：
                            #   X = p0*(gx+0.5) + p1*(gy+0.5) + p2
                            _cc[j] = {'img': _im, 'p': _p6, 'k': 1.0}
                            if _DBG_L and len(_cc) <= 3:
                                try: log('[LSPR][debug] sampler j=%d img %s p=(%.4f,%.4f,%.2f,%.4f,%.4f,%.2f)' % (j, getattr(_im,'shape',None), _p6[0], _p6[1], _p6[2], _p6[3], _p6[4], _p6[5]))
                                except Exception: pass
                            return _cc[j]
                        _lsp_nf = int(len(_pm_l))
                        _lsp_sf = float(np.median(fs.sigma)) if fs.sigma is not None else None
                        log('[LSPR] 帧域验证：p 表 %d 帧（着色器同源）' % _lsp_nf)
                        if _DBG_L:
                            try:
                                log('[LSPR][debug] fs: shape=%s n=%d mm=%s arrays=%s path=%s exists=%s'
                                    % (fs.shape, fs.n, 'None' if getattr(fs,'mm',None) is None else 'ok',
                                       'None' if getattr(fs,'arrays',None) is None else 'ok',
                                       getattr(fs,'path',None),
                                       getattr(getattr(fs,'path',None),'exists',lambda: 'n/a')()))
                            except Exception as _e_d:
                                log('[LSPR][debug] fs dump 失败 %s' % _e_d)
                except Exception as _e_lsp3:
                    log('[LSPR] 帧域验证不可用（%s: %s），退回叠加域判据'
                        % (type(_e_lsp3).__name__, _e_lsp3))
                    _lsp_smp = None
                _lspn, _lspx = _lsp_new.process_auto(Stk, logf=log, sampler=_lsp_smp,
                                                     nframes=_lsp_nf, sig_frame=_lsp_sf)
                log('[LSPR] 结构排异：检出并扣除 %d 条细长结构，改动 %d 像素'
                    '（%.3f%%），用时 %.2fs'
                    % (_lspn, _lspx, 100.0 * _lspx / float(max(1, Stk.size)), time.time() - _t_lsp))
            except Exception as _e_lsp2:
                log('[LSPR] 结构排异跳过（%s: %s）'
                    % (type(_e_lsp2).__name__, _e_lsp2))
        # 成品有效性门禁：几乎全空的图不该写盘（否则用户拿到一张全黑图还收到"完成"）
        _fin = np.isfinite(Stk)
        _vfrac = float(np.count_nonzero(_fin)) / float(max(1, Stk.size))
        if _vfrac < 0.05:
            _nmed = float(np.median(neff)) if neff is not None and len(neff) else float('nan')
            log('[输出] 成品只有 %.2f%% 像素有数据（N_eff 中位 %.2f），判定叠加失败、不写成品。'
                % (_vfrac * 100.0, _nmed))
            log('[输出] 常见原因：参考网格旋转/尺度解错、或所有帧都被配准门禁排除；'
                '请把本次日志发回，可用 --frame 去掉参考网格再试。')
            raise SystemExit(3)
        if _vfrac < 0.5:
            log('[输出] 注意：只有 %.1f%% 像素有数据，边缘大片为空（前面可加「自动裁边」）'
                % (_vfrac * 100.0))
        # 可选：把成品天空标定到「输出网格」参考图（WBPP master）的同一量级，
        # 便于和 PI 的成品直接对比亮度。默认关，保持与输入帧同一量纲。
        _ms = bool(getattr(p, 'match_scale', False))
        _rm = float(_MISC.get('ref_median', 0.0) or 0.0)
        if _ms and _rm <= 0:
            # --aligned 模式不走参考网格匹配，_MISC 里没有 ref_median，
            # 直接读一次 --frame 指定的参考图拿天空中位。
            _fr = str(getattr(p, 'frame', '') or '')
            if _fr:
                try:
                    _rv = np.asarray(
                        __import__('xisf_io').read_xisf(_fr)[0].data, np.float64)
                    _rm = float(np.nanmedian(_rv))
                except Exception:
                    _rm = 0.0
        if _ms and _rm > 0:
            try:
                _om = float(np.nanmedian(Stk[np.isfinite(Stk)]))
                if _om > 0:
                    _k = _rm / _om
                    Stk *= np.float32(_k)
                    scale_out *= _k
                    log('[量纲] 按参考图标定：×%.4f（天空 %.6g → %.6g）'
                        % (_k, _om, _om * _k))
            except Exception as _e:
                log('[量纲] 参考图标定跳过（%s）' % _e)
        try:
            _sat = int(np.count_nonzero(np.isfinite(Stk) & (Stk >= 0.99)))
            if _sat:
                log('[输出] 提示：成品有 %d 个像素 ≥0.99（%.4f%%），这些星点核心在输入帧里'
                    '已经饱和，叠加无法恢复；可缩短单张曝光或用 --match-scale 与参考图对齐量级'
                    % (_sat, 100.0 * _sat / float(Stk.size)))
        except Exception:
            pass
        # ---- --frame + --crop：画布就是参考栅格，裁剪 = 写盘前对数组切片。
        #      裁剪框沿用非 frame 路径的口径（以 center 为中心，缺省画面中心），
        #      只是坐标换到画布上；越界或画布不是参考栅格就直接报错，不静默出图。
        if _frame_ref and int(crop or 0) > 0:
            _cn = int(crop)
            _chh, _cww = int(Stk.shape[0]), int(Stk.shape[1])
            if _ref_wh is None or (_cww, _chh) != (int(_ref_wh[0]), int(_ref_wh[1])):
                log(f'[裁剪] 画布 {_cww}×{_chh} 不是参考栅格 '
                    f'{_ref_wh[0] if _ref_wh else "?"}×{_ref_wh[1] if _ref_wh else "?"}'
                    f'（--frame 配准失败或未启用 --drizzle），拒绝按 --crop 切片出图')
                raise SystemExit(2)
            if _cn > min(_chh, _cww):
                log(f'[裁剪] --crop {_cn} 超出参考画布 {_cww}×{_chh}，拒绝出图')
                raise SystemExit(2)
            if center is None:
                _ccy, _ccx = _chh // 2, _cww // 2
            else:
                _ccy, _ccx = int(center[0]), int(center[1])
            _chalf = _cn // 2
            _cy0 = max(0, min(_ccy - _chalf, _chh - _cn))
            _cx0 = max(0, min(_ccx - _chalf, _cww - _cn))
            Stk = np.ascontiguousarray(Stk[_cy0:_cy0 + _cn, _cx0:_cx0 + _cn])
            neff = np.ascontiguousarray(neff[_cy0:_cy0 + _cn, _cx0:_cx0 + _cn])
            if cov_map is not None:
                cov_map = np.ascontiguousarray(cov_map[_cy0:_cy0 + _cn, _cx0:_cx0 + _cn])
            R = np.ascontiguousarray(R[_cy0:_cy0 + _cn, _cx0:_cx0 + _cn])
            log(f'[裁剪] --frame 模式：画布 {_cww}×{_chh} → 切片 '
                f'y {_cy0}..{_cy0 + _cn}  x {_cx0}..{_cx0 + _cn}'
                f'（{_cn}×{_cn}，中心 {_ccx},{_ccy}）')
        save_linear_xisf(p_out, Stk, metas,
                         f'PWS A={A:g} B={B:g} rej_k={rej_k:g}',
                         diag={'A': A, 'B': B, 'RejK': rej_k, 'RejGate': rej_gate,
                               'ScaleOut': float(scale_out), 'RejRate': float(rej_rate),
                               'Drizzle': (f'{drz_s:g}x p{drz_p:g}' if drz_on else 'off'),
                               'StarCount': len(stars), 'FwhmFrameMedian': float(fmed),
                               'FwhmOutput': float(f_out),
                               'NeffMedian': float(np.median(neff)),
                               'SigmaFrameMedian': float(np.median(sig_out)),
                               'LsprStruct': int(_lspn),
                               'SigmaSky': sky_sd})
        # 辅助数组（R 场 / N_eff / 覆盖）整幅各约 417 MB，三通道就要 1.25 GB。
        # 权重场页与评价页默认读 1/4 分辨率版本（每个约 26 MB，块平均，够看结构）；
        # 需要逐像素原尺寸时设 DWT_SAVE_AUX=1（额外写整幅，每通道约 0.6 s）。
        _save_aux = str(os.environ.get('DWT_SAVE_AUX', '0')).strip() == '1'

        def _q4(a):
            a = np.asarray(a, np.float32)
            h, w = a.shape[-2], a.shape[-1]
            h2, w2 = h - h % 4, w - w % 4
            b = a[..., :h2, :w2]
            b = b.reshape(b.shape[:-2] + (h2 // 4, 4, w2 // 4, 4))
            return b.mean(axis=(-3, -1))

        try:
            np.save(out_dir / f'{stem}_R_q4.npy', _q4(R))
            np.save(out_dir / f'{stem}_neff_q4.npy', _q4(neff))
            if cov_map is not None:
                np.save(out_dir / f'{stem}_ncov_q4.npy', _q4(cov_map))
        except Exception as _e:
            log(f'[输出] 1/4 分辨率辅助图写入失败：{_e}')

        if _save_aux:
            np.save(out_dir / f'{stem}_neff.npy', neff)
            if cov_map is not None:
                np.save(out_dir / f'{stem}_ncov.npy', np.asarray(cov_map, np.float32))
            np.save(out_dir / f'{stem}_R.npy', R)
        _wt = np.arange(n)
        if _tdrop.any():
            _wt = _wt[~_tdrop]
            log(f'[输出] 权重表不含卫星拖尾帧（{len(_wt)}/{n} 行）')
        np.savetxt(out_dir / f'{stem}_weights.txt',
                   np.column_stack([_wt, fwhms[_wt], C[_wt], sig_out[_wt], S[_wt]]),
                   header='idx fwhm_px C_sharp sigma S_snr', comments='# ', fmt='%.6g')
        log(f'[输出] 逐帧权重表 {stem}_weights.txt / R 场与 N_eff 的 1/4 分辨率版 '
            f'{stem}_R_q4.npy、{stem}_neff_q4.npy'
            + (f' / 原尺寸 {stem}_R.npy、{stem}_neff.npy' if _save_aux else
               '（原尺寸需设 DWT_SAVE_AUX=1）'))

        # ---- 原生尺寸副本：按 N×N 块平均缩回（不额外叠一遍）----
        try:
            _want1x = int(getattr(p, 'native_1x', 0) or 0) > 0
            # 按**实际画布倍率**缩，不能按用户勾的 drz_s：勾了输出网格时画布是
            # 参考图定的（可能是 1×），此时再按 2 去块平均会把图缩成一半。
            _s_eff = float(drz_plan['s']) if drz_plan else float(drz_s)
            _fx = int(round(_s_eff)) if drz_on else 1
            if _want1x and _fx > 1:
                _h1 = (Stk.shape[0] // _fx) * _fx
                _w1 = (Stk.shape[1] // _fx) * _fx

                def _blk(im):
                    return im[:_h1, :_w1].reshape(_h1 // _fx, _fx,
                                                  _w1 // _fx, _fx).mean(axis=(1, 3))

                _s1 = _blk(np.asarray(Stk, np.float32))
                _p1x = out_dir / f'{stem}_1x.xisf'
                save_linear_xisf(_p1x, np.ascontiguousarray(_s1), metas,
                                 f'PWS A={A:g} B={B:g} 1x',
                                 diag={'A': A, 'B': B, 'ScaleOut': 1.0,
                                       'Drizzle': f'{drz_s:g}x -> binned 1x'})
                np.save(out_dir / f'{stem}_1x_neff.npy',
                        np.ascontiguousarray(_blk(np.asarray(neff, np.float32))))
                np.save(out_dir / f'{stem}_1x_R.npy',
                        np.ascontiguousarray(_blk(np.asarray(R, np.float32))))
                log(f'[输出] 原生尺寸副本 {_p1x.name}  {_s1.shape[1]}×{_s1.shape[0]}'
                    f'（由 {drz_s:g}× 块平均 {_fx}×{_fx}）')
            elif _want1x:
                log('[输出] 另存原尺寸：当前就是 1×，无需再缩')
        except Exception as _exc:
            log(f'[输出] 原生尺寸副本失败：{_exc}')

        # ---- 按覆盖率自动裁边：四角空白 + 边缘薄覆盖带子 ----
        cc_frac = float(getattr(p, 'crop_cover', 0.0) or 0.0)
        if cc_frac > 0 and cov_map is not None:
            try:
                box = cropcover.crop_box(cov_map, cc_frac, min_frames=3.0)
            except Exception as exc:
                box = None
                log(f'[裁切] 自动裁边失败（{exc}），保留原画幅')
            if box is None:
                log('[裁切] 自动裁边：找不到可信的覆盖区，保留原画幅')
            else:
                _y0, _y1, _x0, _x1 = box
                _h2, _w2 = _y1 - _y0, _x1 - _x0
                if _h2 < 32 or _w2 < 32:
                    log(f'[裁切] 自动裁边：只剩 {_w2}×{_h2}，太小，保留原画幅')
                else:
                    _keep = 100.0 * _h2 * _w2 / float(Stk.size)
                    _p2 = out_dir / f'{stem}_autocrop.xisf'
                    save_linear_xisf(_p2, np.ascontiguousarray(Stk[_y0:_y1, _x0:_x1]),
                                     metas,
                                     f'PWS A={A:g} B={B:g} crop={cc_frac:g}',
                                     diag={'A': A, 'B': B, 'CropCover': cc_frac,
                                           'CropBox': f'{_y0},{_y1},{_x0},{_x1}',
                                           'FwhmOutput': float(f_out)})
                    np.save(out_dir / f'{stem}_autocrop_neff.npy',
                            np.ascontiguousarray(neff[_y0:_y1, _x0:_x1]))
                    np.save(out_dir / f'{stem}_autocrop_R.npy',
                            np.ascontiguousarray(R[_y0:_y1, _x0:_x1]))
                    log(f'[裁切] 自动裁边（覆盖 ≥ {cc_frac:.0%}）：'
                        f'y {_y0}..{_y1}  x {_x0}..{_x1}  → {_w2}×{_h2}'
                        f'（原画幅的 {_keep:.1f}%）  {_p2.name}')

        ref_path = str(getattr(p, 'match', '') or '').strip()
        if ref_path:
            _emit('输出', 0.9, '对齐裁切到参考')
            if matchref is None:
                log('[裁切] 跳过：matchref 模块不可用（缺依赖）')
            else:
                try:
                    _cc = float(getattr(p, 'crop_cover', 0.0) or 0.0)
                    _mask = None
                    if _cc > 0 and cov_map is not None:
                        _mx = float(np.max(cov_map))
                        _mask = cov_map >= max(3.0, _cc * _mx)
                    matchref.apply(Stk, ref_path,
                                   str(out_dir / f'{stem}_match.xisf'), log=log,
                                   trim=(_cc > 0), src_valid=_mask)
                except Exception as exc:      # 裁切失败不该毁掉已叠好的成品
                    log(f'[裁切] 失败：{type(exc).__name__}: {exc}')

    log(f'[时] 总计 {time.perf_counter() - t_start:.1f}s')
    _emit('输出', 1.0, '完成')
    res['stack'] = Stk
    return res


def build_parser() -> argparse.ArgumentParser:
    """命令行参数表由 pws_params.PARAMS 生成：与界面同源，加参数只改一处"""
    ap = argparse.ArgumentParser(
        prog='DWT',
        description='DWT - Partition-Weighted Stacking（PWS 分区加权叠加）')
    base = PwsParams()
    for prm in PARAMS:
        flag = cli_flag(prm.key)
        # argparse 会对 help 做一次 % 插值，说明里的"2~4%"必须写成"2~4%%"
        text = f'{prm.desc}。{prm.usage}'.replace('%', '%%')
        default = getattr(base, prm.key)
        if prm.kind == 'check':
            ap.add_argument(flag, action='store_true', default=default, help=text)
            if default is True:
                # 默认开启的开关要给一个否定形式，否则命令行无法关掉
                ap.add_argument('--no-' + flag.lstrip('-'), dest=prm.key,
                                action='store_false', default=True,
                                help='关闭「%s」' % prm.label)
        elif prm.kind == 'float':
            ap.add_argument(flag, type=float, default=default, help=text)
        elif prm.kind == 'int':
            ap.add_argument(flag, type=int, default=default, help=text)
        elif prm.kind == 'choice':
            ap.add_argument(flag, default=default,
                            choices=[v for _, v in prm.choices], help=text)
        else:
            ap.add_argument(flag, default=default, help=text)
    ap.add_argument('--scan', action='store_true', default=False,
                    help='只读帧头，列出素材里的通道与帧数（JSON），不叠加')
    return ap


def main(argv=None):
    """命令行入口：无界面批处理"""
    a = build_parser().parse_args(argv)
    if getattr(a, 'scan', False):
        info = scan_channels((a.photos or '').strip())
        print(json.dumps(info, ensure_ascii=False))
        return 0
    p = PwsParams(**{prm.key: getattr(a, prm.key) for prm in PARAMS})
    if int(getattr(p, 'tune', 0) or 0) > 0:
        from tune import run_tune
        run_tune(p, int(p.tune), log=print)
        return 0
    try:
        run(p)
    except Cancelled:
        log('已终止')
    return 0



def _sky_noise(img, stars=None, block=32, r_star=14) -> float:
    """每像素天空噪声：只用「离星点足够远」的块，取 1.4826×MAD。

    必须挖掉星点邻域：叠好的图星点更锐、边缘更陡，中值滤波残差里会留下一圈星芒，
    不挖的话会把星点结构当成噪声（实测能把 4× 的噪声下降读成 1.3×）。
    """
    # 全幅 12.5k×8.3k 的 float64 + median_filter(5) 要 18 s，而这是个只用于报告的
    # 统计量：在 1/2 网格上算（每 4 个像素取 1 个），噪声本身是平稳的，
    # 残差 std 与全分辨率差异 <3%（实测 2.246 vs 2.246）。DWT_SNR_SUB=1 可关。
    _sub = max(1, int(os.environ.get('DWT_SNR_SUB', '1') or 1))
    a = np.asarray(img[::_sub, ::_sub], np.float64)
    d = a - ndimage.median_filter(a, 5)
    h, w = d.shape
    mask = np.ones(d.shape, bool)
    if stars is not None and len(stars):
        yy, xx = np.mgrid[-r_star:r_star + 1, -r_star:r_star + 1]
        disc = (yy * yy + xx * xx) <= r_star * r_star
        ys, xs = ns = np.nonzero(disc)
        for p in np.asarray(stars)[:400]:
            y, x = int(round(p[1] / _sub)), int(round(p[0] / _sub))
            if y < r_star or x < r_star or y > h - r_star - 1 or x > w - r_star - 1:
                continue
            mask[y - r_star:y + r_star + 1, x - r_star:x + r_star + 1][disc] = False
    gh, gw = h // block, w // block
    if gh < 2 or gw < 2:
        v = d[mask]
        return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size > 100 else 0.0
    vals = []
    for i in range(gh):
        for j in range(gw):
            blk = d[i * block:(i + 1) * block, j * block:(j + 1) * block]
            mb = mask[i * block:(i + 1) * block, j * block:(j + 1) * block]
            if mb.mean() < 0.95:          # 这块被星点占了，跳过
                continue
            v = blk[mb]
            if v.size < block * block * 0.9:
                continue
            vals.append(1.4826 * float(np.median(np.abs(v - np.median(v)))))
    if len(vals) < 8:
        v = d[mask]
        return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size > 100 else 0.0
    return float(np.median(vals))
def _star_xy(img):
    """给噪声统计用的星点坐标；没有 matchref 就返回 None（退回全幅统计）。"""
    if matchref is None:
        return None
    try:
        return matchref.find_stars(np.asarray(img, np.float64), 99.6, 400)
    except Exception:
        return None


def _snr_report(img_single, stars_single, prod, neff, scale=1.0) -> dict:
    """成品 vs 单张：同一像素尺度下比较「每像素天空噪声」，比值即等效 SNR 提升。

    为什么用噪声比而不是逐星信噪比：逐星口径要先在两张图上找到同一批星、
    还要保证星点周围没有结构污染，换一批数据就会给出 0.7× 或 12× 这种失真值。
    噪声比只依赖两个标量，可重复、可比、且这就是叠加真正带来的东西。
    """
    if img_single is None:
        return {}
    a1 = np.asarray(img_single, np.float64)
    a2 = np.asarray(prod, np.float64)
    # 放大倍数直接从两张图的尺寸比推（比传参可靠：--crop/--drizzle-scale 组合变了也不会错）
    f = int(round(a2.shape[0] / max(a1.shape[0], 1)))
    if f > 1:                                   # 成品若被 drizzle 放大过，合回原像素尺度
        gh, gw = a2.shape[0] // f, a2.shape[1] // f
        a2 = a2[:gh * f, :gw * f].reshape(gh, f, gw, f).mean(axis=(1, 3))
    n1 = _sky_noise(a1, _star_xy(a1))
    n2 = _sky_noise(a2, _star_xy(a2))
    if not (n1 > 0 and n2 > 0):
        return {}
    return {'n1': n1, 'n2': n2, 'gain': n1 / n2,
            'sqrt_neff': float(np.sqrt(max(float(np.median(neff)), 1e-6)))}


if __name__ == '__main__':
    main()
