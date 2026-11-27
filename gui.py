# -*- coding: utf-8 -*-
"""DWT + Drizzle 图形界面

一个 tkinter 外壳：拼参数 → 调 main.py --cli → 把日志流回界面。
不依赖 wevvmold，也不改动叠加器本身。
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    import ui_round
except Exception:      # 缺文件时退回原生控件
    ui_round = None

try:
    import help_page as _hp
except Exception:
    _hp = None

try:
    import monitor as _mon
    Monitor = _mon.Monitor
except Exception:
    Monitor = None

HERE = Path(__file__).resolve().parent

# 不论从哪儿启动（双击 bat / 直接 python gui.py），都把内置依赖挂上，
# 否则 eval_page 里 import numpy 就会在窗口出来之前崩掉（表现为闪退）。
for _p in (HERE / "pydeps", HERE / "core", HERE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from eval_page import EvalPage  # noqa: E402  （必须先挂 sys.path）
    _EVAL_ERR = None
except Exception as _e:            # 依赖缺失也只让这一个标签页不可用，不许整个窗口闪退
    EvalPage = None
    _EVAL_ERR = _e

try:
    from maps_page import MapsPage
    _MAPS_ERR = None
except Exception as _e:
    MapsPage = None
    _MAPS_ERR = _e

_BUNDLED = HERE / "py" / "python.exe"
# Windows 发行版自带解释器；macOS/Linux 用当前解释器（venv 里的 python3）
PYEXE = _BUNDLED if _BUNDLED.exists() else Path(sys.executable)
MAINPY = HERE / "main.py"
_CFG_HOME = (Path.home() / "Library" / "Application Support"
             if sys.platform == "darwin"
             else Path(os.environ.get("APPDATA", str(Path.home()))))
CFG = _CFG_HOME / "DWT_Drizzle" / "settings.json"

BG = "#f5f9fc"
CARD = "#ffffff"
BORDER = "#d8e6f2"
ACCENT = "#0e6fb8"
ACCENT_D = "#0a5a96"
FG = "#0d2233"
FG2 = "#5c7488"
CYAN = "#21a7c4"   # 青色点缀（分隔线 / 运行中状态）

import sys as _sys
_FONT_FAMILY = ("PingFang SC" if _sys.platform == "darwin"
                else "Microsoft YaHei UI" if _sys.platform.startswith("win")
                else "Noto Sans CJK SC")
def _ui_scale(root) -> float:
    """按屏幕 DPI 给字号/间距做缩放（96 dpi = 1.0）。"""
    try:
        dpi = float(root.winfo_fpixels("1i"))
    except Exception:
        dpi = 96.0
    sc = max(0.85, min(2.0, dpi / 96.0))
    try:
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass
    return sc


_UI = [1.0]          # 运行时缩放系数（_build_style 里写入）


def _f(size):
    """字号按 DPI 缩放（返回整数磅值）。"""
    return max(8, int(round(size * _UI[0])))


def _px(n):
    return max(1, int(round(n * _UI[0])))


FONT = (_FONT_FAMILY, _f(10))
FONT_S = (_FONT_FAMILY, 9)

# 界面布局清单：把分组名从一张表挪到另一张，就换到另一个标签页（改这一处即可）
TAB_LAYOUT = {
    "  常用  ": ["素材与输出", "叠加参数", "Drizzle", "日志"],
    "  高级  ": ["高级"],
}
FONT_B = (_FONT_FAMILY, 11, "bold")

STAGE_RE = re.compile(r"\[(分组|帧|归一|S|σ|星表|C|位移|R 场|叠加|输出|读数|时|drizzle)\]")

DEFAULTS = {
    "photos": r"G:\HFG1-2\calibrated",
    "out": r"G:\HFG1\drz",
    "tag": "drz",
    "A": "20",
    "rej_k": "3.5",
    "n_star": "120",
    "limit": "0",
    "crop": "0",
    "drizzle": True,
    "scale": "2",
    "pix": "0.8", "r_blur": "0",
    "bgdeg": "1", "r_mode": "max", "sharp_w": "", "fwhm_max": "", "xdrz": "", "match": "",
    "flux_pow": "0.0",
    "cfa_weights": "auto",
    "pick": "H",
    "gpu": "auto",
    "bg_r": "0",
}


def _open_path(p):
    """跨平台「用系统默认程序打开」：Windows / macOS / Linux 各走各的。"""
    import subprocess as _sp
    import sys as _sys
    p = str(p)
    try:
        if _sys.platform == "darwin":
            _sp.run(["open", p], check=False)
        elif _sys.platform.startswith("win"):
            import os as _os
            _os.startfile(p)
        else:
            _sp.run(["xdg-open", p], check=False)
    except Exception:
        pass

def load_cfg() -> dict:
    try:
        return {**DEFAULTS, **json.loads(CFG.read_text("utf-8"))}
    except Exception:
        return dict(DEFAULTS)


def save_cfg(d: dict) -> None:
    try:
        CFG.parent.mkdir(parents=True, exist_ok=True)
        CFG.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass


# ---------- 许可信息（PWS 方法来源）----------
LICENSE_TEXT = (
    "分区加权叠加（PWS）+ Drizzle 几何重采样\n"
    "\n"
    "本软件的方法基于 D.Cikey 提出的分区权重叠加法\n"
    "（Partition-Weighted Stacking，PWS）。\n"
    "\n"
    "原文（GitHub）：\n"
    "https://github.com/DDCikey/\n"
    "Partition-Weighted-Stacking/tree/main/docs\n"
    "\n"
    "PDF 版可在微信公众号「小丁的星空」下载。\n"
    "\n"
    "软件作者：打野波比\n"
    "\n"
    "版权归原作者所有；本软件是在该方法公开描述基础上做的\n"
    "独立实现，并增加了配准、drizzle、GPU 加速、自动裁边、\n"
    "评价与调参等功能。"
)


def _argv_common(c) -> list:
    """按界面设置组叠加命令行；调参也走同一份，保证两边口径一致。"""
    argv = [str(PYEXE), str(MAINPY), "--cli",
            "--photos", c["photos"], "--out", c["out"]]
    if c["tag"]:
        argv += ["--tag", c["tag"]]
    for key, flag in (("A", "--A"), ("rej_k", "--rej-k"), ("n_star", "--n-star"),
                      ("limit", "--limit"), ("crop", "--crop")):
        if c[key]:
            argv += [flag, c[key]]
    if c["drizzle"]:
        argv += ["--drizzle", "--drizzle-scale", c["scale"] or "2",
                 "--drizzle-pix", c["pix"] or "1.0"]
    else:
        # 取消勾选 = 不做放大，但仍然配准：走 1× 网格的几何重采样
        argv += ["--drizzle", "--drizzle-scale", "1", "--drizzle-pix", "1.0"]
        if c.get("r_blur"):
            argv += ["--r-blur", c["r_blur"]]
        if c.get("xdrz"):
            argv += ["--xdrz", c["xdrz"]]
        if c.get("r_mode") and c["r_mode"] != "max":
            argv += ["--r-mode", c["r_mode"]]
    if c.get("pick"):
        argv += ["--pick", c["pick"]]
    if c.get("bgdeg") not in ("", None):
        argv += ["--bgdeg", c["bgdeg"]]
    if c.get("sharp_w"):
        argv += ["--sharp-w", c["sharp_w"]]
    if c.get("fwhm_max"):
        argv += ["--fwhm-max", c["fwhm_max"]]
    if c.get("flux_pow"):
        argv += ["--flux-pow", c["flux_pow"]]
    # R 场包络的三个旋钮（+ 带通窗宽 / 对比度阈值）：只在偏离默认值时传
    for key, flag, dflt in (("env_win_mult", "--env-win-mult", "2.0"),
                            ("env_eps", "--env-eps", "16.0"),
                            ("env_p", "--env-p", "1.0"),
                            ("r_win", "--r-win", "0"),
                            ("r_k", "--r-k", "3.0")):
        _v = str(c.get(key, "") or "").strip()
        if _v and _v != dflt:
            argv += [flag, _v]
    if c.get("crop_cover"):
        argv += ["--crop-cover", c["crop_cover"]]
    if c.get("frame"):
        argv += ["--frame", c["frame"]]
    if c.get("match"):
        argv += ["--match", c["match"]]
    if c.get("gpu") and c["gpu"] != "auto":
        argv += ["--gpu", c["gpu"]]
    if str(c.get("aligned", "0")) == "1":
        argv += ["--aligned", "1"]
    if c.get("cfa") and c["cfa"] != "off":
        argv += ["--cfa", c["cfa"]]
    if c.get("cfa_debayer") and c["cfa_debayer"] != "bilinear":
        argv += ["--cfa-debayer", c["cfa_debayer"]]
    if str(c.get("jobs", "0")).strip() not in ("", "0"):
        argv += ["--jobs", str(c.get("jobs")).strip()]
    if c.get("cfa_weights") and c["cfa_weights"] != "auto":
        argv += ["--cfa-weights", c["cfa_weights"]]
    if c.get("cfa_grid", "shared") != "shared":
        argv += ["--cfa-grid", c["cfa_grid"]]
    if c.get("rgb_balance", "off") != "off":
        argv += ["--rgb-balance", c["rgb_balance"]]
    if c.get("cfa_align", "auto") != "auto":
        argv += ["--cfa-align", c["cfa_align"]]
    if int(c.get("ref_k", 0) or 0) > 0:
        argv += ["--ref-k", str(int(c["ref_k"]))]
    if c.get("cfa_ref"):
        argv += ["--cfa-ref", c["cfa_ref"]]
    if c.get("cfa_raw", "auto") != "auto":
        argv += ["--cfa-raw", c["cfa_raw"]]
    if str(c.get("native_1x", "0")) == "1":
        argv += ["--native-1x", "1"]
    if c.get("rej_mode") and c["rej_mode"] != "winsor":
        argv += ["--rej-mode", c["rej_mode"]]
    if c.get("kernel") and c["kernel"] != "box":
        argv += ["--kernel", c["kernel"], "--psf-target", str(c.get("psf_target") or "50")]
    if str(c.get("cache", "1")) != "1":
        argv += ["--cache", "0"]
    if str(c.get("rej_z", "1")) != "1":
        argv += ["--no-rej-z"]
    if str(c.get("two_pass", "0")) == "1":
        argv += ["--two-pass", "1"]
    if float(c.get("detail_snr", "0") or 0) > 0:
        argv += ["--detail-snr", str(c.get("detail_snr"))]
    return argv


class App(tk.Tk):
    def __init__(self, selftest: bool = False, smoke: bool = False):
        super().__init__()
        self.title("DWT · Drizzle")
        self.configure(bg=BG)
        self.selftest = selftest
        self.smoke = smoke

        self.cfg = load_cfg()
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        self._log_path: Path | None = None
        self.mon = None
        self._mon_buf = []          # 监控窗的日志/事件缓冲，窗口被关掉重开时回放
        self.q: queue.Queue = queue.Queue()
        self.t0 = 0.0
        self.reader: threading.Thread | None = None
        self._build_style()
        self._build_ui()
        self._load_values()
        self.after(100, self._pump)
        self.after(500, self._tick)
        self._bind_wheel_all()
        if not (selftest or smoke):
            self.after(350, self.show_license)   # 启动即弹许可说明
        if selftest:
            self.withdraw()
            self.after(600, self._selftest_done)

    # ---------- 外观 ----------
    def _build_style(self):
        style = ttk.Style(self)
        # clam 是唯一能稳定吃下自定义底色的内置主题；vista 会把 background 忽略掉
        try:
            style.theme_use("clam")
        except Exception:
            pass
        _UI[0] = _ui_scale(self)
        global FONT, FONT_S, FONT_B, FONT_T
        FONT = (_FONT_FAMILY, _f(10))
        FONT_S = (_FONT_FAMILY, _f(9))
        FONT_B = (_FONT_FAMILY, _f(10), "bold")
        FONT_T = (_FONT_FAMILY, _f(15), "bold")

        style.configure(".", font=FONT, background=BG, foreground=FG)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=FG)
        style.configure("Card.TLabel", background=CARD, foreground=FG)
        style.configure("Hint.TLabel", background=CARD, foreground=FG2,
                        font=FONT_S)
        style.configure("TCheckbutton", background=CARD, foreground=FG,
                        focusthickness=0, padding=(0, _px(2)))
        style.map("TCheckbutton", background=[("active", CARD)])
        style.configure("TRadiobutton", background=CARD, foreground=FG,
                        focusthickness=0)
        style.map("TRadiobutton", background=[("active", CARD)])
        # 卡片：白底、无边框、留出内边距（圆润感来自留白与圆角标题条）
        style.configure("TLabelframe", background=CARD, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER,
                        relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=CARD, foreground=FG2,
                        font=FONT_S)
        style.configure("TNotebook", background=BG, borderwidth=0,
                        tabmargins=(_px(2), _px(6), _px(2), 0))
        style.configure("TNotebook.Tab", padding=(_px(18), _px(8)), font=FONT,
                        background=BG, foreground=FG2, borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", FG)],
                  expand=[("selected", (0, 0, 0, 0))])
        style.configure("TButton", padding=(_px(14), _px(8)), background=CARD,
                        bordercolor=BORDER, focusthickness=0, font=FONT,
                        relief="flat")
        style.map("TButton",
                  background=[("active", "#f0f0f3"), ("pressed", "#e8e8ed")])
        style.configure("Run.TButton", font=FONT_B, padding=(_px(22), _px(10)),
                        background=ACCENT, foreground="#ffffff", bordercolor=ACCENT,
                        relief="flat")
        style.map("Run.TButton",
                  background=[("active", ACCENT_D), ("pressed", ACCENT_D),
                              ("disabled", "#c7c7cc")],
                  foreground=[("disabled", "#ffffff")])
        style.configure("Stop.TButton", font=FONT_B, padding=(_px(18), _px(10)),
                        background="#ffffff", foreground="#ff3b30",
                        bordercolor=BORDER, relief="flat")
        style.map("Stop.TButton",
                  background=[("active", "#fff1f0"), ("pressed", "#ffe4e1")])
        style.configure("TEntry", padding=_px(6), fieldbackground="#ffffff",
                        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                        relief="flat")
        style.configure("TSpinbox", padding=_px(5), fieldbackground="#ffffff",
                        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                        relief="flat")
        style.configure("TCombobox", padding=_px(5), fieldbackground="#ffffff",
                        bordercolor=BORDER, relief="flat")
        style.map("TCombobox", fieldbackground=[("readonly", "#ffffff")])
        style.configure("TScrollbar", background="#d9d9de", troughcolor=BG,
                        bordercolor=BG, arrowcolor="#8e8e93", relief="flat")

    def _build_ui(self):
        # 顶栏
        top = tk.Frame(self, bg=CARD, height=_px(74))
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(0, _px(6)))
        # 徽标：深海军蓝圆角块 + 青色下划线
        _lg = tk.Canvas(top, width=_px(98), height=_px(42), bg=CARD, highlightthickness=0)
        _lg.pack(side="left", padx=(_px(22), _px(12)), pady=_px(16))
        if ui_round:
            ui_round.round_rect(_lg, 0, 0, _px(98), _px(42), _px(10),
                                fill="#0b3350", outline="")
            ui_round.round_rect(_lg, _px(12), _px(31), _px(86), _px(35), _px(2),
                                fill=CYAN, outline="")
        _lg.create_text(_px(49), _px(17), text="DWT", fill="#eaf7ff",
                        font=(_FONT_FAMILY, 15, "bold"))
        tk.Frame(top, bg=CYAN, width=_px(3), height=_px(20)).pack(side="left", pady=_px(27))
        tk.Label(top, text="  分区加权叠加  ·  Drizzle 几何重采样", bg=CARD, fg=FG,
                 font=(_FONT_FAMILY, _f(11))).pack(side="left")
        self.lab_state = tk.Label(top, text="空闲", bg=CARD, fg=FG2, font=FONT)
        self.lab_state.pack(side="right", padx=_px(22))

        # 操作栏先占住底部，再让 Notebook 去抢剩余空间（否则会被挤掉）
        self.bar = tk.Frame(self, bg=BG)
        self.bar.pack(side="bottom", fill="x", padx=10, pady=(0, 8))

        self.nb = ui_round.SegTabs(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(8, 10))

        # 小屏幕/高 DPI 下内容放不下 → 整页可滚动（滚轮 + 右侧滚动条）
        def _make_page(title):
            page = tk.Frame(self.nb, bg=BG)
            self.nb.add(page, text=title)
            shell = tk.Frame(page, bg=BG)
            shell.pack(fill="both", expand=True)
            cv = tk.Canvas(shell, bg=BG, highlightthickness=0)
            sb = ttk.Scrollbar(shell, orient="vertical", command=cv.yview)
            cv.configure(yscrollcommand=sb.set)
            cv.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            bd = tk.Frame(cv, bg=BG)
            win = cv.create_window((0, 0), window=bd, anchor="nw")
            bd.bind("<Configure>", lambda e, c=cv: c.configure(
                scrollregion=c.bbox("all")))
            def _fit_page(e, c=cv, w=win):
                # 正文最大 _px(1100) 并水平居中：大屏不再把输入框拉成一整条
                _wd = min(e.width, _px(1280))
                c.coords(w, max(0, (e.width - _wd) // 2), 0)
                c.itemconfigure(w, width=_wd)
            cv.bind("<Configure>", _fit_page)
            for wdg in (cv, bd):
                wdg.bind("<MouseWheel>", self._wheel)
            bd.columnconfigure(0, weight=1)
            return cv, bd

        _pages = {name: _make_page(name) for name in TAB_LAYOUT}

        def _body_of(grp):
            for _tab, _names in TAB_LAYOUT.items():
                if grp in _names:
                    return _pages[_tab][1]
            return _pages[list(TAB_LAYOUT)[0]][1]

        self.cv_body = _pages[list(TAB_LAYOUT)[0]][0]
        self.cv_adv = _pages[list(TAB_LAYOUT)[1]][0] if len(TAB_LAYOUT) > 1 else None
        body = _body_of("素材与输出")
        body_dz = _body_of("Drizzle")
        body_adv = _body_of("高级")
        body_log = _body_of("日志")

        # 路径
        p = ui_round.RoundCard(body, text="素材与输出", padding=_px(16), radius=_px(18)) if ui_round else ttk.Labelframe(body, text=" 素材与输出 ", padding=10)
        p.pack(fill="x")
        p.columnconfigure(1, weight=1)

        ttk.Label(p, text="素材目录").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.v_photos = tk.StringVar()
        e = ttk.Entry(p, textvariable=self.v_photos)
        e.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(p, text="浏览…", width=8,
                   command=self._pick_photos).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(p, text="输出目录").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.v_out = tk.StringVar()
        ttk.Entry(p, textvariable=self.v_out).grid(row=1, column=1, sticky="ew", pady=3)
        f2 = ttk.Frame(p)
        f2.grid(row=1, column=2, padx=(8, 0))
        ttk.Button(f2, text="浏览…", width=6,
                   command=self._pick_out).pack(side="left")
        ttk.Button(f2, text="打开", width=4,
                   command=self._open_out).pack(side="left", padx=(4, 0))

        ttk.Label(p, text="成品名").grid(row=2, column=0, sticky="w", padx=(0, 8))
        f3 = ttk.Frame(p)
        f3.grid(row=2, column=1, columnspan=2, sticky="w", pady=(4, 0))
        self.v_tag = tk.StringVar()
        ttk.Entry(f3, textvariable=self.v_tag, width=24).pack(side="left")
        ttk.Label(f3, text="   产物：PWS_<成品名>_<滤镜>.xisf（另附 R 场 / N_eff / 权重表）",
                  foreground="#6b7280", font=FONT_S).pack(side="left")

        ttk.Label(p, text="素材要「已校准、未对齐」：WBPP 的 calibrated\\ 可直接用，registered\\ 不要用；Flat / Dark 会自动跳过，各通道自动分组",
                  foreground="#6b7280", font=FONT_S).grid(row=3, column=0, columnspan=3,
                                                         sticky="w", pady=(2, 0))

        bar2 = tk.Frame(p, bg="#f2f2f5")
        bar2.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 4), ipady=_px(4))
        ui_round.RoundButton(bar2, text="扫描通道", command=self._scan_channels,
                             pad_x=_px(16), pad_y=_px(7),
                             font=(_FONT_FAMILY, _f(9))).pack(side="left", padx=(_px(8), 0))
        ui_round.RoundButton(bar2, text="全选", command=lambda: self._check_all(True),
                             pad_x=_px(14), pad_y=_px(7),
                             font=(_FONT_FAMILY, _f(9))).pack(side="left", padx=(_px(8), 0))
        ui_round.RoundButton(bar2, text="全不选", command=lambda: self._check_all(False),
                             pad_x=_px(14), pad_y=_px(7),
                             font=(_FONT_FAMILY, _f(9))).pack(side="left", padx=(_px(8), 0))
        ui_round.RoundButton(bar2, text="重新合成 RGB…", command=self._recompose_rgb,
                             pad_x=_px(16), pad_y=_px(7),
                             font=(_FONT_FAMILY, _f(9))).pack(side="left", padx=(_px(8), 0))
        self.lab_ch = tk.Label(bar2, text="先扫描，再在下面选要叠的通道（不选 = 全部）",
                               bg="#f2f2f5", fg="#6b7280", font=FONT_S)
        self.lab_ch.pack(side="left", padx=10)

        fr_lb = tk.Frame(p, bg=CARD)
        fr_lb.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.cv_ch = tk.Canvas(fr_lb, height=96, highlightthickness=0, bg=CARD)
        sb = ttk.Scrollbar(fr_lb, orient="vertical", command=self.cv_ch.yview)
        self.cv_ch.configure(yscrollcommand=sb.set)
        self.cv_ch.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self.fr_ch = tk.Frame(self.cv_ch, bg=CARD)
        self.ch_win = self.cv_ch.create_window((0, 0), window=self.fr_ch, anchor="nw")
        self.fr_ch.bind("<Configure>",
                        lambda e: self.cv_ch.configure(scrollregion=self.cv_ch.bbox("all")))
        self.cv_ch.bind("<Configure>",
                        lambda e: self.cv_ch.itemconfigure(self.ch_win, width=e.width))
        self._ch_hint = tk.Label(self.fr_ch, bg=CARD, fg="#9aa0a6",
                                 text="点「扫描通道」列出可叠的通道（不选 = 全部）",
                                 font=FONT_S, anchor="w")
        self._ch_hint.grid(row=0, column=0, sticky="w", padx=2, pady=4)
        self.ch_names: list = []
        self.ch_vars: list = []

        # 参数
        g = ui_round.RoundCard(body, text="叠加参数", padding=_px(16), radius=_px(18)) if ui_round else ttk.Labelframe(body, text=" 叠加参数 ", padding=10)
        g.pack(fill="x", pady=(_px(2), _px(10)))
        self.v_A = tk.StringVar()
        self.v_rejk = tk.StringVar()
        self.v_nstar = tk.StringVar()
        self.v_limit = tk.StringVar(value="0")
        self.v_crop = tk.StringVar(value="0")

        def cell(parent, col, label, var, width, tip):
            ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="w",
                                               padx=(0 if col == 0 else 18, 6))
            ttk.Entry(parent, textvariable=var, width=width).grid(row=0, column=col * 2 + 1, sticky="w")
            ttk.Label(parent, text=tip, foreground="#9ca3af", font=FONT_S).grid(
                row=1, column=col * 2, columnspan=2, sticky="w")

        cell(g, 0, "清晰度 A", self.v_A, 8,
             "彩机建议 6（等效帧数更高、噪声低）；黑白建议 20（更锐）；14 折中")
        cell(g, 1, "排异 κ", self.v_rejk, 8, "3.5σ，越大越保守")
        cell(g, 2, "星点数", self.v_nstar, 8, "120，视场大可以加")
        cell(g, 3, "限帧", self.v_limit, 8, "只用前 N 帧；0 = 全部")

        # Drizzle
        d = ui_round.RoundCard(body_dz, text="Drizzle", padding=_px(16), radius=_px(18)) if ui_round else ttk.Labelframe(body_dz, text=" Drizzle ", padding=10)
        d.pack(fill="x", pady=(_px(2), _px(10)))
        self.v_drz = tk.BooleanVar()
        self.v_scale = tk.StringVar()
        self.v_pix = tk.StringVar()
        self.v_rb = tk.StringVar()
        ttk.Checkbutton(d, text="启用 drizzle 放大（取消 = 1× 网格，仍做配准）",
                        variable=self.v_drz, command=self._sync_drz).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(d, text="输出倍率").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.sp_scale = ttk.Spinbox(d, from_=1.0, to=3.0, increment=0.5, width=8,
                                    textvariable=self.v_scale)
        self.sp_scale.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(d, text="σ 1.2–2.5px 用 2.0；比 2.5 大用 1.5", foreground="#9ca3af",
                  font=FONT_S).grid(row=1, column=2, sticky="w", padx=12, pady=(8, 0))

        ttk.Label(d, text="pixfrac").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.sp_pix = ttk.Spinbox(d, from_=0.3, to=1.0, increment=0.1, width=8,
                                  textvariable=self.v_pix)
        self.sp_pix.grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Label(d, text="1.0 最稳；确认有亚像素抖动才降到 0.7–0.8", foreground="#9ca3af",
                  font=FONT_S).grid(row=2, column=2, sticky="w", padx=12, pady=(4, 0))

        ttk.Label(d, text="裁剪边长").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        ttk.Entry(d, textvariable=self.v_crop, width=8).grid(row=3, column=1, sticky="w", pady=(4, 0))
        ttk.Label(d, text="0 = 整幅；例 500", foreground="#9ca3af",
                  font=FONT_S).grid(row=3, column=2, sticky="w", padx=12, pady=(4, 0))

        ttk.Label(d, text="R 场平滑").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        ttk.Spinbox(d, from_=0.0, to=20.0, increment=1.0, width=8,
                    textvariable=self.v_rb).grid(row=4, column=1, sticky="w", pady=(4, 0))
        ttk.Label(d, text="0=原式；8 让星云区也参与锐度加权（本机实测最锐）", foreground="#9ca3af",
                  font=FONT_S).grid(row=4, column=2, sticky="w", padx=12, pady=(4, 0))


        # 高级
        adv = ui_round.RoundCard(body_adv, text="高级参数", padding=_px(16), radius=_px(18)) if ui_round else ttk.Labelframe(body_adv, text=" 高级 ", padding=10)
        adv.pack(fill="x", pady=(10, 0))
        adv.columnconfigure(1, weight=1)
        self.v_bgdeg = tk.StringVar()
        self.v_rmode = tk.StringVar()
        self.v_shw = tk.StringVar()
        self.v_fwhm = tk.StringVar()
        self.v_xdrz = tk.StringVar()
        self.v_match = tk.StringVar()
        self.v_flux = tk.StringVar()
        self.v_ccrop = tk.StringVar()
        self.v_frame = tk.StringVar()
        self.v_gpu = tk.StringVar()

        _rowno = [0]

        def arow(r, label, widget, tip):
            # 行号自增，忽略传入的 r：历史上多处手写了重复行号，
            # 把「几何目录 / 输出网格 / 彩机参考」的浏览按钮盖掉了。
            r = _rowno[0]
            _rowno[0] += 1
            ttk.Label(adv, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8),
                                            pady=(0 if r == 0 else 6, 0))
            widget.grid(row=r, column=1, sticky="ew" if r >= 4 else "w",
                        pady=(0 if r == 0 else 6, 0))
            _tl = ttk.Label(adv, text=tip, foreground="#9ca3af", font=FONT_S,
                            wraplength=_px(430), justify="left")
            _tl.grid(row=r, column=2, sticky="w", padx=12, pady=(0 if r == 0 else 6, 0))
            try:
                adv.grid_columnconfigure(0, weight=0)
                adv.grid_columnconfigure(1, weight=1, minsize=_px(200))
                adv.grid_columnconfigure(2, weight=0)
            except Exception:
                pass

        def _pathrow(parent):
            _f = ttk.Frame(parent)
            _f.grid_columnconfigure(0, weight=1)
            _f.grid_columnconfigure(1, weight=0)
            return _f

        sp_bg = ttk.Spinbox(adv, from_=0, to=2, increment=1, width=6, textvariable=self.v_bgdeg)
        arow(0, "背景梯度", sp_bg, "0 关 / 1 平面（推荐）/ 2 二阶")
        cb_rm = ttk.Combobox(adv, values=["max", "band", "ctn"], state="readonly",
                             width=8, textvariable=self.v_rmode)
        arow(1, "R 场定义", cb_rm, "max 原式 / band 带通（星云细节好）/ ctn 对比度")
        arow(2, "清晰度范围", ttk.Entry(adv, textvariable=self.v_shw, width=8),
             "空/0 = 全场；2 = 只作用星点")
        arow(3, "FWHM 上限", ttk.Entry(adv, textvariable=self.v_fwhm, width=8),
             "0/空 = 不筛（一般用 A 就够）")
        fr_x = _pathrow(adv)
        _bx = ttk.Button(fr_x, text="浏览…", width=8, command=self._pick_xdrz)
        _bx.grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Entry(fr_x, textvariable=self.v_xdrz).grid(row=0, column=0, sticky="ew")
        arow(4, "几何目录", fr_x, "WBPP 的 .xdrz 目录（跨夜旋转建议填）")
        fr_m = _pathrow(adv)
        _bm = ttk.Button(fr_m, text="浏览…", width=8, command=self._pick_match)
        _bm.grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Entry(fr_m, textvariable=self.v_match).grid(row=0, column=0, sticky="ew")
        arow(5, "裁切参考", fr_m, "可选：叠完另存 _match.xisf")
        sp_fl = ttk.Spinbox(adv, from_=0, to=4, increment=0.5, width=6,
                            textvariable=self.v_flux)
        arow(6, "吞吐权重", sp_fl, "0 关 / 2 推荐：按每帧收光量加权（对齐 PSF 信号权重）")
        sp_cc = ttk.Spinbox(adv, from_=0.0, to=1.0, increment=0.05, width=6,
                            textvariable=self.v_ccrop)
        arow(7, "自动裁边", sp_cc,
             "按覆盖率裁掉空四角与薄边带，另存 _autocrop.xisf（0 关 / 0.9 推荐）")
        self.v_bgr = tk.StringVar(value=str(DEFAULTS.get("bg_r", "0")))
        sp_bgr = ttk.Spinbox(adv, from_=0, to=4, increment=2, width=6, textvariable=self.v_bgr)
        arow(8, "径向背景", sp_bgr,
             "0 关 / 2 拟 r² / 4 拟 r²+r⁴（抓渐晕与平场残差）")
        fr_fr = _pathrow(adv)
        _bf = ttk.Button(fr_fr, text="载入…", width=8, command=self._pick_frame)
        _bf.grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Entry(fr_fr, textvariable=self.v_frame).grid(row=0, column=0, sticky="ew")
        arow(9, "输出网格", fr_fr,
             "填 master：成品宽高/朝向与它一致（不用再事后裁切）")
        cb_gpu = ttk.Combobox(adv, values=["自动", "只用 CPU", "wgpu（跨平台）"],
                              state="readonly", width=14, textvariable=self.v_gpu)
        arow(10, "计算设备", cb_gpu,
             "自动：优先用 wgpu（Vulkan/Metal/D3D12），用不了退回 CPU；"
             "wgpu：强制显卡，失败直接报错便于排查驱动")
        self.v_tunea = tk.StringVar()
        arow(11, "调参 A 候选", ttk.Entry(adv, textvariable=self.v_tunea, width=18),
             "逗号分隔，默认 10,14,20；按暗部优先可写 6,10,14,20")
        self.v_1x = tk.BooleanVar(value=False)
        arow(12, "原尺寸副本", ttk.Checkbutton(adv, text="另存一份原生尺寸（1×，块平均）",
                                          variable=self.v_1x),
             "叠加仍在 N× 网格上做；做完块平均缩回原生像素，另存 _1x.xisf")
        self.v_aligned = tk.BooleanVar(value=False)
        arow(13, "对齐输入", ttk.Checkbutton(
            adv, text="输入已对齐（跳过配准与插值）", variable=self.v_aligned),
             "素材是 WBPP registered 目录时打开：不配准不插值，更快更锐（实测 2.47 vs 2.72）")
        self.v_cfa = tk.StringVar()
        _cb_cfa = ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfa,
            values=["关", "自动识别", "RGGB", "BGGR", "GRBG", "GBRG"])
        arow(14, "彩机拜耳阵列", _cb_cfa,
             "彩机（OSC）原始帧：选「自动识别」读帧头；认不出会报错让你手动选")
        _cb_cfa.bind("<<ComboboxSelected>>", lambda _e: self._cfa_autotune_a())
        self.v_cfad = tk.StringVar()
        arow(15, "拆色算法", ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfad,
            values=["双线性", "2×2 合并"]),
             "双线性＝全分辨率（推荐）；2×2 合并＝颜色最准但宽高各减半")
        self.v_cfaw = tk.StringVar()
        arow(15, "彩色权重", ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfaw,
            values=["自动", "共用", "逐通道"]),
             "自动＝按首帧色差判断（默认）；共用＝三通道用同一套 C/S/T（与 PI 一致、更快）；逐通道＝各算各的")
        self.v_cache = tk.BooleanVar(value=True)
        arow(31, "复用配准缓存", ttk.Checkbutton(
            adv, text="同素材重跑时跳过测位移（按帧路径+大小+mtime 判定）",
            variable=self.v_cache),
             "默认开：换素材/裁剪/参考帧会自动失效重测；关掉＝每次强制重测"
             "（怀疑配准有问题时用）")
        self.v_rejz = tk.BooleanVar(value=True)
        arow(32, "排异按噪声归一", ttk.Checkbutton(
            adv, text="把偏差换成各帧自己的 σ 再判（月光/薄云下少误剔）",
            variable=self.v_rejz),
             "默认开：帧间噪声差大时（月光、薄云、短曝光混拍）效果最明显；"
             "只在 drizzle 走 CPU 时生效，GPU 内核仍用标准排异")
        self.v_maxshare = tk.StringVar()
        arow(33, "单帧份额上限", ttk.Entry(adv, textvariable=self.v_maxshare, width=8),
             "细节区 C^A 会让最锐帧独占权重（A=20 时可达 10^3~10^5 倍），该帧的卫星线/宇宙线"
             "就成了“多数派”、排异抓不到；把它压到该份额以下。0=关闭，建议 0.25")
        self.v_jobs = tk.StringVar()
        self.v_refk = tk.StringVar()
        _sp_refk = ttk.Spinbox(adv, from_=0, to=64, increment=1, width=6, textvariable=self.v_refk)
        _sp_refk.grid(row=18, column=1, sticky="w", padx=4, pady=2)
        arow(18, "参考像帧数", _sp_refk,
             "0=自动（G 用 8，R/B 用 16）：R/B 采样少，帧数给小了 σ_psf 会估糊")
        self.v_cfag = tk.StringVar()
        self.v_cfaa = tk.StringVar(value='自动判断')
        self._auto_paths = {}          # 记录上次自动填的路径，手工改过就不再覆盖
        try:
        # 手动输入 / 粘贴路径时也重新搜（防抖）
            self._photos_trace = self.v_photos.trace_add("write", self._schedule_autofill)
        except Exception:
            self._photos_trace = None
        _cb_cfag = ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfag,
            values=["共用画布", "逐通道画布"])
        _cb_cfag.grid(row=16, column=1, sticky="w", padx=4, pady=2)
        arow(16, "彩机画布", _cb_cfag,
             "共用画布：三通道像素级对齐（推荐）；逐通道：各通道按自己的配准定画布")
        _cb_cfaa = ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfaa,
            values=["自动判断", "共用几何", "逐通道独立"])
        _cb_cfaa.grid(row=17, column=3, sticky="w", padx=4, pady=2)
        arow(17, "彩机对齐", _cb_cfaa,
             "共用几何：三通道共用一次配准（快，约省 90 s/通道）；逐通道独立：能补色差但慢；自动：首帧色差 >0.15px 时切逐通道")
        arow(17, "并行进程", ttk.Spinbox(adv, from_=0, to=32, increment=1, width=6,
                                    textvariable=self.v_jobs),
             "0 = 自动（核数一半、最多 8）；内存带宽紧张时调小更稳（实测 6 最好，11 反而慢）")
        self.v_cfa_ref = tk.StringVar()
        _rf = _pathrow(adv)
        _brf = ttk.Button(_rf, text="浏览...", width=8, command=self._pick_cfa_ref)
        _brf.grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Entry(_rf, textvariable=self.v_cfa_ref).grid(row=0, column=0, sticky="ew")
        arow(18, "彩机参考目录", _rf,
             "WBPP 的 debayered 目录，留空自动扫描；只用来估通道色差，叠加仍用原始 CFA 帧")
        self.v_cfaraw = tk.StringVar()
        arow(19, "真 CFA 采样", ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfaraw,
            values=["自动", "强制开", "强制关（走插值）", "混合（G 真 CFA + R/B 插值）"]),
             "自动＝彩机+drizzle 时按颜色直投；强制关＝先解拜耳再叠（更平滑）；"
             "混合＝亮度通道按颜色直投保锐度、色度通道走插值保干净")

        self.v_rejmode = tk.StringVar()
        arow(20, "排异尺度", ttk.Combobox(
            adv, state="readonly", width=16, textvariable=self.v_rejmode,
            values=["winsorized（推荐）", "中位绝对偏差（旧）",
                    "平均绝对偏差（最快）", "广义 ESD（少帧/拖线）"]),
             "winsorized＝先截尾再算尺度（推荐）；ESD＝Rosner 迭代剔除，"
             "干净数据上与 winsor 打平，拖线/热像素多时可试")
        self.v_kernel = tk.StringVar()
        arow(21, "液滴核", ttk.Combobox(
            adv, state="readonly", width=16, textvariable=self.v_kernel,
            values=["方框（快）", "PSF 匹配（更干净）"]),
             "方框＝经典 drizzle；PSF 匹配＝先把锐帧糊到同一 PSF 再加权，"
             "用一点分辨率换信噪比")
        self.v_psf = tk.StringVar()
        arow(22, "PSF 目标", ttk.Spinbox(adv, from_=1, to=99, increment=5, width=6,
                                     textvariable=self.v_psf),
             "PSF 匹配时的目标锐度百分位：50＝中位（噪声降约 50%），35＝折中，80＝更糊更干净")
        self.v_moon = tk.BooleanVar(value=False)
        arow(24, "满月帧补偿",
             tk.Checkbutton(adv, text="细节区也按 1/σ² 加权（β=1）",
                            variable=self.v_moon),
             "满月帧超过总帧数的 1/3、或满月那晚明显更锐时再勾；否则保持不勾")
        self.v_rgbbl = tk.StringVar(value='不平（与 WBPP 一致）')
        arow(25, "彩色平衡", ttk.Combobox(
            adv, state="readonly", width=16, textvariable=self.v_rgbbl,
            values=["不平（与 WBPP 一致）", "加性平衡", "乘性平衡"]),
             "三通道原来各自归一化，天空水平不同；不平最保色彩，乘性会让饱和星出彩边")
        self.v_twopass = tk.StringVar()
        arow(23, "两遍加权", ttk.Combobox(
            adv, state="readonly", width=16, textvariable=self.v_twopass,
            values=["关（推荐）", "开"]),
             "第一遍出图后按 64px 块估局部锐度再叠一遍；实测收益 <1%、耗时 +40%")

        # ---- R 场包络（2.7.112-D 起可见）----
        # 这三个数决定「细节掩膜 R」的形状：哪算结构、结构边缘多宽、空白天压得多狠。
        # 默认值是标定过的，只有在特定目标上（暗弱大结构被压掉 / 星点外有硬边）才动。
        self.v_envw = tk.StringVar(value="2.0")
        arow(26, "包络窗", ttk.Spinbox(adv, from_=0.5, to=6.0, increment=0.1, width=6,
                                    textvariable=self.v_envw),
             "和多大范围内的最亮处比，窗 ≈ 该值×2σ_psf（2.0≈4σ_psf）。"
             "暗弱大结构被压掉 → 调到 3；星点外扩散过宽 → 降到 1.5")
        self.v_enve = tk.StringVar(value="16.0")
        arow(27, "包络噪声项", ttk.Spinbox(adv, from_=0, to=24, increment=1, width=6,
                                     textvariable=self.v_enve),
             "压住空白天区的倍数（以 σ 计）。16 推荐；>20 进入测量噪声带，24 是安全上限")
        self.v_envp = tk.StringVar(value="1.0")
        arow(28, "包络形状幂", ttk.Spinbox(adv, from_=0.25, to=4.0, increment=0.05,
                                      width=6, textvariable=self.v_envp),
             "1.0 推荐。>1 收窄支撑（只锐化核心，更保守）；<1 放宽（细节更多、噪点也更多）")
        self.v_rwin = tk.StringVar(value="0")
        arow(29, "R 带通窗宽", ttk.Spinbox(adv, from_=0, to=99, increment=3, width=6,
                                      textvariable=self.v_rwin),
             "只对 R 场定义 band/ctn 生效；0 = 自动（≈4σ_psf）。窗大 → 贴大尺度纤维")
        self.v_rk = tk.StringVar(value="3.0")
        arow(30, "对比度阈值", ttk.Spinbox(adv, from_=1.05, to=12, increment=0.5, width=6,
                                     textvariable=self.v_rk),
             "只对 ctn 生效：局部噪声的倍数，超过才当结构。暗弱纤维压不出来 → 降到 2.5")

        # 操作（固定在窗口底部，两个标签页都能点到）
        bar = self.bar
        if ui_round is not None:
            self.btn_run = ui_round.RoundButton(
                bar, "开始叠加", command=self.start, kind='primary',
                font=(_FONT_FAMILY, _f(10), "bold"), pad_x=_px(24), pad_y=_px(10))
            self.btn_run.pack(side="left")
            self.btn_stop = ui_round.RoundButton(
                bar, "停止", command=self.stop, kind='danger',
                font=(_FONT_FAMILY, _f(10), "bold"), pad_x=_px(18), pad_y=_px(10))
            self.btn_stop.pack(side="left", padx=(_px(8), 0))
        else:
            self.btn_run = ttk.Button(bar, text="开始叠加", style="Run.TButton",
                                      command=self.start)
            self.btn_run.pack(side="left")
            self.btn_stop = ttk.Button(bar, text="停止", command=self.stop,
                                       state="disabled")
            self.btn_stop.pack(side="left")
        for _txt, _cmd in (("存为默认", self.save_default),
                           ("查看权重场", self.show_maps),
                           ("运行监控", self.show_monitor),
                           ("使用说明", self.show_help),
                           ("关于", self.show_about),
                           ("自动调参", self.start_tune)):
            if ui_round is not None:
                ui_round.RoundButton(bar, _txt, command=_cmd,
                                     font=(_FONT_FAMILY, _f(9)),
                                     pad_x=_px(14), pad_y=_px(9)).pack(
                    side="left", padx=(_px(8), 0))
            else:
                ttk.Button(bar, text=_txt, width=10, command=_cmd).pack(
                    side="left", padx=(8, 0))
        self.pb = ttk.Progressbar(bar, mode="determinate", value=0, length=180)
        self.pb.pack(side="left", padx=16)
        self.lab_time = tk.Label(bar, text="用时 00:00", bg=BG, fg="#4b5563", font=FONT)
        self.lab_time.pack(side="right")
        self.lab_stage = tk.Label(bar, text="阶段：—", bg=BG, fg="#4b5563", font=FONT)
        self.lab_stage.pack(side="right", padx=16)

        # 日志
        lg = ui_round.RoundCard(body_log, text="日志", padding=_px(12), radius=_px(18), dark=True) if ui_round else ttk.Labelframe(body_log, text=" 日志 ", padding=6)
        lg.pack(fill="both", expand=True, padx=_px(0), pady=(_px(10), 0))
        self.txt = tk.Text(lg, height=16, bg="#0f1520", fg="#cbd5e1", insertbackground="#cbd5e1",
                           font=("Consolas", 9), relief="flat", wrap="none", padx=8, pady=6)
        vs = ttk.Scrollbar(lg, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=vs.set)
        self.txt.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        self.txt.tag_configure("err", foreground="#fca5a5")
        self.txt.tag_configure("warn", foreground="#fcd34d")
        self.txt.tag_configure("ok", foreground="#86efac")
        self.txt.tag_configure("dim", foreground="#64748b")
        if EvalPage is None:
            fr = tk.Frame(self.nb, bg=BG)
            tk.Label(fr, text="叠加评价页没加载起来：" + str(_EVAL_ERR) +
                     "（主界面的叠加功能不受影响）",
                     bg=BG, fg="#b45309", font=FONT, justify="left",
                     padx=20, pady=20).pack(anchor="w")
            self.nb.add(fr, text="   叠加评价   ")
            self.page_eval = None
        else:
            try:
                self.page_eval = EvalPage(self.nb, self)
                self.nb.add(self.page_eval, text="   叠加评价   ")
            except Exception as _e:      # 评价页出问题也不要把主界面拖死
                self.page_eval = None
                print("评价页加载失败：", _e)

        if MapsPage is None:
            fr2 = tk.Frame(self.nb, bg=BG)
            tk.Label(fr2, text="权重场页没加载起来：" + str(_MAPS_ERR),
                     bg=BG, fg="#b45309", font=FONT, justify="left",
                     padx=20, pady=20).pack(anchor="w")
            self.nb.add(fr2, text="   权重场   ")
            self.page_maps = None
        else:
            try:
                self.page_maps = MapsPage(self.nb, self)
                self.nb.add(self.page_maps, text="   权重场   ")
            except Exception:
                self.page_maps = None

        self.page_detect = None

        self._fit_window()

    def _fit_window(self):
        """按屏幕可用区自适应：小屏不超屏、大屏给足，记住上次大小。"""
        self.update_idletasks()
        sc = max(1.0, self.winfo_fpixels("1i") / 96.0)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        avail_w, avail_h = sw - 60, sh - 100
        w = min(int(1040 * sc), avail_w)
        h = min(int(940 * sc), avail_h)
        w = max(w, min(int(860 * sc), avail_w))
        h = max(h, min(int(600 * sc), avail_h))
        geo = str(self.cfg.get("geometry", "") or "")
        if re.fullmatch(r"\d+x\d+", geo or ""):
            gw, gh = (int(v) for v in geo.split("x"))
            w = max(min(gw, avail_w), min(int(720 * sc), avail_w))
            h = max(min(gh, avail_h), min(int(520 * sc), avail_h))
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(min(int(720 * sc), avail_w), min(int(520 * sc), avail_h))

    def _wheel(self, ev):
        cv = self._wheel_canvas(ev.widget)
        if cv is None:
            return
        try:
            cv.yview_scroll(int(-1 * (ev.delta / 120.0)), "units")
        except Exception:
            pass

    def show_maps(self):
        if self.page_maps is None:
            messagebox.showinfo("权重场", "这个包里的权重场页没加载起来。")
            return
        self.nb.select(self.page_maps)
        self.page_maps.load_last()

    # ---------- 取值 / 存值 ----------
    def _load_values(self):
        c = self.cfg
        self.v_photos.set(c["photos"])
        self.v_out.set(c["out"])
        self.v_tag.set(c["tag"])
        self.v_A.set(c["A"])
        self.v_rejk.set(c["rej_k"])
        self.v_nstar.set(c["n_star"])
        self.v_limit.set("0")          # 限帧不记忆：每次启动都从整组开始
        self.v_crop.set("0")            # 裁剪边长同理，默认整幅
        self.v_drz.set(bool(c["drizzle"]))
        self.v_scale.set(c["scale"])
        self.v_pix.set(c["pix"])
        self.v_rb.set(c.get("r_blur", ""))
        self.v_bgdeg.set(str(c.get("bgdeg", "1")))
        self.v_bgr.set(str(c.get("bg_r", "0")))
        self.v_rmode.set(c.get("r_mode", "max"))
        self.v_shw.set(c.get("sharp_w", ""))
        self.v_fwhm.set(c.get("fwhm_max", ""))
        self.v_flux.set(c.get("flux_pow", "0.0"))
        self.v_envw.set(str(c.get("env_win_mult", "2.0")))
        self.v_enve.set(str(c.get("env_eps", "16.0")))
        self.v_envp.set(str(c.get("env_p", "1.0")))
        self.v_rwin.set(str(c.get("r_win", "0")))
        self.v_rk.set(str(c.get("r_k", "3.0")))
        self.v_ccrop.set(c.get("crop_cover", "0.9"))
        self.v_gpu.set({"auto": "自动", "cpu": "只用 CPU",
                        "wgpu": "wgpu（跨平台）", "gpu": "自动"}.get(
            c.get("gpu", "auto"), "自动"))
        self.v_frame.set(c.get("frame", ""))
        self.v_tunea.set(c.get("tune_a", "10,14,20"))
        self.v_1x.set(str(c.get("native_1x", "0")) == "1")
        self.v_cache.set(str(c.get("cache", "1")) not in ("0", "", "False"))
        self.v_rejz.set(str(c.get("rej_z", "1")) not in ("0", "", "False"))
        self.v_maxshare.set(str(c.get("max_share", "0.25")))
        self.v_aligned.set(str(c.get("aligned", "0")) == "1")
        self.v_cfa.set({"off": "关", "auto": "自动识别"}.get(c.get("cfa", "off"),
                                                       c.get("cfa", "off") or "关"))
        self.v_cfad.set({"bilinear": "双线性", "superpixel": "2×2 合并"}.get(
            c.get("cfa_debayer", "bilinear"), "双线性"))
        self.v_jobs.set(str(c.get("jobs", "0")))
        self.v_refk.set(str(c.get("ref_k", 0)))
        self.v_rgbbl.set({"off": "不平（与 WBPP 一致）", "add": "加性平衡",
                         "mul": "乘性平衡"}.get(str(c.get("rgb_balance", "off")), "不平（与 WBPP 一致）"))
        self.v_cfag.set({"shared": "共用画布", "own": "逐通道画布"}.get(
            str(c.get("cfa_grid", "shared")), "共用画布"))
        self.v_cfaa.set({"shared": "共用几何", "own": "逐通道独立"}.get(
            str(c.get("cfa_align", "auto")), "自动判断"))
        self.v_cfaw.set({"auto": "自动", "shared": "共用", "own": "逐通道"}.get(
            str(c.get("cfa_weights", "auto")), "自动"))
        self.v_cfaraw.set({"auto": "自动", "on": "强制开",
                           "off": "强制关（走插值）"}.get(
            c.get("cfa_raw", "auto"), "自动"))
        self.v_rejmode.set({"winsor": "winsorized（推荐）", "mad": "中位绝对偏差（旧）",
                            "fast": "平均绝对偏差（最快）", "esd": "广义 ESD（少帧/拖线）"}.get(
            c.get("rej_mode", "winsor"), "winsorized（推荐）"))
        self.v_kernel.set({"box": "方框（快）", "psfmatch": "PSF 匹配（更干净）"}.get(
            c.get("kernel", "box"), "方框（快）"))
        self.v_psf.set(str(c.get("psf_target", "50")))
        self.v_twopass.set("开" if int(c.get("two_pass", "0") or 0) else "关（推荐）")
        self.v_moon.set(float(c.get("detail_snr", "0") or 0) >= 0.5)
        self.v_xdrz.set(c.get("xdrz", ""))
        self.v_match.set(c.get("match", ""))
        self._pick_saved = (c.get("pick") or "")
        self._sync_drz()

    def save_default(self):
        c = self._collect()
        c["geometry"] = "%dx%d" % (self.winfo_width(), self.winfo_height())
        save_cfg(c)
        self.lab_state.configure(text="已存默认", fg="#34d399")
        messagebox.showinfo("已保存", "当前这套参数已存为默认，下次启动自动带出来。")

    def _collect(self) -> dict:
        return {
            "photos": self.v_photos.get().strip(),
            "out": self.v_out.get().strip(),
            "tag": self.v_tag.get().strip(),
            "A": self.v_A.get().strip(),
            "rej_k": self.v_rejk.get().strip(),
            "n_star": self.v_nstar.get().strip(),
            "limit": self.v_limit.get().strip(),
            "crop": self.v_crop.get().strip(),
            "drizzle": bool(self.v_drz.get()),
            "scale": self.v_scale.get().strip(),
            "pix": self.v_pix.get().strip(),
            "r_blur": self.v_rb.get().strip(),
            "bgdeg": self.v_bgdeg.get().strip(),
            "r_mode": self.v_rmode.get().strip(),
            "sharp_w": self.v_shw.get().strip(),
            "fwhm_max": self.v_fwhm.get().strip(),
            "flux_pow": self.v_flux.get().strip(),
            "crop_cover": self.v_ccrop.get().strip(),
            "gpu": {"自动": "auto", "只用 CPU": "cpu",
                    "wgpu（跨平台）": "wgpu"}.get(self.v_gpu.get().strip(), "auto"),
            "frame": self.v_frame.get().strip(),
            "tune_a": self.v_tunea.get().strip() or "10,14,20",
            "native_1x": "1" if bool(self.v_1x.get()) else "0",
            "cache": "1" if bool(self.v_cache.get()) else "0",
            "rej_z": "1" if bool(self.v_rejz.get()) else "0",
            "max_share": self.v_maxshare.get().strip() or "0.25",
            "aligned": "1" if bool(self.v_aligned.get()) else "0",
            "cfa_ref": self.v_cfa_ref.get().strip(),
            "cfa_raw": {"自动": "auto", "强制开": "on",
                        "强制关（走插值）": "off",
                        "混合（G 真 CFA + R/B 插值）": "hybrid"}.get(
                            self.v_cfaraw.get(), "auto"),
            "ref_k": int(self.v_refk.get().strip() or 0),
            "cfa_align": {"自动判断": "auto", "共用几何": "shared",
                          "逐通道独立": "own"}.get(
                self.v_cfaa.get(), "auto"),
            "rgb_balance": {"不平（与 WBPP 一致）": "off", "加性平衡": "add",
                            "乘性平衡": "mul"}.get(
                self.v_rgbbl.get(), "off"),
            "cfa_grid": {"共用画布": "shared", "逐通道画布": "own"}.get(
                self.v_cfag.get().strip(), "shared"),
            "cfa": {"关": "off", "自动识别": "auto"}.get(self.v_cfa.get().strip(),
                                                        self.v_cfa.get().strip() or "off"),
            "cfa_debayer": {"双线性": "bilinear", "2×2 合并": "superpixel"}.get(
                self.v_cfad.get().strip(), "bilinear"),
            "jobs": self.v_jobs.get().strip(),
            "cfa_weights": {"自动": "auto", "共用": "shared", "逐通道": "own"}.get(
                self.v_cfaw.get().strip(), "auto"),
            "xdrz": self.v_xdrz.get().strip(),
            "match": self.v_match.get().strip(),
            "pick": self._pick_value(),
            "rej_mode": {"winsorized（推荐）": "winsor", "中位绝对偏差（旧）": "mad",
                         "平均绝对偏差（最快）": "fast",
                         "广义 ESD（少帧/拖线）": "esd"}.get(
                             self.v_rejmode.get().strip(), "winsor"),
            "kernel": {"方框（快）": "box", "PSF 匹配（更干净）": "psfmatch"}.get(
                self.v_kernel.get().strip(), "box"),
            "psf_target": self.v_psf.get().strip() or "50",
            "env_win_mult": self.v_envw.get().strip() or "2.0",
            "env_eps": self.v_enve.get().strip() or "16.0",
            "env_p": self.v_envp.get().strip() or "1.0",
            "r_win": self.v_rwin.get().strip() or "0",
            "r_k": self.v_rk.get().strip() or "3.0",
            "two_pass": "1" if self.v_twopass.get().strip() == "开" else "0",
            "detail_snr": "1" if bool(self.v_moon.get()) else "0",
        }

    def _sync_drz(self):
        st = "normal" if self.v_drz.get() else "disabled"
        self.sp_scale.configure(state=st)
        self.sp_pix.configure(state=st)

    # ---------- 按钮 ----------
    def _pick_photos(self):
        d = filedialog.askdirectory(title="选择素材目录（已校准的 XISF / FITS，未对齐最好）",
                                    initialdir=self.v_photos.get() or str(HERE))
        if d:
            self.v_photos.set(d)
            self._autofill_related()

    def _pick_out(self):
        d = filedialog.askdirectory(title="选择输出目录",
                                    initialdir=self.v_out.get() or str(HERE))
        if d:
            self.v_out.set(d)

    def _scan_channels(self):
        """读帧头列出通道（不含像素），勾选后 --pick 只叠所选"""
        photos = self.v_photos.get().strip()
        if not photos or not Path(photos).exists():
            messagebox.showerror("素材不存在", "请先在「素材目录」里选一个存在的目录。")
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(HERE / "pydeps"), str(HERE), str(HERE / "core")])
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            r = subprocess.run([str(PYEXE), str(MAINPY), "--cli", "--scan",
                                "--photos", photos],
                               cwd=str(HERE), env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=900)
        except Exception as e:
            messagebox.showerror("扫描失败", str(e))
            return
        info = None
        for line in reversed((r.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    info = json.loads(line)
                    break
                except Exception:
                    continue
        if not info:
            messagebox.showerror("扫描失败",
                                 ((r.stdout or "") + (r.stderr or ""))[-1200:])
            return
        self._ch_total = int(info.get("total", 0) or 0)
        self.ch_names = []
        self.ch_vars = []
        for w in self.fr_ch.winfo_children():
            w.destroy()
        _want = set(x for x in (getattr(self, "_pick_saved", "") or "").split(";") if x)
        for c in info.get("channels", []):
            self.ch_names.append(c["name"])
            v = tk.BooleanVar(value=(not _want) or (c["name"] in _want))
            self.ch_vars.append(v)
            _lab = str(c["label"])
            if _lab.isdigit() or not _lab.strip():
                _lab = "通道 %s" % _lab if _lab.isdigit() else "未标滤镜"
            tk.Checkbutton(self.fr_ch, text="%s   ·   %d 帧" % (_lab, c["n"]),
                           variable=v, command=self._sync_check_glyph,
                           bg=CARD, activebackground=CARD, selectcolor="#ffffff",
                           anchor="w", relief="flat", bd=0, highlightthickness=0,
                           font=FONT, fg=FG).pack(anchor="w", padx=_px(4), pady=_px(1))
        self._sync_check_glyph()
        self.fr_ch.update_idletasks()
        self.cv_ch.configure(scrollregion=self.cv_ch.bbox("all"))
        self._sync_check_glyph()

    def _check_all(self, flag: bool):
        for v in getattr(self, "ch_vars", []):
            v.set(bool(flag))
        self._sync_check_glyph()

    def _sync_check_glyph(self):
        """☑ / ☐ 跟着状态走，一眼看出勾了哪些"""
        if not getattr(self, "ch_vars", None):
            return
        for w, v in zip(self.fr_ch.winfo_children(), self.ch_vars):
            try:
                txt = w.cget("text")
            except Exception:
                continue
            if txt[:1] in ("☑", "☐"):
                w.configure(text=("☑" if v.get() else "☐") + txt[1:])
        n_on = sum(1 for v in self.ch_vars if v.get())
        tot = getattr(self, "_ch_total", 0)
        head = ("共 %d 帧 → %d 个通道" % (tot, len(self.ch_vars))) if tot else                ("%d 个通道" % len(self.ch_vars))
        tail = ("，全选 = 不加筛选" if n_on == len(self.ch_vars) else
                ("，一个都没选也按全部处理" if n_on == 0 else "，只叠勾选的"))
        self.lab_ch.configure(text=head + "，勾选 %d 个%s" % (n_on, tail))
        self._pick_saved = self._pick_value()

    def _pick_value(self) -> str:
        if not getattr(self, "ch_names", None):
            return getattr(self, "_pick_saved", "") or ""
        sel = [n for n, v in zip(self.ch_names, self.ch_vars) if v.get()]
        if not sel or len(sel) == len(self.ch_names):
            return ""
        return ";".join(x if x else "无滤镜信息" for x in sel)

    def _pick_cfa_ref(self):
        d = filedialog.askdirectory(title="选 WBPP 的 debayered 目录")
        if d:
            self.v_cfa_ref.set(d)

    def _pick_frame(self):
        p = filedialog.askopenfilename(
            title="选一张参考图（master）", filetypes=[("XISF", "*.xisf"), ("全部", "*.*")])
        if p:
            self.v_frame.set(p)

    def _pick_match(self):
        f = filedialog.askopenfilename(
            title="选择参考图（master .xisf）",
            initialdir=str(Path(self.v_match.get()).parent) if self.v_match.get() else str(HERE),
            filetypes=[("XISF", "*.xisf"), ("所有文件", "*.*")])
        if f:
            self.v_match.set(f)

    def _pick_xdrz(self):
        d = filedialog.askdirectory(title="选择 .xdrz 所在目录（WBPP registered\\ 下）",
                                    initialdir=self.v_xdrz.get() or str(HERE))
        if d:
            self.v_xdrz.set(d)


    def _bind_wheel_all(self):
        """整页滚轮：滚轮在页面上任意位置都滚动当前页，数值框**不改变数值**。"""
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self.bind_all(seq, self._on_wheel, add="+")
            except Exception:
                pass

    def _wheel_canvas(self, w):
        """从当前控件往上找最近的可竖直滚动 Canvas。"""
        try:
            cur = w
            for _ in range(24):
                if isinstance(cur, tk.Canvas) and cur.cget("yscrollcommand"):
                    return cur
                cur = cur.master
                if cur is None:
                    break
        except Exception:
            pass
        return getattr(self, "cv_body", None)

    def _on_wheel(self, ev):
        # 日志框、通道列表这类自己有滚动条的控件，让它们自己滚
        if isinstance(ev.widget, (tk.Text, ttk.Treeview)):
            return None
        up = (getattr(ev, "delta", 0) or 0) > 0 or getattr(ev, "num", 0) == 4
        cv = self._wheel_canvas(ev.widget)
        if cv is None:
            return None
        try:
            cv.yview_scroll(-3 if up else 3, "units")
        except Exception:
            return None
        return "break"

    def _schedule_autofill(self, *_a):
        try:
            if getattr(self, "_af_job", None):
                self.after_cancel(self._af_job)
        except Exception:
            pass

        def _run():
            self._af_job = None
            self._autofill_related()

        try:
            self._af_job = self.after(500, _run)
        except Exception:
            self._af_job = None

    def _autofill_related(self, _evt=None):
        """选了亮场主路径后，自动找同级的 registered（几何目录）与 debayered（彩机参考）。

        每次改主路径都会重新找一遍；只要对应输入框还是「空」或「上次自动填的值」，
        就跟着刷新（手工改过的不会被覆盖）。
        """
        src = self.v_photos.get().strip()
        if not src:
            return
        p = Path(src)
        roots = []
        if p.name.lower().startswith(("calibrat", "light")):
            roots.append(p.parent)
        roots.append(p.parent)
        roots.append(p)
        for var, key, names, hint in (
            (self.v_xdrz, "xdrz", ("registered",),
             "几何目录"),
            (self.v_cfa_ref, "cfa_ref", ("debayered", "Debayered"),
             "彩机参考目录"),
        ):
            found = None
            for root in roots:
                for nm in names:
                    cands = [root / nm, root / (nm.lower())]
                    for c in cands:
                        try:
                            if c.is_dir() and any(c.iterdir()):
                                found = c
                                break
                        except Exception:
                            continue
                    if found is not None:
                        break
                if found is not None:
                    break
            cur = var.get().strip()
            prev = self._auto_paths.get(key, "")
            # 也算「自动值」的情况：路径已不存在，或者它本身就叫 registered/debayered
            # （重启后 prev 是空的，若不这样判断，换了素材目录也不会刷新）
            _cur_looks_auto = False
            if cur:
                try:
                    _cur_looks_auto = (not Path(cur).is_dir()) or                         (Path(cur).name.lower() in ("registered", "debayered"))
                except Exception:
                    _cur_looks_auto = False
            if found is not None:
                if cur == "" or cur == prev or _cur_looks_auto:
                    var.set(str(found))
                    self._auto_paths[key] = str(found)
                    self._log_line("[界面] 已自动识别并填入%s：%s" % (hint, found))
            elif cur and cur == prev:
                var.set("")
                self._auto_paths.pop(key, None)
                self._log_line("[界面] 没找到%s，已清空（原来自动填的）：%s" % (hint, cur))

    def _recompose_rgb(self):
        """选 R/G/B 三个成品重新合成 RGB（不做白平衡；可选按天空对齐）。"""
        out_dir = self.v_out.get().strip() or str(HERE)
        picks = []
        for ch in ("R", "G", "B"):
            f = filedialog.askopenfilename(
                title="选择 %s 通道成品（PWS_*_%s.xisf）" % (ch, ch),
                initialdir=out_dir,
                filetypes=[("XISF", "*.xisf"), ("所有文件", "*.*")])
            if not f:
                return
            picks.append(f)
        tag = self.v_tag.get().strip() or "RGB"
        default = str(Path(out_dir) / ("PWS_%s_RGB.xisf" % tag))
        out = filedialog.asksaveasfilename(
            title="RGB 保存为", initialdir=out_dir, initialfile=Path(default).name,
            defaultextension=".xisf", filetypes=[("XISF", "*.xisf")])
        if not out:
            return
        sky = messagebox.askyesno(
            "白平衡", "要把 R/B 的天空亮度对齐到 G 吗？\n\n"
                     "选「否」= 完全保持原始比例（默认，不做白平衡）")
        script = HERE / "recompose_rgb.py"
        if not script.exists():
            messagebox.showerror("缺少脚本", "包里找不到 recompose_rgb.py")
            return
        argv = [sys.executable, str(script),
                "--r", picks[0], "--g", picks[1], "--b", picks[2], "--out", out]
        if sky:
            argv.append("--sky-match")
        argv += ["--preview", str(Path(out).with_suffix(".png"))]
        self._log_line("[合成] R=%s" % picks[0])
        self._log_line("[合成] G=%s" % picks[1])
        self._log_line("[合成] B=%s" % picks[2])
        self._log_line("[合成] 输出 %s（白平衡：%s）" % (out, "按天空对齐" if sky else "不做"))

        def _run():
            try:
                env = dict(os.environ)
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(HERE / "pydeps"), str(HERE), str(HERE / "core"),
                     str(HERE / "gpudeps")])
                pr = subprocess.Popen(argv, cwd=str(HERE), env=env,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
                for raw in pr.stdout:
                    try:
                        line = raw.decode("gbk")
                    except Exception:
                        line = raw.decode("utf-8", "replace")
                    line = line.rstrip()
                    if line:
                        self.after(0, self._log_line, "[合成] " + line)
                code = pr.wait()
                self.after(0, self._log_line,
                           "[合成] 完成（退出码 %d）：%s" % (code, out))
            except Exception as e:
                self.after(0, self._log_line, "[合成] 失败：%s: %s" % (type(e).__name__, e))

        threading.Thread(target=_run, daemon=True).start()

    def _open_out(self):
        p = self.v_out.get().strip()
        if p and Path(p).exists():
            _open_path(p)
        else:
            messagebox.showinfo("提示", "输出目录还不存在。")

    # ---------- 启动时的许可提示 ----------
    def show_license(self):
        """开机弹窗：PWS 方法来源与版权说明。点「我知道了」才能继续。"""
        if os.environ.get("DWT_NO_STARTUP_DIALOG") == "1":
            return
        win = tk.Toplevel(self)
        win.title("关于本软件的方法来源")
        win.configure(bg=CARD)
        win.transient(self)
        win.resizable(False, False)
        try:
            win.grab_set()
        except Exception:
            pass

        head = tk.Frame(win, bg=CARD)
        head.pack(fill="x", padx=22, pady=(18, 6))
        tk.Label(head, text="DWT · Drizzle", bg=CARD, fg=ACCENT,
                 font=(_FONT_FAMILY, 15, "bold")).pack(anchor="w")
        tk.Label(head, text="使用前请先阅读下面的方法来源与版权说明",
                 bg=CARD, fg="#64748b", font=(_FONT_FAMILY, 10)).pack(anchor="w", pady=(4, 0))

        body = tk.Text(win, wrap="word", font=(_FONT_FAMILY, 10), bg="#f8fafc", fg=FG,
                       relief="flat", padx=14, pady=12, width=64, height=16,
                       highlightthickness=1, highlightbackground=BORDER)
        body.pack(fill="both", expand=True, padx=22, pady=(6, 12))
        body.insert("1.0", LICENSE_TEXT)
        body.configure(state="disabled")

        bar = tk.Frame(win, bg=CARD)
        bar.pack(fill="x", padx=22, pady=(0, 18))
        ttk.Button(bar, text="打开原文链接",
                   command=lambda: _open_path("https://github.com/DDCikey/"
                                              "Partition-Weighted-Stacking/tree/main/docs")
                   ).pack(side="left")
        ok = ttk.Button(bar, text="我知道了", style="Run.TButton",
                        command=lambda: self._close_license(win))
        ok.pack(side="right")
        win.bind("<Return>", lambda e: self._close_license(win))
        win.bind("<Escape>", lambda e: self._close_license(win))

        win.update_idletasks()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        sw = self.winfo_screenwidth() or 1600     # 未映射的 Toplevel 可能返回 0，用主窗的问
        sh = self.winfo_screenheight() or 900
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        win.geometry("%dx%d+%d+%d" % (w, h, x, y))
        # 再补一次：有些 WM 会在映射时把 transient 挪回 (0,0)
        win.after(60, lambda: win.winfo_exists() and win.geometry("+%d+%d" % (x, y)))
        win.attributes("-topmost", True)
        win.after(600, lambda: win.winfo_exists() and win.attributes("-topmost", False))
        win.lift()
        win.focus_force()
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_license(win))
        ok.focus_set()
        self._license_win = win

    def _close_license(self, win):
        try:
            win.grab_release()
        except Exception:
            pass
        self._license_win = None
        win.destroy()

    def show_about(self):
        """关于：方法与原文出处（PWS，D.Cikey）"""
        messagebox.showinfo(
            "关于 DWT · Drizzle",
            "DWT · Drizzle 2.7.118-D\n\n" + LICENSE_TEXT)

    def show_help(self):
        """打开内置使用说明（不依赖浏览器 / 外部程序）"""
        if _hp is None:
            messagebox.showinfo("使用说明", "找不到使用说明文件")
            return None
        try:
            return _hp.open_help(self)
        except Exception as e:
            messagebox.showinfo("使用说明", "打不开内置说明：%s" % e)
            return None

    def show_monitor(self):
        """打开/唤起运行监控窗（叠加开始时会自动打开）"""
        if Monitor is None:
            messagebox.showinfo("运行监控", "监控模块不可用")
            return
        fresh = self.mon is None or not self.mon.winfo_exists()
        if fresh:
            self.mon = Monitor(self)
            try:                       # 窗口曾被杀掉时，把已有日志回放进去
                for ln in list(getattr(self, '_mon_buf', [])):
                    self.mon.feed(ln)
            except Exception:
                pass
        self.mon.deiconify()
        self.mon.lift()

    def start_tune(self):
        """叠加前自动调参：抽若干帧试几组 A / r_mode，给出建议。"""
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("正在运行", "已有任务在跑，等它结束或先停止。")
            return
        c = self._collect()
        if not c["photos"] or not Path(c["photos"]).exists():
            messagebox.showerror("素材不存在", f"素材目录不存在：{c['photos']}")
            return
        if not c["out"]:
            messagebox.showerror("缺少输出目录", "请先选择输出目录。")
            return
        n = simpledialog.askinteger("自动调参", "抽几帧试参数？（建议 16，越多越准也越慢）",
                                    initialvalue=16, minvalue=6, maxvalue=64, parent=self)
        if not n:
            return
        save_cfg(c)
        try:
            self.show_monitor()
            self.mon.reset()
            self._mon_buf = []
        except Exception:
            pass
        argv = _argv_common(c) + ["--tune", str(int(n))]
        if c.get("tune_a"):
            argv += ["--tune-a", c["tune_a"]]
        self._tune_mode = True
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(HERE / "pydeps"), str(HERE)])
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self.txt.delete("1.0", "end")
        self._log("命令：" + " ".join(argv) + chr(10), "dim")
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(HERE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            return
        self.t0 = time.time()
        self.pb.configure(mode="indeterminate")
        self.pb.start(12)
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lab_state.configure(text="调参中", fg="#fbbf24")
        self.reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self.reader.start()

    def _tune_best(self, line: str):
        """收到 TUNE-BEST 就把建议填回界面。"""
        try:
            import json as _json
            d = _json.loads(line.split('TUNE-BEST', 1)[1].strip())
        except Exception:
            return
        try:
            self.v_A.set('%g' % d['A'])
            self.v_rmode.set(d['r_mode'])
        except Exception:
            return
        msg = ('建议：清晰度 A = %g，R 场定义 = %s') % (d['A'], d['r_mode'])
        msg += '\n' + ('成品 FWHM %.2fpx，改善 %+.1f%%，N_eff %.1f' % (d['fwhm'], d['gain'], d.get('neff', 0)))
        msg += '\n\n已填回界面，直接点「开始叠加」即可。'
        messagebox.showinfo('自动调参结论', msg)

    def _cfa_autotune_a(self):
        """不再自动改 A（用户要求「填多少就是多少」），只在选彩机时提示建议值。"""
        try:
            if self.v_cfa.get().strip() in ("", "关"):
                return
            self._log("[界面] 提示：彩机建议清晰度 A=6（等效帧数更高、噪声更低）；"
                      "黑白建议 A=20。当前值不会被自动修改。")
        except Exception:
            pass

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        c = self._collect()
        if not c["photos"] or not Path(c["photos"]).exists():
            messagebox.showerror("素材不存在", f"素材目录不存在：\n{c['photos']}")
            return
        if not c["out"]:
            messagebox.showerror("缺少输出目录", "请先选择输出目录。")
            return
        save_cfg(c)
        try:
            self.show_monitor()
            self.mon.reset()
        except Exception:
            pass

        argv = _argv_common(c)


        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(HERE / "pydeps"), str(HERE)])
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        self.txt.delete("1.0", "end")
        self._open_run_log(c, argv)
        self._log("命令：" + " ".join(argv) + "\n", "dim")
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(HERE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            return

        self.t0 = time.time()
        self.pb.configure(mode="indeterminate")
        self.pb.start(12)
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lab_state.configure(text="运行中", fg=CYAN)
        self.reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self.reader.start()

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            return
        self._log("\n[界面] 正在停止…\n", "warn")
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _read(self, proc):
        try:
            for line in proc.stdout:      # type: ignore[union-attr]
                self.q.put(line.rstrip("\n"))
        except Exception as e:
            self.q.put(f"[界面] 读取日志出错：{e}")
        proc.wait()
        self.q.put(("__done__", proc.returncode))

    # ---------- 事件循环 ----------
    def _pump(self):
        try:
            for _ in range(200):
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    self._finish(item[1])
                else:
                    self._log_line(item)
        except queue.Empty:
            pass
        self.after(80, self._pump)

    def _log_line(self, line: str):
        if 'TUNE-BEST' in line:
            self._tune_best(line)
        tag = ""
        if ("Traceback" in line or "Error" in line or "错误" in line
                or "失败" in line and "匹配失败" not in line):
            tag = "err"
        elif "WARNING" in line or "注意" in line:
            tag = "warn"
        elif line.startswith("[时]") or "完成" in line:
            tag = "ok"
        self._log(line + "\n", tag)
        try:
            self._mon_buf.append(line)
            if len(self._mon_buf) > 4000:
                del self._mon_buf[:1000]
        except Exception:
            pass
        try:
            if self.mon is not None and self.mon.winfo_exists():
                self.mon.feed(line)
        except Exception:
            pass
        m = STAGE_RE.search(line)
        if m:
            stage = {"S": "信噪比权重", "σ": "噪声", "C": "测星点", "R 场": "R 场",
                     "叠加": "叠加", "输出": "输出", "位移": "配准", "星表": "星表",
                     "drizzle": "配准"}.get(m.group(1), m.group(1))
            self.lab_stage.configure(text=f"阶段：{stage}")

    def _open_run_log(self, c, argv):
        """把本次叠加日志写到输出目录（与成品同目录）。"""
        self._close_run_log()
        try:
            out_s = str(c.get("out") or "").strip()
            if not out_s:
                return
            out = Path(out_s).expanduser()
            out.mkdir(parents=True, exist_ok=True)
            tag = (str(c.get("tag") or "stack").strip() or "stack")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._log_path = out / ("%s_叠加日志_%s.txt" % (tag, stamp))
            self._log_fh = open(self._log_path, "w", encoding="utf-8")
            self._log_fh.write("DWT-Drizzle 叠加日志" + chr(10))
            self._log_fh.write("开始时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S") + chr(10))
            self._log_fh.write("命令：%s" % " ".join(argv) + chr(10))
            self._log_fh.write("=" * 60 + chr(10))
            self._log_fh.flush()
            self._log("[界面] 日志同时保存到 %s" % self._log_path + chr(10), "dim")
        except Exception as e:
            self._log_fh = None
            self._log_path = None
            self._log("[界面] 日志文件创建失败（不影响叠加）：%s" % e + chr(10), "warn")

    def _close_run_log(self, code=None):
        try:
            if self._log_fh:
                self._log_fh.write("=" * 60 + chr(10))
                if code is not None:
                    self._log_fh.write("结束时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S") + chr(10))
                    self._log_fh.write("退出码：%s" % code + chr(10))
                self._log_fh.flush()
                self._log_fh.close()
        except Exception:
            pass
        self._log_fh = None

    def _log(self, s: str, tag: str = ""):
        try:
            if self._log_fh is not None:
                self._log_fh.write(s)
                self._log_fh.flush()
        except Exception:
            pass
        at_bottom = self.txt.yview()[1] > 0.98
        self.txt.insert("end", s, tag)
        if at_bottom:
            self.txt.see("end")

    def _finish(self, code: int):
        self._close_run_log(code if self._log_path else None)
        try:
            if self.mon is not None and self.mon.winfo_exists():
                self.mon.finish(code)
        except Exception:
            pass
        self.pb.stop()
        self.pb.configure(mode="determinate", value=0)
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if code == 0:
            self.lab_state.configure(text="完成", fg="#34d399")
            self._log("\n[界面] 完成。产物在：" + self.v_out.get() + "\n", "ok")
        else:
            self.lab_state.configure(text=f"失败（{code}）", fg="#f87171")
            self._log(f"\n[界面] 进程退出码 {code}。\n", "err")
        self.lab_stage.configure(text="阶段：—")
        if self.smoke:
            print("SMOKE-OK" if code == 0 else f"SMOKE-FAIL code={code}", flush=True)
            self.after(300, self.destroy)

    def _tick(self):
        if self.proc and self.proc.poll() is None and self.t0:
            s = int(time.time() - self.t0)
            self.lab_time.configure(text=f"用时 {s // 60:02d}:{s % 60:02d}")
        self.after(500, self._tick)

    def _selftest_done(self):
        self.destroy()
        print("SELFTEST-OK")

    def destroy(self):
        save_cfg(self._collect())
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               capture_output=True)
            except Exception:
                pass
        super().destroy()


def _dpi_aware():
    """让窗口按物理像素绘制，避免高 DPI 下字体放大而布局被截断。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def main():
    _dpi_aware()
    selftest = "--selftest" in sys.argv
    smoke = "--smoke" in sys.argv
    try:
        app = App(selftest=selftest, smoke=smoke)
    except Exception:
        import traceback
        msg = traceback.format_exc()
        try:
            Path(CFG.parent).mkdir(parents=True, exist_ok=True)
            (CFG.parent / "crash.log").write_text(msg, "utf-8")
        except Exception:
            pass
        try:
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror("DWT 界面启动失败", msg)
            r.destroy()
        except Exception:
            pass
        raise
    if smoke:
        app.withdraw()
        app.v_limit.set("2")
        app.v_crop.set("400")
        app.v_drz.set(True)
        app.v_scale.set("2")
        app.after(400, app.start)
        app.after(300000, lambda: (print("SMOKE-TIMEOUT", flush=True), app.destroy()))
    app.mainloop()


if __name__ == "__main__":
    main()
