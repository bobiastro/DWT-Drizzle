use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dwt_backend::{self, EvalReport, JobProgress, JobResult, NpyMap, StatBlock};
use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, FontFamily, FontId, Label, Layout, Margin,
    RichText, Sense, Stroke, TextureHandle, TextureOptions, Ui, Vec2,
};

use crate::config::Params;
use crate::license;
use crate::theme;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Common,
    Advanced,
    Eval,
    Maps,
}

impl Tab {
    fn label(self) -> &'static str {
        match self {
            Tab::Common => "  常用  ",
            Tab::Advanced => "  高级  ",
            Tab::Eval => "  叠加评价  ",
            Tab::Maps => "  权重场  ",
        }
    }
}

#[derive(Clone, Copy)]
enum LogKind {
    Normal,
    Dim,
    Ok,
    Warn,
    Err,
}

struct LogLine {
    text: String,
    kind: LogKind,
}

impl LogLine {
    fn color(&self) -> Color32 {
        match self.kind {
            LogKind::Normal => theme::LOG_FG,
            LogKind::Dim => Color32::from_rgb(0x64, 0x74, 0x8b),
            LogKind::Ok => Color32::from_rgb(0x86, 0xef, 0xac),
            LogKind::Warn => Color32::from_rgb(0xfc, 0xd3, 0x4d),
            LogKind::Err => Color32::from_rgb(0xfc, 0xa5, 0xa5),
        }
    }
}

enum JobEvent {
    Log(String),
    Progress(JobProgress),
    Done(Result<JobResult, String>),
}

// ───────────────────────── 自证模式（--dump-pages） ─────────────────────────

/// 固定文件名，顺序与 [`DumpPages::tab`] 对应。
const DUMP_NAMES: [&str; 5] = [
    "01_common.png",
    "02_advanced.png",
    "03_eval.png",
    "04_maps.png",
    "05_about.png",
];
/// 每页先渲染这么多帧（第一帧布局、后两帧稳定），再请求截图。
const DUMP_SETTLE_FRAMES: u32 = 3;
/// 兜底：超过这个时间还没截完就直接失败退出，避免挂死。
const DUMP_DEADLINE: Duration = Duration::from_secs(60);

static DUMP_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// 解析 `--dump-pages <DIR>` / `--dump-pages=<DIR>`；不认识的其他参数原样忽略。
pub fn parse_dump_pages(args: &[String]) -> Result<Option<PathBuf>, String> {
    let mut dir: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--dump-pages" {
            match args.get(i + 1) {
                Some(v) => dir = Some(PathBuf::from(v)),
                None => return Err("--dump-pages 后面缺少目录".into()),
            }
            i += 2;
        } else if let Some(v) = a.strip_prefix("--dump-pages=") {
            if v.is_empty() {
                return Err("--dump-pages 后面缺少目录".into());
            }
            dir = Some(PathBuf::from(v));
            i += 1;
        } else {
            i += 1;
        }
    }
    Ok(dir)
}

/// 由 `main` 在 `run_native` 之前调用，把目标目录交给 [`DrizzleApp::new`]。
pub fn set_dump_dir(dir: PathBuf) {
    let _ = DUMP_DIR.set(dir);
}

struct DumpPages {
    dir: PathBuf,
    index: usize,
    frames_on_page: u32,
    requested: bool,
    written: Vec<(String, [usize; 2], u64)>,
    started: Instant,
}

impl DumpPages {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            index: 0,
            frames_on_page: 0,
            requested: false,
            written: Vec::new(),
            started: Instant::now(),
        }
    }

    /// 第 `index` 张图对应的页面；「关于」叠在权重场那一页上。
    fn tab(&self) -> Tab {
        match self.index {
            0 => Tab::Common,
            1 => Tab::Advanced,
            2 => Tab::Eval,
            _ => Tab::Maps,
        }
    }
}

pub struct DrizzleApp {
    params: Params,
    tab: Tab,
    license_open: bool,
    about_open: bool,
    help_open: bool,
    monitor_open: bool,
    log: Vec<LogLine>,
    log_file: Option<File>,
    log_path: Option<PathBuf>,
    state: String,
    run_started: Option<Instant>,
    elapsed: Duration,
    progress: Option<JobProgress>,
    rx: Option<Receiver<JobEvent>>,
    cancel: Option<Arc<AtomicBool>>,
    channels: Vec<(String, usize)>,
    selected: BTreeSet<String>,
    last_out: String,
    last_stem: String,
    eval_path: String,
    eval_report: Option<EvalReport>,
    eval_msg: String,
    map_path: String,
    map_r: Option<NpyMap>,
    map_neff: Option<NpyMap>,
    tex_r: Option<TextureHandle>,
    tex_neff: Option<TextureHandle>,
    map_log: bool,
    map_auto_loaded: bool,
    map_msg: String,
    auto_start: bool,
    started_once: bool,
    stop_after: Option<Duration>,
    stop_requested: bool,
    exit_after_job: bool,
    exit_at: Option<Instant>,
    frames: u32,
    /// `Some(..)` = 自证模式：不靠人点，自己切页截图然后退出。
    dump: Option<DumpPages>,
}

impl DrizzleApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&cc.egui_ctx);
        let (params, load_msg) = Params::load();
        let last_out = params.out.clone();
        let last_stem = params.last_stem();
        let initial_product = if last_out.trim().is_empty() {
            String::new()
        } else {
            PathBuf::from(&last_out)
                .join(format!("{last_stem}.xisf"))
                .display()
                .to_string()
        };
        let mut app = Self {
            params,
            tab: match std::env::var("DWT_GUI_TAB").unwrap_or_default().to_ascii_lowercase().as_str() {
                "advanced" | "高级" => Tab::Advanced,
                "eval" | "评价" => Tab::Eval,
                "maps" | "权重场" => Tab::Maps,
                _ => Tab::Common,
            },
            license_open: if DUMP_DIR.get().is_some() {
                // 自证模式不要开机说明窗挡住页面
                false
            } else {
                std::env::var("DWT_NO_STARTUP_DIALOG").map(|v| v != "1").unwrap_or(true)
            },
            about_open: false,
            help_open: false,
            monitor_open: false,
            log: Vec::new(),
            log_file: None,
            log_path: None,
            state: "空闲".into(),
            run_started: None,
            elapsed: Duration::ZERO,
            progress: None,
            rx: None,
            cancel: None,
            channels: Vec::new(),
            selected: BTreeSet::new(),
            last_out,
            last_stem,
            eval_path: initial_product.clone(),
            eval_report: None,
            eval_msg: String::new(),
            map_path: initial_product,
            map_r: None,
            map_neff: None,
            tex_r: None,
            tex_neff: None,
            map_log: false,
            map_auto_loaded: false,
            map_msg: String::new(),
            auto_start: std::env::var("DWT_GUI_AUTOSTART").map(|v| v == "1").unwrap_or(false),
            started_once: false,
            stop_after: std::env::var("DWT_GUI_STOP_AFTER_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(Duration::from_millis),
            stop_requested: false,
            exit_after_job: std::env::var("DWT_GUI_EXIT_AFTER_JOB").map(|v| v == "1").unwrap_or(false),
            exit_at: None,
            frames: 0,
            dump: DUMP_DIR.get().cloned().map(DumpPages::new),
        };
        app.log.push(LogLine { text: format!("DWT · Drizzle {}  |  原生 Rust GUI", license::VERSION), kind: LogKind::Ok });
        app.log.push(LogLine { text: "分区加权叠加 · Drizzle 几何重采样".into(), kind: LogKind::Dim });
        if let Some(m) = load_msg {
            app.log.push(LogLine { text: m, kind: LogKind::Dim });
        }
        app.log.push(LogLine {
            text: format!("设置文件：{}", Params::config_path().display()),
            kind: LogKind::Dim,
        });
        app
    }

    fn push(&mut self, text: impl Into<String>, kind: LogKind) {
        let text = text.into();
        if let Some(f) = self.log_file.as_mut() {
            let _ = writeln!(f, "{text}");
            let _ = f.flush();
        }
        self.log.push(LogLine { text, kind });
        if self.log.len() > 5000 {
            self.log.drain(0..1000);
        }
    }

    fn push_auto(&mut self, text: impl Into<String>) {
        let s: String = text.into();
        let kind = if s.contains("[err]") || s.contains("错误") || s.contains("失败") {
            LogKind::Err
        } else if s.contains("警告") || s.contains("warn") || s.contains("跳过") {
            LogKind::Warn
        } else if s.contains("[输出]") || s.contains("完成") {
            LogKind::Ok
        } else if s.starts_with("[界面]") || s.starts_with("[命令]") {
            LogKind::Dim
        } else {
            LogKind::Normal
        };
        self.push(s, kind);
    }

    fn save_settings(&mut self, quiet: bool) {
        match self.params.save() {
            Ok(p) => {
                if !quiet {
                    self.push(format!("[界面] 设置已保存：{}", p.display()), LogKind::Dim);
                }
            }
            Err(e) => self.push(format!("[err] 保存设置失败：{e}"), LogKind::Err),
        }
    }

    fn open_log_file(&mut self) {
        let out = PathBuf::from(self.params.out.trim());
        if out.as_os_str().is_empty() {
            return;
        }
        if let Err(e) = std::fs::create_dir_all(&out) {
            self.push(format!("[err] 创建输出目录失败：{e}"), LogKind::Err);
            return;
        }
        let tag = if self.params.tag.trim().is_empty() { "PWS" } else { self.params.tag.trim() };
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let path = out.join(format!("{tag}_叠加日志_{stamp}.txt"));
        match File::create(&path) {
            Ok(mut f) => {
                let _ = writeln!(f, "DWT-Drizzle 叠加日志");
                let _ = writeln!(f, "开始时间：{}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
                let _ = writeln!(f, "{}", "=".repeat(60));
                let _ = f.flush();
                self.log_file = Some(f);
                self.log_path = Some(path);
            }
            Err(e) => self.push(format!("[err] 创建日志文件失败：{e}"), LogKind::Err),
        }
    }

    fn close_log_file(&mut self) {
        if let Some(mut f) = self.log_file.take() {
            let _ = writeln!(f, "结束时间：{}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
            let _ = f.flush();
        }
    }

    fn stop_run(&mut self) {
        if self.stop_requested {
            return;
        }
        if let Some(c) = &self.cancel {
            c.store(true, Ordering::Relaxed);
            self.stop_requested = true;
            self.push("[界面] 正在停止…", LogKind::Warn);
        }
    }

    fn start_run(&mut self, ctx: &egui::Context) {
        if self.state == "运行中" {
            return;
        }
        self.save_settings(true);
        let mut cfg = match self.params.job_config() {
            Ok(c) => c,
            Err(e) => {
                self.push(format!("[err] 参数错误：{e}"), LogKind::Err);
                self.state = "失败".into();
                return;
            }
        };
        // A run targets one filter group.  The GUI checklist's first selected
        // channel wins; this is also what the Python --pick argument does.
        if let Some(first) = self.selected.iter().next() {
            cfg.pick = first.clone();
            self.params.pick = first.clone();
        }
        if self.selected.len() > 1 {
            self.push(
                format!(
                    "[警告] 已勾选 {} 个通道，本次先叠 {}；其余通道可分别运行。",
                    self.selected.len(),
                    cfg.pick
                ),
                LogKind::Warn,
            );
        }
        self.log.clear();
        self.open_log_file();
        self.push(format!("DWT · Drizzle {} 叠加开始", license::VERSION), LogKind::Ok);
        self.push(
            format!(
                "命令：dwt-drizzle --photos \"{}\" --out \"{}\" --tag \"{}\" --A {} --rej-k {} --pick {}",
                cfg.photos, cfg.out, cfg.tag, cfg.a, cfg.rej_k, cfg.pick
            ),
            LogKind::Dim,
        );
        self.push("=".repeat(60), LogKind::Dim);
        self.state = "运行中".into();
        self.run_started = Some(Instant::now());
        self.elapsed = Duration::ZERO;
        self.progress = Some(JobProgress { stage: "准备".into(), current: 0, total: 1 });
        self.cancel = Some(Arc::new(AtomicBool::new(false)));
        self.stop_requested = false;
        self.exit_at = None;
        // Python opens the monitor automatically.  Automation runs can turn
        // that off so the common-page log stays visible for screenshots.
        self.monitor_open = std::env::var("DWT_GUI_MONITOR").map(|v| v != "0").unwrap_or(true);
        self.started_once = true;

        let cancel = self.cancel.as_ref().unwrap().clone();
        let (tx, rx) = mpsc::channel::<JobEvent>();
        let tx_log = tx.clone();
        let tx_prog = tx.clone();
        let cfg2 = cfg.clone();
        std::thread::spawn(move || {
            let log = move |s: &str| {
                let _ = tx_log.send(JobEvent::Log(s.to_string()));
            };
            let progress = move |p: JobProgress| {
                let _ = tx_prog.send(JobEvent::Progress(p));
            };
            let res = dwt_backend::run_job_with_progress(&cfg2, log, progress, &cancel);
            let _ = tx.send(JobEvent::Done(res));
        });
        self.rx = Some(rx);
        ctx.request_repaint();
    }

    fn poll_job(&mut self, ctx: &egui::Context) {
        let events: Vec<JobEvent> = self
            .rx
            .as_ref()
            .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
            .unwrap_or_default();
        let mut done = false;
        for ev in events {
            match ev {
                JobEvent::Log(s) => self.push_auto(s),
                JobEvent::Progress(p) => {
                    self.progress = Some(p.clone());
                    self.stage_from_progress(&p);
                }
                JobEvent::Done(res) => {
                    done = true;
                    self.finish_run(res);
                }
            }
        }
        if done {
            self.rx = None;
            self.cancel = None;
            self.run_started = None;
            ctx.request_repaint();
        } else if self.rx.is_some() {
            // The worker only wakes us through the channel; keep a light
            // repaint rhythm so progress and log lines stay live.
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn stage_from_progress(&mut self, p: &JobProgress) {
        // The visible stage label is intentionally short, like the Tk app.
        let _ = p;
    }

    fn finish_run(&mut self, res: Result<JobResult, String>) {
        let elapsed = self.run_started.map(|t| t.elapsed()).unwrap_or(self.elapsed);
        self.elapsed = elapsed;
        match res {
            Ok(job) => {
                self.state = "完成".into();
                self.progress = Some(JobProgress { stage: "完成".into(), current: 1, total: 1 });
                self.last_out = self.params.out.clone();
                self.eval_path = job.image_path.clone();
                self.map_path = job.image_path.clone();
                self.last_stem = Path::new(&job.image_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("PWS")
                    .to_string();
                let out_msg = format!("[界面] 完成。产物在：{}", self.last_out);
                self.push(out_msg, LogKind::Ok);
                let out_line = format!(
                    "[输出] {}  {}×{}  用时 {:.1}s",
                    job.image_path, job.width, job.height, job.elapsed_secs
                );
                self.push(out_line, LogKind::Ok);
            }
            Err(e) => {
                if e.contains("取消") {
                    self.state = "已停止".into();
                    self.push("[界面] 已停止。", LogKind::Warn);
                } else {
                    self.state = "失败".into();
                    self.push(format!("[err] {e}"), LogKind::Err);
                }
            }
        }
        self.push(
            format!(
                "[界面] 结束时间：{}  用时 {}",
                chrono::Local::now().format("%H:%M:%S"),
                fmt_elapsed(elapsed)
            ),
            LogKind::Dim,
        );
        self.close_log_file();
        if self.exit_after_job {
            self.exit_at = Some(Instant::now() + Duration::from_millis(900));
        }
    }

    fn scan_channels(&mut self) {
        self.channels.clear();
        self.selected.clear();
        let root = PathBuf::from(self.params.photos.trim());
        if !root.is_dir() {
            self.push(format!("[err] 素材目录不存在：{}", root.display()), LogKind::Err);
            return;
        }
        let mut files = Vec::new();
        if let Err(e) = collect_xisf(&root, &mut files) {
            self.push(format!("[err] 扫描素材失败：{e}"), LogKind::Err);
            return;
        }
        let mut groups: BTreeMap<String, usize> = BTreeMap::new();
        for p in files {
            *groups.entry(filter_of(&p).unwrap_or_else(|| "UNKNOWN".into())).or_insert(0) += 1;
        }
        self.channels = groups.into_iter().collect();
        self.channels.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (name, _) in &self.channels {
            self.selected.insert(name.clone());
        }
        let scan_line = format!(
            "[分组] 扫描到 {} 帧，{} 个通道：{}",
            self.channels.iter().map(|(_, n)| n).sum::<usize>(),
            self.channels.len(),
            self.channels
                .iter()
                .map(|(k, n)| format!("{k}({n})"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        self.push(scan_line, LogKind::Ok);
    }

    fn load_maps(&mut self, ctx: &egui::Context) {
        let p = PathBuf::from(self.map_path.trim());
        if p.as_os_str().is_empty() || !p.exists() {
            self.map_msg = "成品文件不存在".into();
            return;
        }
        let rp = dwt_backend::find_aux(&p, "r");
        let np = dwt_backend::find_aux(&p, "neff");
        self.map_msg.clear();
        match rp.as_ref().map(|x| dwt_backend::load_npy_f32(x)) {
            Some(Ok(m)) => {
                self.map_msg.push_str(&format!("R：{}  ", rp.as_ref().unwrap().display()));
                self.tex_r = Some(build_texture(ctx, &m, false, "r_map"));
                self.map_r = Some(m);
            }
            Some(Err(e)) => self.map_msg.push_str(&format!("R 读取失败：{e}  ")),
            None => self.map_msg.push_str("未找到 R 场（_R_q4.npy）  "),
        }
        match np.as_ref().map(|x| dwt_backend::load_npy_f32(x)) {
            Some(Ok(m)) => {
                if let Some(p) = &np {
                    self.map_msg.push_str(&format!("N_eff：{}", p.display()));
                }
                self.tex_neff = Some(build_texture(ctx, &m, self.map_log, "neff_map"));
                self.map_neff = Some(m);
            }
            Some(Err(e)) => self.map_msg.push_str(&format!("  N_eff 读取失败：{e}")),
            None => self.map_msg.push_str("  未找到 N_eff（_neff_q4.npy）"),
        }
        let msg = format!("[界面] 权重场：{}", self.map_msg);
        self.push(msg, LogKind::Dim);
    }

    fn load_eval(&mut self) {
        let p = PathBuf::from(self.eval_path.trim());
        if p.as_os_str().is_empty() || !p.exists() {
            self.eval_msg = "请选择成品 .xisf".into();
            return;
        }
        match dwt_backend::eval_product(&p) {
            Ok(r) => {
                self.eval_msg = format!("已评价：{}", p.display());
                self.eval_report = Some(r);
            }
            Err(e) => {
                self.eval_msg = format!("评价失败：{e}");
                self.eval_report = None;
            }
        }
    }

    fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header")
            .exact_height(74.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::CARD)
                    .stroke(Stroke::new(1.0_f32, theme::BORDER))
                    .inner_margin(Margin::symmetric(18, 0)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(98.0, 42.0), Sense::hover());
                    let p = ui.painter();
                    p.rect_filled(rect, egui::CornerRadius::same(10), theme::NAVY);
                    p.text(
                        rect.center() + Vec2::new(0.0, -3.0),
                        Align2::CENTER_CENTER,
                        "DWT",
                        FontId::new(15.0, FontFamily::Proportional),
                        Color32::from_rgb(0xea, 0xf7, 0xff),
                    );
                    p.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(rect.left() + 12.0, rect.bottom() - 11.0),
                            Vec2::new(74.0, 4.0),
                        ),
                        egui::CornerRadius::same(2),
                        theme::CYAN,
                    );
                    ui.add_space(14.0);
                    let (bar, _) = ui.allocate_exact_size(Vec2::new(3.0, 22.0), Sense::hover());
                    ui.painter().rect_filled(bar, egui::CornerRadius::same(1), theme::CYAN);
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
                        ui.add_space(2.0);
                        ui.label(RichText::new("DWT · Drizzle").size(20.0).strong().color(theme::NAVY));
                        ui.label(
                            RichText::new("分区加权叠加 · Drizzle 几何重采样")
                                .size(11.0)
                                .color(theme::FG2),
                        );
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let c = theme::state_color(&self.state);
                        ui.label(RichText::new(&self.state).size(13.0).color(c).strong());
                        ui.label(RichText::new("状态").size(11.0).color(theme::FG2));
                    });
                });
            });
    }

    fn tabs(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("tabs")
            .exact_height(47.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG)
                    .inner_margin(Margin::symmetric(18, 6)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    for tab in [Tab::Common, Tab::Advanced, Tab::Eval, Tab::Maps] {
                        if theme::tab_button(ui, self.tab == tab, tab.label()).clicked() {
                            self.tab = tab;
                        }
                    }
                });
            });
    }

    fn bottom_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("bottom")
            .exact_height(54.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::CARD)
                    .stroke(Stroke::new(1.0_f32, theme::BORDER))
                    .inner_margin(Margin::symmetric(16, 9)),
            )
            .show(ctx, |ui| {
                let running = self.state == "运行中";
                ui.horizontal_centered(|ui| {
                    if ui
                        .add_enabled(!running, theme::primary_button("开始叠加"))
                        .on_hover_text("扫描并按当前参数叠加")
                        .clicked()
                    {
                        self.start_run(ctx);
                    }
                    if ui
                        .add_enabled(running, theme::danger_button("停止"))
                        .clicked()
                    {
                        self.stop_run();
                    }
                    ui.add_space(4.0);
                    if ui.button("存为默认").clicked() {
                        self.save_settings(false);
                    }
                    if ui.button("查看权重场").clicked() {
                        self.tab = Tab::Maps;
                        if !self.map_path.is_empty() {
                            self.load_maps(ctx);
                        }
                    }
                    if ui.button("运行监控").clicked() {
                        self.monitor_open = true;
                    }
                    if ui.button("使用说明").clicked() {
                        self.help_open = true;
                    }
                    if ui.button("关于").clicked() {
                        self.about_open = true;
                    }
                    if ui.button("自动调参").clicked() {
                        self.tab = Tab::Advanced;
                        self.push(
                            "[界面] 自动调参：高级页可设「调参 A 候选」与「调参帧数」；当前版本先手动选择 A，再运行叠加。",
                            LogKind::Dim,
                        );
                    }
                    ui.add_space(8.0);
                    let frac = if running {
                        self.progress.as_ref().map(|p| p.fraction()).unwrap_or(0.35)
                    } else if self.state == "完成" {
                        1.0
                    } else {
                        0.0
                    };
                    ui.add(egui::ProgressBar::new(frac).desired_width(180.0).show_percentage());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let stage = self
                            .progress
                            .as_ref()
                            .map(|p| p.stage.as_str())
                            .unwrap_or("—");
                        let el = if let Some(t) = self.run_started {
                            t.elapsed()
                        } else {
                            self.elapsed
                        };
                        ui.label(
                            RichText::new(format!("用时 {}", fmt_elapsed(el)))
                                .size(11.0)
                                .color(theme::FG2),
                        );
                        ui.add_space(16.0);
                        ui.label(RichText::new(format!("阶段：{stage}")).size(11.0).color(theme::FG2));
                    });
                });
            });
    }

    fn central(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let width = ui.available_width().min(1260.0);
                        let pad = ((ui.available_width() - width) * 0.5).max(0.0);
                        ui.horizontal(|ui| {
                            ui.add_space(pad);
                            ui.vertical(|ui| {
                                ui.set_width(width);
                                ui.add_space(12.0);
                                match self.tab {
                                    Tab::Common => self.common_page(ui),
                                    Tab::Advanced => self.advanced_page(ui),
                                    Tab::Eval => self.eval_page(ui),
                                    Tab::Maps => self.maps_page(ui),
                                }
                                ui.add_space(20.0);
                            });
                        });
                    });
            });
    }

    fn common_page(&mut self, ui: &mut Ui) {
        let mut scan = false;
        let mut open_out = false;
        let mut select_all = false;
        let mut select_none = false;
        let mut rgb = false;

        theme::card(ui, "素材与输出", |ui| {
            let row = |ui: &mut Ui, label: &str, value: &mut String, browse: bool| {
                ui.horizontal(|ui| {
                    ui.add_sized([76.0, 24.0], Label::new(RichText::new(label).color(theme::FG)));
                    let w = (ui.available_width() - if browse { 76.0 } else { 0.0 }).max(120.0);
                    ui.add_sized([w, 24.0], theme::line_edit(value));
                    if browse && ui.button("浏览…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_folder() {
                            *value = p.display().to_string();
                        }
                    }
                });
            };
            row(ui, "素材目录", &mut self.params.photos, true);
            row(ui, "输出目录", &mut self.params.out, true);
            ui.horizontal(|ui| {
                ui.add_sized([76.0, 24.0], Label::new("成品名"));
                ui.add_sized([260.0, 24.0], theme::line_edit(&mut self.params.tag));
                theme::hint(ui, "   产物：PWS_<成品名>_<滤镜>.xisf（另附 R 场 / N_eff / 权重表）");
            });
            ui.add_space(4.0);
            theme::hint(
                ui,
                "素材要「已校准、未对齐」：WBPP 的 calibrated\\ 可直接用，registered\\ 不要用；Flat / Dark 会自动跳过，各通道自动分组",
            );
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("扫描通道").clicked() {
                    scan = true;
                }
                if ui.button("全选").clicked() {
                    select_all = true;
                }
                if ui.button("全不选").clicked() {
                    select_none = true;
                }
                if ui.button("重新合成 RGB…").clicked() {
                    rgb = true;
                }
                if ui.button("打开输出目录").clicked() {
                    open_out = true;
                }
                theme::hint(ui, "先扫描，再在下面选要叠的通道（不选 = 全部）");
            });
            ui.add_space(4.0);
            egui::Frame::new()
                .fill(Color32::from_rgb(0xfa, 0xfd, 0xff))
                .stroke(Stroke::new(1.0_f32, theme::BORDER))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(Margin::same(8))
                .show(ui, |ui| {
                    ui.set_min_height(50.0);
                    if self.channels.is_empty() {
                        theme::hint(ui, "点「扫描通道」列出可叠的通道（不选 = 全部）");
                    } else {
                        let channels = self.channels.clone();
                        egui::ScrollArea::vertical().max_height(48.0).show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for (name, n) in channels {
                                    let mut checked = self.selected.contains(&name);
                                    if ui
                                        .checkbox(&mut checked, format!("{name}  ({n})"))
                                        .changed()
                                    {
                                        if checked {
                                            self.selected.insert(name.clone());
                                        } else {
                                            self.selected.remove(&name);
                                        }
                                    }
                                }
                            });
                        });
                    }
                });
        });

        if open_out {
            let p = PathBuf::from(&self.params.out);
            if p.is_dir() {
                Params::open_dir(&p);
            }
        }
        if scan {
            self.scan_channels();
        }
        if select_all {
            self.selected = self.channels.iter().map(|(n, _)| n.clone()).collect();
        }
        if select_none {
            self.selected.clear();
        }
        if rgb {
            let mut chosen = BTreeSet::new();
            for name in ["R", "G", "B"] {
                if self.channels.iter().any(|(n, _)| n == name) {
                    chosen.insert(name.to_string());
                }
            }
            if chosen.is_empty() {
                self.push("[警告] 未找到 R/G/B 通道，仍按全部通道处理", LogKind::Warn);
            } else {
                self.selected = chosen;
                self.push(
                    "[界面] 已选择 RGB 通道；实际运行仍按单通道叠加，RGB 合成请分别运行后合成。",
                    LogKind::Dim,
                );
            }
        }

        theme::card(ui, "叠加参数", |ui| {
            ui.columns(4, |cols| {
                let fields: [(&str, &str, &mut String); 4] = [
                    ("清晰度 A", "彩机建议 6（等效帧数更高、噪声低）；黑白建议 20（更锐）；14 折中", &mut self.params.a),
                    ("排异 κ", "3.5σ，越大越保守", &mut self.params.rej_k),
                    ("星点数", "120，视场大可以加", &mut self.params.n_star),
                    ("限帧", "只用前 N 帧；0 = 全部", &mut self.params.limit),
                ];
                for (i, (label, hint_text, value)) in fields.into_iter().enumerate() {
                    cols[i].vertical(|ui| {
                        theme::field_label(ui, label);
                        ui.add_sized([84.0, 24.0], theme::line_edit(value));
                        theme::hint(ui, hint_text);
                    });
                }
            });
        });

        theme::card(ui, "Drizzle", |ui| {
            ui.checkbox(
                &mut self.params.drizzle,
                "启用 drizzle 放大（取消 = 1× 网格，仍做配准）",
            );
            ui.add_space(6.0);
            ui.columns(2, |cols| {
                let c0 = &mut cols[0];
                c0.horizontal(|ui| {
                    ui.label("输出倍率");
                    ui.add_sized([72.0, 24.0], theme::line_edit(&mut self.params.scale));
                    theme::hint(ui, "σ 1.2–2.5px 用 2.0；比 2.5 大用 1.5");
                });
                c0.horizontal(|ui| {
                    ui.label("pixfrac");
                    ui.add_sized([72.0, 24.0], theme::line_edit(&mut self.params.pix));
                    theme::hint(ui, "1.0 最稳；确认有亚像素抖动才降到 0.7–0.8");
                });
                let c1 = &mut cols[1];
                c1.horizontal(|ui| {
                    ui.label("裁剪边长");
                    ui.add_sized([72.0, 24.0], theme::line_edit(&mut self.params.crop));
                    theme::hint(ui, "0 = 整幅；例 500。");
                });
                c1.horizontal(|ui| {
                    ui.label("R 场平滑");
                    ui.add_sized([72.0, 24.0], theme::line_edit(&mut self.params.r_blur));
                    theme::hint(ui, "0=原式；8 让星云区也参与锐度加权");
                });
            });
        });

        let mut clear = false;
        theme::card(ui, "日志", |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("实时输出").size(11.0).color(theme::FG2));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("清屏").clicked() {
                        clear = true;
                    }
                });
            });
            ui.add_space(4.0);
            egui::Frame::new()
                .fill(theme::LOG_BG)
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.set_min_height(140.0);
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .max_height(160.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            for line in &self.log {
                                ui.add(
                                    Label::new(
                                        RichText::new(&line.text)
                                            .font(FontId::monospace(12.0))
                                            .color(line.color()),
                                    )
                                    .selectable(true)
                                    .wrap_mode(egui::TextWrapMode::Extend),
                                );
                            }
                        });
                });
            if let Some(p) = &self.log_path {
                theme::hint(ui, &format!("日志文件：{}", p.display()));
            }
        });
        if clear {
            self.log.clear();
            self.push("[界面] 日志已清屏。", LogKind::Dim);
        }
    }

    fn advanced_page(&mut self, ui: &mut Ui) {
        theme::card(ui, "高级参数", |ui| {
            theme::hint(
                ui,
                "参数清单与 core/pws_params.py 对齐；空值按 Python 的“默认/自动”处理。数值控件采用可编辑文本框，便于粘贴参数。",
            );
            ui.add_space(6.0);
            {
                    theme::text_row(ui, "背景梯度", &mut self.params.bgdeg, "0 关 / 1 平面（推荐）/ 2 二阶");
                    theme::choice_row(
                        ui,
                        "R 场定义",
                        &mut self.params.r_mode,
                        &[("max", "max 原式"), ("band", "band 带通"), ("ctn", "ctn 对比度")],
                        "max 原式 / band 带通（星云细节好）/ ctn 对比度",
                    );
                    theme::text_row(ui, "清晰度范围", &mut self.params.sharp_w, "空/0 = 全场；2 = 只作用星点");
                    theme::text_row(ui, "FWHM 上限", &mut self.params.fwhm_max, "0/空 = 不筛（一般用 A 就够）");
                    theme::unsupported_choice(
                        ui,
                        "计算设备",
                        &mut self.params.gpu,
                        &[("auto", "自动"), ("cpu", "只用 CPU"), ("wgpu", "wgpu（跨平台）")],
                        "自动，或强制 wgpu；失败时自动回退 CPU",
                    );
                    theme::text_row(ui, "吞吐权重", &mut self.params.flux_pow, "0 关 / 2 推荐：按每帧收光量加权");
                    theme::text_row(ui, "自动裁边", &mut self.params.crop_cover, "0 关 / 0.9 推荐；裁掉空四角与薄边带");
                    theme::text_row(ui, "径向背景", &mut self.params.bg_r, "0 关 / 2 拟 r² / 4 拟 r²+r⁴");
                    theme::unsupported_text(ui, "调参 A 候选", &mut self.params.tune_a, "自动调参时尝试的 A 值，逗号分隔");
                    theme::unsupported_check(ui, "原尺寸副本", &mut self.params.native_1x, "另存一份原生尺寸（1×，块平均）");
                    theme::check_row(ui, "对齐输入", &mut self.params.aligned, "输入已对齐（跳过配准与插值）");
                    theme::choice_row(
                        ui,
                        "彩机拜耳阵列",
                        &mut self.params.cfa,
                        &[("off", "关"), ("auto", "自动识别"), ("RGGB", "RGGB"), ("BGGR", "BGGR"), ("GRBG", "GRBG"), ("GBRG", "GBRG")],
                        "单色素材保持“关”",
                    );
                    theme::choice_row(
                        ui,
                        "拆色算法",
                        &mut self.params.cfa_debayer,
                        &[("bilinear", "双线性"), ("superpixel", "2×2 合并")],
                        "双线性更锐；2×2 合并更稳",
                    );
                    theme::choice_row(
                        ui,
                        "彩色权重",
                        &mut self.params.cfa_weights,
                        &[("auto", "自动"), ("shared", "共用"), ("own", "逐通道")],
                        "自动按通道信噪比决定",
                    );
                    theme::unsupported_check(ui, "复用配准缓存", &mut self.params.cache, "跨运行复用配准结果");
                    theme::check_row(ui, "排异按噪声归一", &mut self.params.rej_z, "按每帧噪声归一后排异");
                    theme::text_row(ui, "单帧份额上限", &mut self.params.max_share, "0.25 = 单帧最多 25% 权重");
                    theme::unsupported_text(ui, "参考像帧数", &mut self.params.ref_k, "0 = 自动（取最锐的 8 帧）");
                    theme::choice_row(
                        ui,
                        "彩机画布",
                        &mut self.params.cfa_grid,
                        &[("shared", "共用画布"), ("own", "逐通道画布")],
                        "彩机对齐后共用几何或逐通道独立",
                    );
                    theme::choice_row(
                        ui,
                        "彩机对齐",
                        &mut self.params.cfa_align,
                        &[("auto", "自动判断"), ("shared", "共用几何"), ("own", "逐通道独立")],
                        "自动判断通常最好",
                    );
                    theme::text_row(ui, "并行进程", &mut self.params.jobs, "0 = 自动（CPU 核数）");
                    theme::path_row(ui, "彩机参考目录", &mut self.params.cfa_ref, "彩机各通道配准参考目录（可选）", true);
                    theme::choice_row(
                        ui,
                        "真 CFA 采样",
                        &mut self.params.cfa_raw,
                        &[("auto", "自动"), ("on", "强制开"), ("off", "强制关（走插值）"), ("hybrid", "混合（G 真 CFA + R/B 插值）")],
                        "自动按素材类型选择",
                    );
                    theme::choice_row(
                        ui,
                        "排异尺度",
                        &mut self.params.rej_mode,
                        &[("winsor", "winsorized（推荐）"), ("mad", "中位绝对偏差（旧）"), ("fast", "平均绝对偏差（最快）"), ("esd", "广义 ESD（少帧/拖线）")],
                        "默认 winsorized",
                    );
                    theme::choice_row(
                        ui,
                        "液滴核",
                        &mut self.params.kernel,
                        &[("box", "方框（快）"), ("psfmatch", "PSF 匹配（更干净）")],
                        "drizzle 落点核",
                    );
                    theme::text_row(ui, "PSF 目标", &mut self.params.psf_target, "1..99，默认 50");
                    theme::check_row(ui, "满月帧补偿", &mut self.params.detail_snr, "细节区也按 1/σ² 加权（β=1）");
                    theme::choice_row(
                        ui,
                        "彩色平衡",
                        &mut self.params.rgb_balance,
                        &[("off", "不平（与 WBPP 一致）"), ("add", "加性平衡"), ("mul", "乘性平衡")],
                        "默认保持与 WBPP 一致",
                    );
                    theme::check_row(ui, "两遍加权", &mut self.params.two_pass, "先估权重再叠一遍（更慢）");
                    theme::text_row(ui, "包络窗", &mut self.params.env_win_mult, "默认 2.0；R 场包络窗口倍数");
                    theme::text_row(ui, "包络噪声项", &mut self.params.env_eps, "默认 16.0；R 场包络噪声项");
                    theme::text_row(ui, "包络形状幂", &mut self.params.env_p, "默认 1.0；R 场包络形状幂");
                    theme::text_row(ui, "R 带通窗宽", &mut self.params.r_win, "0 = 关；band 模式下有效");
                    theme::text_row(ui, "对比度阈值", &mut self.params.r_k, "ctn 模式的对比度阈值，默认 3.0");
                    theme::text_row(ui, "FWHM 搜索", &mut self.params.fwhm_search, "0 = 不搜索（默认，与 Python 一致）");
                    theme::text_row(ui, "星点生长 %", &mut self.params.star_grow_pct, "R 场包络星点生长百分比，默认 100");
                    theme::text_row(ui, "R 场下限", &mut self.params.r_floor, "0..0.95，默认 0");
                    theme::unsupported_text(ui, "C 星点数", &mut self.params.c_star, "清晰度测量星点数，默认 64");
                    theme::text_row(ui, "信噪比裁剪", &mut self.params.snr_clip, "S 权重限幅 K，默认 3；0 = 关");
                    theme::unsupported_text(ui, "调参帧数", &mut self.params.tune, "0 = 不调参");
                    theme::unsupported_text(ui, "调参进程", &mut self.params.tune_jobs, "自动调参并行数，默认 2");
                    theme::unsupported_path(ui, "帧缓存目录", &mut self.params.frames_dir, "可选：已归一化帧缓存目录", true);
                    theme::unsupported_check(ui, "保留帧缓存", &mut self.params.keep_frames, "调参时保留中间帧");
                    theme::unsupported_text(ui, "清晰度分块", &mut self.params.sharp_tiles, "0 = 自动；分块测清晰度");
                    theme::text_row(ui, "裁切中心", &mut self.params.center, "cy,cx 或 cy×cx；空 = 自动找目标");
                    theme::check_row(ui, "彩机 drizzle", &mut self.params.cfa_drizzle, "彩机按通道做 drizzle");
                    theme::text_row(ui, "彩机 drizzle 倍率", &mut self.params.cfa_drizzle_scale, "1..4，默认 2");
                    theme::check_row(ui, "线性噪声", &mut self.params.ln, "按线性噪声模型归一");
                    theme::check_row(ui, "裁切匹配缩放", &mut self.params.match_scale, "用裁切参考文件匹配缩放");
                    theme::text_row(ui, "通道 pick", &mut self.params.pick, "H / O / S / R / G / B；留空自动选最大组");
                    theme::text_row(ui, "S 指数 B", &mut self.params.b, "S 权重指数，默认 2.0");
                    theme::text_row(ui, "排异 κ（高级）", &mut self.params.rej_k, "3.5σ，越大越保守");
                    theme::text_row(ui, "清晰度 A（高级）", &mut self.params.a, "与常用页同步");
                    theme::text_row(ui, "星点数（高级）", &mut self.params.n_star, "与常用页同步");
                    theme::unsupported_path(ui, "几何目录", &mut self.params.xdrz, "WBPP 的 .xdrz 目录（跨夜旋转建议填）", true);
                    theme::unsupported_path(ui, "裁切参考", &mut self.params.match_path, "可选：裁切参考 XISF", false);
                    theme::path_row(ui, "输出网格", &mut self.params.frame, "填 master：成品宽高/朝向与它一致", false);
            }
        });
    }

    fn eval_page(&mut self, ui: &mut Ui) {
        let mut run = false;
        theme::card(ui, "叠加评价", |ui| {
            ui.horizontal(|ui| {
                ui.add_sized([76.0, 24.0], Label::new("成品 XISF"));
                let w = (ui.available_width() - 150.0).max(180.0);
                ui.add_sized([w, 24.0], theme::line_edit(&mut self.eval_path));
                if ui.button("浏览…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("XISF", &["xisf"]).pick_file() {
                        self.eval_path = p.display().to_string();
                    }
                }
                if ui.button("开始评价").clicked() {
                    run = true;
                }
            });
            theme::hint(ui, "自动寻找同名 _R_q4.npy / _neff_q4.npy / _weights.txt；评价只读，不修改成品。");
            if !self.eval_msg.is_empty() {
                ui.label(RichText::new(&self.eval_msg).size(11.0).color(theme::FG2));
            }
        });
        if run {
            self.load_eval();
        }
        let Some(r) = &self.eval_report else {
            theme::card(ui, "评价结果", |ui| {
                theme::hint(ui, "选择成品后点「开始评价」。");
            });
            return;
        };
        theme::card(ui, "图像指标", |ui| {
            egui::Grid::new("eval_main").num_columns(4).spacing([18.0, 6.0]).show(ui, |ui| {
                metric(ui, "尺寸", &format!("{}×{}", r.width, r.height));
                metric(ui, "星点数", &r.star_count.to_string());
                metric(ui, "天空背景", &format!("{:.5}", r.sky));
                metric(ui, "噪声 σ", &format!("{:.5}", r.sigma));
                ui.end_row();
                metric(ui, "FWHM 中位", &format!("{:.2} px", r.fwhm));
                metric(ui, "R 场", &r.r.as_ref().map(|s| format!("mean {:.2}", s.mean)).unwrap_or_else(|| "—".into()));
                metric(ui, "N_eff", &r.neff.as_ref().map(|s| format!("P50 {:.2}", s.p50)).unwrap_or_else(|| "—".into()));
                metric(ui, "权重表", if r.weights.is_some() { "已找到" } else { "无" });
                ui.end_row();
            });
        });
        theme::card(ui, "径向 FWHM", |ui| {
            for (label, v) in &r.rings {
                ui.horizontal(|ui| {
                    ui.add_sized([90.0, 20.0], Label::new(label));
                    let frac = if v.is_finite() { (v / 8.0).clamp(0.0, 1.0) } else { 0.0 };
                    ui.add(egui::ProgressBar::new(frac).desired_width(240.0).text(if v.is_finite() { format!("{v:.2} px") } else { "—".into() }));
                });
            }
        });
        theme::card(ui, "R 场统计", |ui| stat_ui(ui, r.r.as_ref()));
        theme::card(ui, "N_eff 统计", |ui| stat_ui(ui, r.neff.as_ref()));
        if let Some(w) = &r.weights {
            theme::card(ui, "权重表（尾部）", |ui| {
                egui::Frame::new().fill(theme::LOG_BG).inner_margin(Margin::same(8)).show(ui, |ui| {
                    ui.add(Label::new(RichText::new(w).monospace().size(11.0).color(theme::LOG_FG)).selectable(true));
                });
            });
        }
        if !r.aux_note.is_empty() {
            theme::hint(ui, &r.aux_note);
        }
    }

    fn maps_page(&mut self, ui: &mut Ui) {
        let mut load = false;
        let mut save = false;
        theme::card(ui, "权重场 / N_eff", |ui| {
            ui.horizontal(|ui| {
                ui.add_sized([76.0, 24.0], Label::new("成品 XISF"));
                let w = (ui.available_width() - 230.0).max(180.0);
                ui.add_sized([w, 24.0], theme::line_edit(&mut self.map_path));
                if ui.button("浏览…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("XISF", &["xisf"]).pick_file() {
                        self.map_path = p.display().to_string();
                    }
                }
                if ui.button("载入").clicked() {
                    load = true;
                }
                if ui.button("保存 PNG…").clicked() {
                    save = true;
                }
            });
            ui.checkbox(&mut self.map_log, "N_eff 对数色标");
            if !self.map_msg.is_empty() {
                theme::hint(ui, &self.map_msg);
            }
        });
        if load {
            self.load_maps(ui.ctx());
        }
        if save {
            self.save_map_png();
        }
        if !self.map_log {
            // texture already built with the current scale; no action
        }
        ui.columns(2, |cols| {
            map_panel(&mut cols[0], "R 场（锐度/权重）", self.tex_r.as_ref(), self.map_r.as_ref());
            map_panel(&mut cols[1], "N_eff（有效帧数）", self.tex_neff.as_ref(), self.map_neff.as_ref());
        });
    }

    fn save_map_png(&mut self) {
        let (Some(r), Some(n)) = (&self.map_r, &self.map_neff) else {
            self.map_msg = "请先载入 R 场与 N_eff".into();
            return;
        };
        let path = rfd::FileDialog::new()
            .set_file_name("weights.png")
            .add_filter("PNG", &["png"])
            .save_file();
        let Some(path) = path else { return };
        // Side-by-side RGBA canvas, nearest-neighbour, 1:1 pixels.
        let (w, h) = (r.w + n.w + 8, r.h.max(n.h));
        let mut rgba = vec![255u8; w * h * 4];
        blit_map(&mut rgba, w, r, 0, 0, false);
        blit_map(&mut rgba, w, n, r.w + 8, 0, self.map_log);
        match image::save_buffer(&path, &rgba, w as u32, h as u32, image::ColorType::Rgba8) {
            Ok(()) => {
                self.map_msg = format!("已保存：{}", path.display());
                let msg = format!("[界面] 已保存权重场 PNG：{}", path.display());
                self.push(msg, LogKind::Dim);
            }
            Err(e) => self.map_msg = format!("保存失败：{e}"),
        }
    }

    fn windows(&mut self, ctx: &egui::Context) {
        if self.license_open {
            ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("license_dim"),
            ))
            .rect_filled(ctx.screen_rect(), egui::CornerRadius::ZERO, Color32::from_black_alpha(60));
            let mut open = self.license_open;
            egui::Window::new("关于本软件的方法来源")
                .collapsible(false)
                .resizable(false)
                .default_width(600.0)
                .min_width(560.0)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_min_width(520.0);
                    ui.label(RichText::new("DWT · Drizzle").size(20.0).strong().color(theme::NAVY));
                    ui.label(
                        RichText::new("使用前请先阅读下面的方法来源与版权说明")
                            .size(11.0)
                            .color(theme::FG2),
                    );
                    ui.add_space(8.0);
                    egui::ScrollArea::vertical().max_height(330.0).max_width(560.0).show(ui, |ui| {
                        ui.add(Label::new(RichText::new(license::LICENSE_TEXT).size(13.0).color(theme::FG)).wrap());
                    });
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("打开原文链接").clicked() {
                            theme::open_url(license::URL);
                        }
                        if ui.button("我知道了").clicked() {
                            open = false;
                        }
                    });
                });
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                open = false;
            }
            self.license_open = open;
        }

        if self.about_open {
            let mut open = self.about_open;
            egui::Window::new("关于")
                .collapsible(false)
                .resizable(true)
                .default_width(560.0)
                .show(ctx, |ui| {
                    ui.label(RichText::new(format!("DWT · Drizzle {}", license::VERSION)).strong().size(16.0));
                    ui.add_space(6.0);
                    egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                        ui.add(Label::new(RichText::new(license::LICENSE_TEXT).size(12.0)).wrap());
                    });
                    if ui.button("关闭").clicked() {
                        open = false;
                    }
                });
            self.about_open = open;
        }

        if self.help_open {
            let mut open = self.help_open;
            egui::Window::new("使用说明")
                .collapsible(false)
                .resizable(true)
                .default_width(700.0)
                .default_height(560.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.add(Label::new(RichText::new(license::HELP_TEXT).size(13.0)).wrap());
                    });
                    if ui.button("关闭").clicked() {
                        open = false;
                    }
                });
            self.help_open = open;
        }

        if self.monitor_open {
            let mut open = self.monitor_open;
            let mut stop = false;
            egui::Window::new("运行监控")
                .collapsible(false)
                .resizable(true)
                .default_width(620.0)
                .default_height(480.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("状态：{}", self.state)).strong().color(theme::state_color(&self.state)));
                        ui.add_space(20.0);
                        let stage = self.progress.as_ref().map(|p| p.stage.as_str()).unwrap_or("—");
                        ui.label(format!("阶段：{stage}"));
                        ui.add_space(20.0);
                        ui.label(format!("用时 {}", fmt_elapsed(self.run_started.map(|t| t.elapsed()).unwrap_or(self.elapsed))));
                    });
                    let frac = self.progress.as_ref().map(|p| p.fraction()).unwrap_or(0.0);
                    ui.add(egui::ProgressBar::new(frac).show_percentage());
                    ui.separator();
                    ui.label(RichText::new("最近日志").strong());
                    egui::Frame::new().fill(theme::LOG_BG).inner_margin(Margin::same(8)).show(ui, |ui| {
                        egui::ScrollArea::vertical().stick_to_bottom(true).max_height(260.0).show(ui, |ui| {
                            for line in self.log.iter().rev().take(80).rev() {
                                ui.add(Label::new(RichText::new(&line.text).font(FontId::monospace(11.0)).color(line.color())).wrap());
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        if ui.add_enabled(self.state == "运行中", theme::danger_button("停止")).clicked() {
                            stop = true;
                        }
                        if ui.button("关闭").clicked() {
                            open = false;
                        }
                    });
                });
            if stop {
                self.stop_run();
            }
            self.monitor_open = open;
        }
    }

    /// 自证模式的一帧：收图 → 换页 → 请求下一张。
    fn dump_tick(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.dump.take() else {
            return;
        };
        let _ = std::fs::create_dir_all(&d.dir);

        // 1) 收上一帧请求的截图（eframe 在 paint 之后 read_pixels，下一帧作为事件送进来）
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = shot {
            if d.requested {
                let name = DUMP_NAMES[d.index].to_string();
                let path = d.dir.join(&name);
                let [w, h] = img.size;
                let raw: &[u8] = img.as_raw();
                let bytes = raw.len() as u64;
                if let Err(e) =
                    image::save_buffer(&path, raw, w as u32, h as u32, image::ExtendedColorType::Rgba8)
                {
                    self.dump_fail(&d, format!("写 {} 失败：{e}", path.display()));
                    return;
                }
                // 顺手自检：整幅图是否纯色（纯色说明没渲染出东西）
                let uniform = img.pixels.windows(2).all(|p| p[0] == p[1]);
                self.log.push(LogLine {
                    text: format!(
                        "[dump] {name}  {w}x{h}  {bytes} B{}",
                        if uniform { "  ← 纯色！" } else { "" }
                    ),
                    kind: if uniform { LogKind::Err } else { LogKind::Ok },
                });
                d.written.push((name, [w, h], bytes));
                d.index += 1;
                d.frames_on_page = 0;
                d.requested = false;
            }
        }

        // 2) 五张都拿到 → 写报告，退出 0
        if d.index >= DUMP_NAMES.len() {
            dump_write_report(&d);
            std::process::exit(0);
        }

        // 3) 换页（最后一页把「关于」弹窗叠上去）
        self.tab = d.tab();
        self.about_open = d.index == 4;
        self.license_open = false;

        // 4) 页面稳定若干帧后再请求截图
        if !d.requested && d.frames_on_page >= DUMP_SETTLE_FRAMES {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            d.requested = true;
        }
        d.frames_on_page = d.frames_on_page.saturating_add(1);

        if d.started.elapsed() > DUMP_DEADLINE {
            self.dump_fail(&d, format!("超时（{:.1}s）", d.started.elapsed().as_secs_f32()));
            return;
        }

        ctx.request_repaint();
        self.dump = Some(d);
    }

    fn dump_fail(&mut self, d: &DumpPages, why: String) {
        self.log.push(LogLine { text: format!("[dump] 失败：{why}"), kind: LogKind::Err });
        dump_write_report(d);
        std::process::exit(3);
    }
}

fn dump_write_report(d: &DumpPages) {
    let mut s = String::from("DWT · Drizzle 自证截图\n");
    for (name, [w, h], bytes) in &d.written {
        s.push_str(&format!("{name}\t{w}x{h}\t{bytes} B\n"));
    }
    s.push_str(&format!("elapsed\t{:.2} s\n", d.started.elapsed().as_secs_f32()));
    let _ = std::fs::write(d.dir.join("_dump_report.txt"), s);
}

impl eframe::App for DrizzleApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frames += 1;
        if self.dump.is_some() {
            // 自证模式：自己切页、自己截图，不走人的点击
            self.dump_tick(ctx);
        }
        self.poll_job(ctx);
        if self.state == "运行中" {
            ctx.request_repaint_after(Duration::from_millis(120));
        }
        if let Some(t) = self.run_started {
            self.elapsed = t.elapsed();
        }
        if self.state == "运行中" && !self.stop_requested {
            if let Some(limit) = self.stop_after {
                if self.run_started.map(|t| t.elapsed() >= limit).unwrap_or(false) {
                    self.stop_run();
                }
            }
        }
        if self.auto_start && !self.started_once && self.frames >= 3 {
            self.auto_start = false;
            self.start_run(ctx);
        }
        if let Some(at) = self.exit_at {
            if Instant::now() >= at {
                self.exit_at = None;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                ctx.request_repaint_after(Duration::from_millis(100));
            }
        }
        if self.tab == Tab::Maps
            && !self.map_auto_loaded
            && !self.map_path.is_empty()
            && Path::new(&self.map_path).exists()
        {
            self.map_auto_loaded = true;
            self.load_maps(ctx);
        }
        self.header(ctx);
        self.tabs(ctx);
        self.bottom_bar(ctx);
        self.central(ctx);
        self.windows(ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(c) = &self.cancel {
            c.store(true, Ordering::Relaxed);
        }
        self.close_log_file();
    }
}

fn metric(ui: &mut Ui, label: &str, value: &str) {
    ui.vertical(|ui| {
        ui.set_min_width(150.0);
        ui.label(RichText::new(label).size(11.0).color(theme::FG2));
        ui.label(RichText::new(value).size(16.0).strong().color(theme::FG));
    });
}

fn stat_ui(ui: &mut Ui, s: Option<&StatBlock>) {
    let Some(s) = s else {
        theme::hint(ui, "没有对应的 .npy（R 场 / N_eff）");
        return;
    };
    egui::Grid::new("stat_grid").num_columns(6).spacing([18.0, 6.0]).show(ui, |ui| {
        for (i, (k, v)) in [
            ("像素数", format!("{}", s.count)),
            ("最小 / 最大", format!("{:.3} / {:.3}", s.min, s.max)),
            ("均值", format!("{:.3}", s.mean)),
            ("P10", format!("{:.3}", s.p10)),
            ("P50", format!("{:.3}", s.p50)),
            ("P90", format!("{:.3}", s.p90)),
            ("零值比例", format!("{:.1}%", s.zero_frac * 100.0)),
            ("> 50", format!("{:.2}%", s.gt50 * 100.0)),
            ("< 10", format!("{:.2}%", s.lt10 * 100.0)),
        ]
        .into_iter()
        .enumerate()
        {
            ui.label(RichText::new(k).size(11.0).color(theme::FG2));
            ui.label(RichText::new(v).strong());
            if i % 3 == 2 {
                ui.end_row();
            }
        }
    });
}

fn map_panel(
    ui: &mut Ui,
    title: &str,
    tex: Option<&TextureHandle>,
    map: Option<&NpyMap>,
) {
    egui::Frame::new()
        .fill(theme::CARD)
        .stroke(Stroke::new(1.0_f32, theme::BORDER))
        .corner_radius(egui::CornerRadius::same(10))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            ui.label(RichText::new(title).strong().color(theme::FG));
            ui.add_space(4.0);
            if let (Some(t), Some(m)) = (tex, map) {
                let avail = ui.available_width();
                let h = (avail * m.h as f32 / m.w.max(1) as f32).min(380.0);
                let resp = ui.add(
                    egui::Image::new((t.id(), t.size_vec2()))
                        .fit_to_exact_size(Vec2::new(avail, h)),
                );
                if let Some(pos) = resp.hover_pos() {
                    let fx = ((pos.x - resp.rect.left()) / resp.rect.width()).clamp(0.0, 0.999);
                    let fy = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 0.999);
                    let x = (fx * m.w as f32) as usize;
                    let y = (fy * m.h as f32) as usize;
                    let v = m.data[y.min(m.h - 1) * m.w + x.min(m.w - 1)];
                    ui.label(
                        RichText::new(format!("x={x} y={y}  value={v:.3}"))
                            .size(11.0)
                            .color(theme::FG2),
                    );
                } else {
                    ui.label(
                        RichText::new(format!("{}×{}  P50 {:.3}", m.w, m.h, m.stat().p50))
                            .size(11.0)
                            .color(theme::FG2),
                    );
                }
            } else {
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 180.0), Sense::hover());
                theme::hint(ui, "载入成品后显示伪彩色图");
            }
        });
}

fn build_texture(ctx: &egui::Context, map: &NpyMap, log: bool, name: &str) -> TextureHandle {
    let mut vals: Vec<f32> = map.data.iter().copied().filter(|v| v.is_finite()).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    let lo = if vals.is_empty() { 0.0 } else { vals[(vals.len() as f32 * 0.02) as usize] };
    let hi = if vals.is_empty() { 1.0 } else { vals[(vals.len() as f32 * 0.98) as usize].max(lo + 1e-6) };
    let mut img = ColorImage::new([map.w, map.h], Color32::BLACK);
    for y in 0..map.h {
        for x in 0..map.w {
            let v = map.data[y * map.w + x];
            let t = if v.is_finite() {
                let mut t = (v - lo) / (hi - lo);
                if log {
                    t = (1.0 + t.max(0.0) * 9.0).ln() / 10.0_f32.ln();
                }
                t.clamp(0.0, 1.0)
            } else {
                0.0
            };
            img[(x, y)] = viridis(t);
        }
    }
    ctx.load_texture(name, img, TextureOptions::LINEAR)
}

fn blit_map(rgba: &mut [u8], stride_w: usize, map: &NpyMap, ox: usize, oy: usize, log: bool) {
    let mut vals: Vec<f32> = map.data.iter().copied().filter(|v| v.is_finite()).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    let lo = vals.first().copied().unwrap_or(0.0);
    let hi = vals.last().copied().unwrap_or(1.0).max(lo + 1e-6);
    for y in 0..map.h {
        for x in 0..map.w {
            let v = map.data[y * map.w + x];
            let mut t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            if log {
                t = (1.0 + t * 9.0).ln() / 10.0_f32.ln();
            }
            let c = viridis(t);
            let p = ((oy + y) * stride_w + ox + x) * 4;
            rgba[p] = c.r();
            rgba[p + 1] = c.g();
            rgba[p + 2] = c.b();
            rgba[p + 3] = 255;
        }
    }
}

fn viridis(t: f32) -> Color32 {
    // Small polynomial approximation of the viridis ramp.
    let t = t.clamp(0.0, 1.0);
    let r = (0.28 + 0.72 * t - 0.45 * t * t).clamp(0.0, 1.0);
    let g = (0.05 + 0.93 * t - 0.22 * t * t).clamp(0.0, 1.0);
    let b = (0.33 + 0.55 * (1.0 - t) - 0.45 * (1.0 - t).powi(2)).clamp(0.0, 1.0);
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn fmt_elapsed(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}", s / 60, s % 60)
}

fn collect_xisf(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        let lower = p.to_string_lossy().to_ascii_lowercase();
        let skip = ["flat", "dark", "bias", "registered", "registration", "aligned"]
            .iter()
            .any(|s| lower.split(['/', '\\']).any(|seg| seg.contains(s)));
        if p.is_dir() {
            if !skip {
                collect_xisf(&p, out)?;
            }
        } else if !skip
            && p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("xisf"))
        {
            out.push(p);
        }
    }
    Ok(())
}

fn filter_of(path: &Path) -> Option<String> {
    let s = path.to_string_lossy();
    let up = s.to_ascii_uppercase();
    if let Some(i) = up.find("FILTER-") {
        let rest = &s[i + 7..];
        let end = rest.find(['_', '\\', '/', '.']).unwrap_or(rest.len());
        let v = rest[..end].trim_matches(['\'', '"', ' ']);
        if !v.is_empty() {
            return Some(v.to_string());
        }
    }
    let stem = path.file_stem()?.to_string_lossy();
    stem.split('_')
        .nth(2)
        .filter(|v| v.len() == 1 && v.chars().all(|c| c.is_ascii_alphabetic()))
        .map(|v| v.to_ascii_uppercase())
}
