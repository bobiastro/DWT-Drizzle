//! GUI settings persisted as `config.toml`.
//!
//! Keys intentionally match the Python front end's `_collect` dictionary so a
//! settings file can be exchanged between the two implementations.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use dwt_backend::JobConfig;

#[derive(Debug, Clone)]
pub struct Params {
    pub photos: String,
    pub out: String,
    pub tag: String,
    pub a: String,
    pub rej_k: String,
    pub n_star: String,
    pub limit: String,
    pub crop: String,
    pub drizzle: bool,
    pub scale: String,
    pub pix: String,
    pub r_blur: String,
    pub bgdeg: String,
    pub r_mode: String,
    pub sharp_w: String,
    pub fwhm_max: String,
    pub xdrz: String,
    pub match_path: String,
    pub flux_pow: String,
    pub crop_cover: String,
    pub bg_r: String,
    pub frame: String,
    pub gpu: String,
    pub tune_a: String,
    pub native_1x: bool,
    pub aligned: bool,
    pub cfa: String,
    pub cfa_debayer: String,
    pub cfa_weights: String,
    pub cache: bool,
    pub rej_z: bool,
    pub max_share: String,
    pub ref_k: String,
    pub cfa_grid: String,
    pub cfa_align: String,
    pub jobs: String,
    pub cfa_ref: String,
    pub cfa_raw: String,
    pub rej_mode: String,
    pub kernel: String,
    pub psf_target: String,
    pub detail_snr: bool,
    pub rgb_balance: String,
    pub two_pass: bool,
    pub env_win_mult: String,
    pub env_eps: String,
    pub env_p: String,
    pub r_win: String,
    pub r_k: String,
    pub fwhm_search: String,
    pub star_grow_pct: String,
    pub r_floor: String,
    pub c_star: String,
    pub snr_clip: String,
    pub tune: String,
    pub tune_jobs: String,
    pub frames_dir: String,
    pub keep_frames: bool,
    pub sharp_tiles: String,
    pub center: String,
    pub cfa_drizzle: bool,
    pub cfa_drizzle_scale: String,
    pub ln: bool,
    pub match_scale: bool,
    pub pick: String,
    pub b: String,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            // The first-run values mirror gui.py's DEFAULTS.
            photos: r"G:\HFG1-2\calibrated".into(),
            out: r"G:\HFG1\drz".into(),
            tag: "drz".into(),
            a: "20".into(),
            rej_k: "3.5".into(),
            n_star: "120".into(),
            limit: "0".into(),
            crop: "0".into(),
            drizzle: true,
            scale: "2".into(),
            pix: "0.8".into(),
            r_blur: "0".into(),
            bgdeg: "1".into(),
            r_mode: "max".into(),
            sharp_w: String::new(),
            fwhm_max: String::new(),
            xdrz: String::new(),
            match_path: String::new(),
            flux_pow: "0.0".into(),
            crop_cover: "0.9".into(),
            bg_r: "0".into(),
            frame: String::new(),
            gpu: "auto".into(),
            tune_a: "10,14,20".into(),
            native_1x: false,
            aligned: false,
            cfa: "off".into(),
            cfa_debayer: "bilinear".into(),
            cfa_weights: "auto".into(),
            cache: true,
            rej_z: true,
            max_share: "0.25".into(),
            ref_k: "0".into(),
            cfa_grid: "shared".into(),
            cfa_align: "auto".into(),
            jobs: "0".into(),
            cfa_ref: String::new(),
            cfa_raw: "auto".into(),
            rej_mode: "winsor".into(),
            kernel: "box".into(),
            psf_target: "50".into(),
            detail_snr: false,
            rgb_balance: "off".into(),
            two_pass: false,
            env_win_mult: "2.0".into(),
            env_eps: "16.0".into(),
            env_p: "1.0".into(),
            r_win: "0".into(),
            r_k: "3.0".into(),
            fwhm_search: "0".into(),
            star_grow_pct: "100".into(),
            r_floor: "0".into(),
            c_star: "64".into(),
            snr_clip: "3".into(),
            tune: "0".into(),
            tune_jobs: "2".into(),
            frames_dir: String::new(),
            keep_frames: false,
            sharp_tiles: "0".into(),
            center: String::new(),
            cfa_drizzle: false,
            cfa_drizzle_scale: "2".into(),
            ln: false,
            match_scale: false,
            pick: "H".into(),
            b: "2".into(),
        }
    }
}

impl Params {
    pub fn config_path() -> PathBuf {
        if let Ok(p) = std::env::var("DWT_CONFIG") {
            if !p.trim().is_empty() {
                return PathBuf::from(p);
            }
        }
        if let Ok(appdata) = std::env::var("APPDATA") {
            return PathBuf::from(appdata).join("DWT_Drizzle").join("config.toml");
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                return dir.join("config.toml");
            }
        }
        PathBuf::from("config.toml")
    }

    pub fn load() -> (Self, Option<String>) {
        let path = Self::config_path();
        if !path.exists() {
            return (Self::default(), None);
        }
        match fs::read_to_string(&path) {
            Ok(text) => match Self::from_toml(&text) {
                Ok(p) => (p, Some(format!("已载入设置：{}", path.display()))),
                Err(e) => (Self::default(), Some(format!("读取 {} 失败（使用默认值）：{e}", path.display()))),
            },
            Err(e) => (Self::default(), Some(format!("读取 {} 失败（使用默认值）：{e}", path.display()))),
        }
    }

    pub fn save(&self) -> Result<PathBuf, String> {
        let path = Self::config_path();
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(|e| format!("创建设置目录失败: {e}"))?;
        }
        let text = self.to_toml().map_err(|e| format!("序列化设置失败: {e}"))?;
        fs::write(&path, text).map_err(|e| format!("写设置失败: {e}"))?;
        Ok(path)
    }

    pub fn from_toml(text: &str) -> Result<Self, String> {
        let table: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut p = Self::default();
        p.apply_table(&table);
        Ok(p)
    }

    fn apply_table(&mut self, t: &toml::Table) {
        let get = |k: &str| -> Option<String> {
            t.get(k).map(|v| match v {
                toml::Value::String(s) => s.clone(),
                toml::Value::Boolean(b) => if *b { "1".into() } else { "0".into() },
                toml::Value::Integer(i) => i.to_string(),
                toml::Value::Float(f) => f.to_string(),
                other => other.to_string(),
            })
        };
        let get_bool = |k: &str, d: bool| -> bool {
            match get(k).map(|v| v.to_ascii_lowercase()) {
                Some(v) => matches!(v.as_str(), "1" | "true" | "yes" | "on"),
                None => d,
            }
        };
        let str_fields: [(&str, &mut String); 48] = [
            ("photos", &mut self.photos),
            ("out", &mut self.out),
            ("tag", &mut self.tag),
            ("A", &mut self.a),
            ("rej_k", &mut self.rej_k),
            ("n_star", &mut self.n_star),
            ("limit", &mut self.limit),
            ("crop", &mut self.crop),
            ("scale", &mut self.scale),
            ("pix", &mut self.pix),
            ("r_blur", &mut self.r_blur),
            ("bgdeg", &mut self.bgdeg),
            ("r_mode", &mut self.r_mode),
            ("sharp_w", &mut self.sharp_w),
            ("fwhm_max", &mut self.fwhm_max),
            ("xdrz", &mut self.xdrz),
            ("match", &mut self.match_path),
            ("flux_pow", &mut self.flux_pow),
            ("crop_cover", &mut self.crop_cover),
            ("bg_r", &mut self.bg_r),
            ("frame", &mut self.frame),
            ("gpu", &mut self.gpu),
            ("tune_a", &mut self.tune_a),
            ("cfa", &mut self.cfa),
            ("cfa_debayer", &mut self.cfa_debayer),
            ("cfa_weights", &mut self.cfa_weights),
            ("max_share", &mut self.max_share),
            ("ref_k", &mut self.ref_k),
            ("cfa_grid", &mut self.cfa_grid),
            ("cfa_align", &mut self.cfa_align),
            ("jobs", &mut self.jobs),
            ("cfa_ref", &mut self.cfa_ref),
            ("cfa_raw", &mut self.cfa_raw),
            ("rej_mode", &mut self.rej_mode),
            ("kernel", &mut self.kernel),
            ("psf_target", &mut self.psf_target),
            ("rgb_balance", &mut self.rgb_balance),
            ("env_win_mult", &mut self.env_win_mult),
            ("env_eps", &mut self.env_eps),
            ("env_p", &mut self.env_p),
            ("r_win", &mut self.r_win),
            ("r_k", &mut self.r_k),
            ("fwhm_search", &mut self.fwhm_search),
            ("star_grow_pct", &mut self.star_grow_pct),
            ("r_floor", &mut self.r_floor),
            ("c_star", &mut self.c_star),
            ("snr_clip", &mut self.snr_clip),
            ("pick", &mut self.pick),
        ];
        for (k, dst) in str_fields {
            if let Some(v) = get(k) {
                *dst = v;
            }
        }
        if let Some(v) = get("B") {
            self.b = v;
        }
        if let Some(v) = get("tune") {
            self.tune = v;
        }
        if let Some(v) = get("tune_jobs") {
            self.tune_jobs = v;
        }
        if let Some(v) = get("frames_dir") {
            self.frames_dir = v;
        }
        if let Some(v) = get("sharp_tiles") {
            self.sharp_tiles = v;
        }
        if let Some(v) = get("center") {
            self.center = v;
        }
        if let Some(v) = get("cfa_drizzle_scale") {
            self.cfa_drizzle_scale = v;
        }
        self.drizzle = get_bool("drizzle", self.drizzle);
        self.native_1x = get_bool("native_1x", self.native_1x);
        self.aligned = get_bool("aligned", self.aligned);
        self.cache = get_bool("cache", self.cache);
        self.rej_z = get_bool("rej_z", self.rej_z);
        self.detail_snr = get_bool("detail_snr", self.detail_snr);
        self.two_pass = get_bool("two_pass", self.two_pass);
        self.keep_frames = get_bool("keep_frames", self.keep_frames);
        self.cfa_drizzle = get_bool("cfa_drizzle", self.cfa_drizzle);
        self.ln = get_bool("ln", self.ln);
        self.match_scale = get_bool("match_scale", self.match_scale);
    }

    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        let mut t = BTreeMap::new();
        let mut put_s = |k: &str, v: &str| {
            t.insert(k.to_string(), toml::Value::String(v.to_string()));
        };
        put_s("photos", &self.photos);
        put_s("out", &self.out);
        put_s("tag", &self.tag);
        put_s("A", &self.a);
        put_s("rej_k", &self.rej_k);
        put_s("n_star", &self.n_star);
        put_s("limit", &self.limit);
        put_s("crop", &self.crop);
        put_s("scale", &self.scale);
        put_s("pix", &self.pix);
        put_s("r_blur", &self.r_blur);
        put_s("bgdeg", &self.bgdeg);
        put_s("r_mode", &self.r_mode);
        put_s("sharp_w", &self.sharp_w);
        put_s("fwhm_max", &self.fwhm_max);
        put_s("xdrz", &self.xdrz);
        put_s("match", &self.match_path);
        put_s("flux_pow", &self.flux_pow);
        put_s("crop_cover", &self.crop_cover);
        put_s("bg_r", &self.bg_r);
        put_s("frame", &self.frame);
        put_s("gpu", &self.gpu);
        put_s("tune_a", &self.tune_a);
        put_s("cfa", &self.cfa);
        put_s("cfa_debayer", &self.cfa_debayer);
        put_s("cfa_weights", &self.cfa_weights);
        put_s("max_share", &self.max_share);
        put_s("ref_k", &self.ref_k);
        put_s("cfa_grid", &self.cfa_grid);
        put_s("cfa_align", &self.cfa_align);
        put_s("jobs", &self.jobs);
        put_s("cfa_ref", &self.cfa_ref);
        put_s("cfa_raw", &self.cfa_raw);
        put_s("rej_mode", &self.rej_mode);
        put_s("kernel", &self.kernel);
        put_s("psf_target", &self.psf_target);
        put_s("rgb_balance", &self.rgb_balance);
        put_s("env_win_mult", &self.env_win_mult);
        put_s("env_eps", &self.env_eps);
        put_s("env_p", &self.env_p);
        put_s("r_win", &self.r_win);
        put_s("r_k", &self.r_k);
        put_s("fwhm_search", &self.fwhm_search);
        put_s("star_grow_pct", &self.star_grow_pct);
        put_s("r_floor", &self.r_floor);
        put_s("c_star", &self.c_star);
        put_s("snr_clip", &self.snr_clip);
        put_s("tune", &self.tune);
        put_s("tune_jobs", &self.tune_jobs);
        put_s("frames_dir", &self.frames_dir);
        put_s("sharp_tiles", &self.sharp_tiles);
        put_s("center", &self.center);
        put_s("cfa_drizzle_scale", &self.cfa_drizzle_scale);
        put_s("pick", &self.pick);
        put_s("B", &self.b);
        for (k, v) in [
            ("drizzle", self.drizzle),
            ("native_1x", self.native_1x),
            ("aligned", self.aligned),
            ("cache", self.cache),
            ("rej_z", self.rej_z),
            ("detail_snr", self.detail_snr),
            ("two_pass", self.two_pass),
            ("keep_frames", self.keep_frames),
            ("cfa_drizzle", self.cfa_drizzle),
            ("ln", self.ln),
            ("match_scale", self.match_scale),
        ] {
            t.insert(k.to_string(), toml::Value::Boolean(v));
        }
        let mut out = String::from("# DWT · Drizzle settings (Python-compatible keys)\n");
        out.push_str(&toml::to_string(&t)?);
        Ok(out)
    }

    pub fn job_config(&self) -> Result<JobConfig, String> {
        let mut c = JobConfig::default();
        c.photos = self.photos.trim().to_string();
        c.out = self.out.trim().to_string();
        c.tag = self.tag.trim().to_string();
        c.a = f32v(&self.a, "清晰度 A")?;
        c.rej_k = f32v(&self.rej_k, "排异 κ")?;
        c.snr_clip = f32v(&self.snr_clip, "信噪比裁剪")?;
        c.detail_snr = if self.detail_snr { 1.0 } else { 0.0 };
        c.limit = usizev(&self.limit, 0);
        c.pick = self.pick.trim().to_string();
        c.tune = usizev(&self.tune, 0);
        c.tune_a = self.tune_a.clone();
        c.tune_jobs = usizev(&self.tune_jobs, 2);
        c.frames_dir = self.frames_dir.clone();
        c.keep_frames = self.keep_frames;
        c.crop = usizev(&self.crop, 0);
        c.sharp_tiles = usizev(&self.sharp_tiles, 0);
        c.center = self.center.clone();
        c.b = f32v(&self.b, "S 指数 B")?;
        c.flux_pow = f32v(&self.flux_pow, "吞吐权重")?;
        c.fwhm_search = usizev(&self.fwhm_search, 0);
        c.gpu = self.gpu.clone();
        c.frame = self.frame.clone();
        c.crop_cover = f32v(&self.crop_cover, "自动裁边")?;
        c.native_1x = self.native_1x;
        c.bgdeg = i32v(&self.bgdeg, 1);
        c.bg_r = f32v(&self.bg_r, "径向背景")?;
        c.env_win_mult = f32v(&self.env_win_mult, "包络窗")?;
        c.env_eps = f32v(&self.env_eps, "包络噪声项")?;
        c.env_p = f32v(&self.env_p, "包络形状幂")?;
        c.star_grow_pct = f32v(&self.star_grow_pct, "星点生长")?;
        c.aligned = self.aligned;
        c.cfa = self.cfa.clone();
        c.cfa_debayer = self.cfa_debayer.clone();
        c.cfa_weights = self.cfa_weights.clone();
        c.cfa_ref = self.cfa_ref.clone();
        c.cfa_raw = self.cfa_raw.clone();
        c.cfa_align = self.cfa_align.clone();
        c.rgb_balance = self.rgb_balance.clone();
        c.cfa_grid = self.cfa_grid.clone();
        c.ref_k = usizev(&self.ref_k, 0);
        c.cfa_drizzle = self.cfa_drizzle;
        c.cfa_drizzle_scale = f32v(&self.cfa_drizzle_scale, "彩机 drizzle 倍率")?;
        c.r_floor = f32v(&self.r_floor, "R 场下限")?;
        c.n_star = usizev(&self.n_star, 120);
        c.c_star = usizev(&self.c_star, 64);
        c.jobs = usizev(&self.jobs, 0);
        c.rej_mode = self.rej_mode.clone();
        c.rej_z = self.rej_z;
        c.max_share = f32v(&self.max_share, "单帧份额上限")?;
        c.kernel = self.kernel.clone();
        c.two_pass = self.two_pass;
        c.cache = self.cache;
        c.cache_dir = String::new();
        c.psf_target = f32v(&self.psf_target, "PSF 目标")?;
        c.rej_gate = 1.0 / 3.0;
        c.drizzle = self.drizzle;
        c.drizzle_scale = f32v(&self.scale, "输出倍率")?;
        c.drizzle_pix = f32v(&self.pix, "pixfrac")?;
        c.sharp_w = f32v(&self.sharp_w, "清晰度范围")?;
        c.fwhm_max = f32v(&self.fwhm_max, "FWHM 上限")?;
        c.r_blur = f32v(&self.r_blur, "R 场平滑")?;
        c.r_mode = self.r_mode.clone();
        c.r_win = f32v(&self.r_win, "R 带通窗宽")?;
        c.r_k = f32v(&self.r_k, "对比度阈值")?;
        c.ln = self.ln;
        c.xdrz = self.xdrz.clone();
        c.match_path = self.match_path.clone();
        c.match_scale = self.match_scale;
        Ok(c)
    }

    /// Last product stem, used by "查看权重场" before a run in this session.
    pub fn last_stem(&self) -> String {
        let tag = self.tag.trim();
        let pick = self.pick.trim();
        let mut stem = if tag.is_empty() { "PWS".to_string() } else { format!("PWS_{tag}") };
        if !pick.is_empty() {
            stem.push('_');
            stem.push_str(pick);
        }
        stem
    }

    pub fn open_dir(path: &Path) {
        let _ = std::process::Command::new("explorer").arg(path).spawn();
    }
}

fn f32v(s: &str, name: &str) -> Result<f32, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(0.0);
    }
    s.parse::<f32>().map_err(|_| format!("{name} 不是数字：{s}"))
}

fn usizev(s: &str, def: usize) -> usize {
    let s = s.trim();
    if s.is_empty() { def } else { s.parse::<usize>().unwrap_or(def) }
}

fn i32v(s: &str, def: i32) -> i32 {
    let s = s.trim();
    if s.is_empty() { def } else { s.parse::<i32>().unwrap_or(def) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_python_keys() {
        let p = Params::default();
        let text = p.to_toml().unwrap();
        for key in [
            "photos", "out", "tag", "A", "rej_k", "n_star", "limit", "crop", "drizzle",
            "scale", "pix", "r_blur", "bgdeg", "r_mode", "sharp_w", "fwhm_max", "xdrz",
            "match", "flux_pow", "crop_cover", "bg_r", "frame", "gpu", "tune_a", "native_1x",
            "aligned", "cfa", "cfa_debayer", "cfa_weights", "cache", "rej_z", "max_share",
            "ref_k", "cfa_grid", "cfa_align", "jobs", "cfa_ref", "cfa_raw", "rej_mode",
            "kernel", "psf_target", "detail_snr", "rgb_balance", "two_pass", "env_win_mult",
            "env_eps", "env_p", "r_win", "r_k", "fwhm_search", "star_grow_pct", "r_floor",
            "c_star", "snr_clip", "tune", "tune_jobs", "frames_dir", "keep_frames",
            "sharp_tiles", "center", "cfa_drizzle", "cfa_drizzle_scale", "ln", "match_scale",
            "pick", "B",
        ] {
            assert!(text.contains(&format!("{key} =")), "missing {key}");
        }
        let q = Params::from_toml(&text).unwrap();
        assert_eq!(q.a, p.a);
        assert_eq!(q.drizzle, p.drizzle);
        assert_eq!(q.photos, p.photos);
    }

    #[test]
    fn job_config_parses_defaults() {
        let c = Params::default().job_config().unwrap();
        assert_eq!(c.a, 20.0);
        assert_eq!(c.n_star, 120);
        assert!(c.drizzle);
        assert_eq!(c.r_mode, "max");
        // 与 Python 对齐的默认值（这几个以前是 off/8）
        assert_eq!(c.fwhm_search, 0);
        assert_eq!(c.cfa_raw, "auto");
        assert_eq!(c.gpu, "auto");
    }

    /// 面板上的控制项必须真的走到 `JobConfig`，不能只是存进 toml。
    #[test]
    fn job_config_passes_wired_parameters_through() {
        let mut p = Params::default();
        p.snr_clip = "5".into();
        p.rej_z = false;
        p.rej_mode = "mad".into();
        p.flux_pow = "2".into();
        p.sharp_w = "5".into();
        p.r_mode = "band".into();
        p.r_win = "21".into();
        p.r_k = "8".into();
        p.bg_r = "2".into();
        p.kernel = "psfmatch".into();
        p.psf_target = "60".into();
        p.two_pass = true;
        p.fwhm_max = "5".into();
        let c = p.job_config().unwrap();
        assert_eq!(c.snr_clip, 5.0);
        assert!(!c.rej_z);
        assert_eq!(c.rej_mode, "mad");
        assert_eq!(c.flux_pow, 2.0);
        assert_eq!(c.sharp_w, 5.0);
        assert_eq!(c.r_mode, "band");
        assert_eq!(c.r_win, 21.0);
        assert_eq!(c.r_k, 8.0);
        assert_eq!(c.bg_r, 2.0);
        assert_eq!(c.kernel, "psfmatch");
        assert_eq!(c.psf_target, 60.0);
        assert!(c.two_pass);
        assert_eq!(c.fwhm_max, 5.0);
    }
}
