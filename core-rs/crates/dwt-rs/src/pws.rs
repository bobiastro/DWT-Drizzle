//! 从 `core/pws.py` / `core/drizzle.py` 迁过来的「数值参数」原语。
//!
//! 这些函数此前在 Rust 侧只有字段、没有算法（`WIRING.md` 里的缺口表），
//! 现在集中在这里，由 CLI（`crates/dwt-rs/src/main.rs`）与 GUI 后端
//! （`crates/dwt-backend/src/job.rs`）共同调用，保证两条路径数值一致。
//! 每个函数的文档都注明它对应的 Python 行号：语义以 Python 为准。

use crate::image::Img;

/// 噪声权重 `S_i = median(σ)/σ_i`，再按 `snr_clip` 限幅。
///
/// Python `core/pws.py:3305-3320`：
/// ```text
/// _k_clip = float(getattr(p, 'snr_clip', 3.0) or 0.0)
/// if _k_clip > 1.0: S = np.clip(S, 1.0/_k_clip, _k_clip)
/// ```
/// `snr_clip <= 1.0` 视为关闭限幅（Python 的 `if` 护栏）。
pub fn snr_weights(sigmas: &[f32], sig_med: f32, snr_clip: f32) -> Vec<f32> {
    let k = snr_clip;
    sigmas
        .iter()
        .map(|v| {
            let s = sig_med / (*v).max(1e-30);
            if k > 1.0 {
                s.clamp(1.0 / k, k)
            } else {
                s
            }
        })
        .collect()
}

/// 未使用参数占位（随各参数接线逐步填充，保持模块可增量编译）。
pub(crate) fn _unused(_img: &Img) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snr_weights_clip_matches_python() {
        // σ 中位 2.0：σ=1 → S=2（未触限），σ=0.5 → 4 → 被 3 截断，
        // σ=8 → 0.25 → 被 1/3 截断。
        let s = snr_weights(&[1.0, 0.5, 8.0], 2.0, 3.0);
        assert!((s[0] - 2.0).abs() < 1e-6);
        assert!((s[1] - 3.0).abs() < 1e-6);
        assert!((s[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn snr_weights_clip_is_wired() {
        // 非默认 snr_clip 必须改变限幅结果，且 snr_clip<=1 时不限幅。
        let a = snr_weights(&[0.5, 8.0], 2.0, 3.0);
        let b = snr_weights(&[0.5, 8.0], 2.0, 5.0);
        assert_ne!(a, b, "snr_clip 未接线：5.0 与 3.0 结果相同");
        assert!((b[0] - 4.0).abs() < 1e-6);
        let off = snr_weights(&[0.5], 2.0, 1.0);
        assert!((off[0] - 4.0).abs() < 1e-6);
    }
}
