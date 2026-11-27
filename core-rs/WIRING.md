# 参数接线表（branch-E-rust）

每条：`- <参数> <commit> <A/B 结论>`。验收命令（基线 HEAD 9f6c3c9，改动前 md5 `a857a193b8bd0311c640549efb5826e4`）：

```
DWT_RS_ALLOW_CROP_REG=1 dwt-rs --photos G:/HFG1-3/master3/calibrated --out OUT --tag T \
  --pick O --limit 8 --crop 400 --drizzle --drizzle-scale 1
```
（`--crop 400` 与配准同时出现时要放开 crop+配准保护，否则 HEAD 自己就会拒绝运行。）

- snr_clip 93dfa83（本地）/ 615904b（branch-E-rust） S 限幅独立接线：S∈[0.907,1.107]（本数据集），故 `--snr-clip 5`/`2.0` 与默认 3.0 **完全一致**（md5 a857a19…）；`--snr-clip 1.05` 才真正触发限幅 → md5 `f7aecf9d465bbcf2527ffe2cc009d3ff`，FWHM 3.3285（默认 3.3310），σ 7.113e-5（默认 7.129e-5）
- rej_z PENDING Python 口径：`rej_z` 只在非 drizzle 的 `stack_strip` 生效（pws.py:1855/4834/4867），drizzle 路径不传 `frame_sig`。A/B：drizzle 默认 vs `--rej-z 0` 成品**相同**（md5 都是 `ae979a21cc7a6ca187ffd3c2b1fdab37`，剔除率 1.638%）；`--aligned 1 --drizzle 0` 下 rej_z 1→0 成品改变（`51a8304b…` → `045567c9…`，剔除率 2.063% → 2.111%）。附带修正：HEAD 的 drizzle 分支多除了一个 σ_i，与 Python 不符；改回后默认 md5 由 `a857a193…` 变为 `ae979a21…`。
- cli-defaults PENDING `--fwhm-search` 8→0、`--cfa-raw` off→auto、`--gpu` off→auto（对齐 Python 默认）。A/B：默认成品 md5 由 `ae979a21…`（= 旧默认等价，见 rej_z 行）变为 `1abe99c0…`，FWHM 3.3346→3.3335，σ 7.1402e-5→7.4343e-5；用 `--fwhm-search 8` 可逐字节复现 `ae979a21…`，说明差异只来自该默认值。cfa-raw/gpu 在单色路径为空操作。
- sharp_w PENDING `--sharp-w` 进权重指数（`e += sharp_w*ln C_i`，core/drizzle.py:1990-2010）。A/B：默认 0 与 d3 一致（md5 `1abe99c0…`）；`--sharp-w 5` → md5 `9a811a2a…`，FWHM 3.3335→3.3736，σ 7.4343e-5→7.5341e-5；`--sharp-w -5` → `a9587d00…`，FWHM 3.3936。单测：`sharp_w_zero_matches_plain_strip_weights`、`sharp_w_adds_log_c_to_the_exponent`。
- rej_mode PENDING `--rej-mode winsor|mad|fast|esd`：照 drizzle.py:1668-1800 的三个分块尺度估计（`_winsor_scale`/`_block_mad`/`_fast_scale`）+ `_esd_bad`(n_iter=3, α=0.05, λ 由 Rosner t 分布算)，REJ_BLOCK=64。A/B（drizzle 路径）：winsor 默认 md5 `1abe99c0…` 剔除率 1.638%；mad → `45665d93…` 0.693% FWHM 3.3736；fast → `6ba52098…` 0.458% FWHM 3.3919；esd → `90c1587b…` 1.082% FWHM 3.4580。单测：`winsor_matches_legacy_reject`、`other_modes_change_the_rejection_scale`。
