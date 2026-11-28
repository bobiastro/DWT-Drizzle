"""wgpu 可移植 GPU 后端（branch B）。

为什么不是 CubeCL：CubeCL 是 Rust 库（`#[cube]` 宏 + pliron IR），没有 Python
绑定，接入要么用 PyO3 重新打包一份 Rust 扩展（Win/mac 各一份），要么改成 Rust
主程序。这里用 **wgpu-py + WGSL**：同一份着色器走 Vulkan / D3D12 / Metal，
`pip install wgpu` 就能用，不需要 CUDA Toolkit。

算法：**gather（收集）** 而不是 scatter。
  - 每个输出像素反解到源坐标，双线性取一个样本；
  - 每个输出像素只被一个线程写，天然不需要原子操作；
  - 每帧把 num/cov 累加一次，权值是该帧的标量权重。

与 scatter 的差别（有意为之）：
  - scatter 用 drop 面积铺开，边界与旋转会带来覆盖率不均（横纹来源之一）；
  - gather 每个输出像素每帧恰好一个样本，覆盖率恒等于 1（边缘除外），
    行方向不再有相位锁定的权重起伏。
  - 代价：没有逐像素排异（那需要保留每帧样本），饱和星的点扩散略软。

R 场处理：strip_weights 的权重是 W_i(p)=C_i^{A·R(p)}·S_i^{B·(1-R(p))}，指数随
像素变化，GPU 端传不了空间变化的指数。这里跑两遍——R=0（只看信噪比）与 R=1
（只看清晰度）——再按 R 线性混合。这是**近似**，与 CPU 的逐像素指数不同；
分支 B 的目的是验证可移植后端与速度，数值等价留给后续。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# 打包后启动脚本只把 pydeps/./core 放进 sys.path，gpudeps/ 不在里面。
# 这里自己找同级的 gpudeps（放在 pydeps 之前，保证用包里自带的那份 wgpu）。
_GPUD = Path(__file__).resolve().parent.parent / 'gpudeps'
if _GPUD.is_dir() and str(_GPUD) not in sys.path:
    sys.path.insert(0, str(_GPUD))

_WGSL = """
struct Params {
  a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
  w_src: u32, h_src: u32, w_out: u32, h_out: u32,
  wgt: f32, _p0: f32, _p1: f32, _p2: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> num: array<f32>;
@group(0) @binding(3) var<storage, read_write> cov: array<f32>;

fn crw(t: f32) -> vec4<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return vec4<f32>(
    -0.5 * t3 + t2 - 0.5 * t,
     1.5 * t3 - 2.5 * t2 + 1.0,
    -1.5 * t3 + 2.0 * t2 + 0.5 * t,
     0.5 * t3 - 0.5 * t2);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let J = gid.x;
  let I = gid.y;
  if (J >= P.w_out || I >= P.h_out) { return; }
  let X = f32(J) + 0.5;
  let Y = f32(I) + 0.5;
  let sx = P.a * X + P.b * Y + P.c;
  let sy = P.d * X + P.e * Y + P.f;
  let w1 = f32(P.w_src) - 1.0;
  let h1 = f32(P.h_src) - 1.0;
  if (sx < 0.0 || sy < 0.0 || sx > w1 || sy > h1) { return; }
  let bx = floor(sx) - 1.0;
  let by = floor(sy) - 1.0;
  let fx = sx - floor(sx);
  let fy = sy - floor(sy);
  let wx = crw(fx);
  let wy = crw(fy);
  var v = 0.0;
  for (var k: i32 = 0; k < 4; k = k + 1) {
    let yy = clamp(i32(by) + k, 0, i32(P.h_src) - 1);
    let w_y = wy[k];
    var row = 0.0;
    for (var m: i32 = 0; m < 4; m = m + 1) {
      let xx = clamp(i32(bx) + m, 0, i32(P.w_src) - 1);
      row = row + src[u32(yy) * P.w_src + u32(xx)] * wx[m];
    }
    v = v + row * w_y;
  }
  let idx = I * P.w_out + J;
  num[idx] = num[idx] + P.wgt * v;
  cov[idx] = cov[idx] + P.wgt;
}
"""

_cache: Dict[str, object] = {}


def _wp():
    import wgpu
    return wgpu


def _device():
    if 'dev' in _cache:
        return _cache['dev'], _cache['queue']
    wgpu = _wp()
    adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
    if adapter is None:
        raise RuntimeError('wgpu 找不到可用显卡')
    # 没有独显/集显时，Windows 的 WARP、Linux 的 lavapipe 这类**软件适配器**也会被枚举到。
    # 它们在 CPU 上跑模拟 GPU，比直接走我们的 numpy 分带还慢一个数量级，必须拒绝。
    try:
        _at = str((adapter.info or {}).get('adapter_type', '')).lower()
    except Exception:
        _at = ''
    if _at in ('cpu', 'software', 'virtual_cpu') or not _at:
        raise RuntimeError('只找到软件适配器（%s），没有可用的硬件 GPU' % (_at or '未知'))
    dev = adapter.request_device_sync()
    mod = dev.create_shader_module(code=_WGSL)
    pipe = dev.create_compute_pipeline(
        layout='auto', compute={'module': mod, 'entry_point': 'main'})
    _cache.update({'adapter': adapter, 'dev': dev, 'queue': dev.queue,
                   'module': mod, 'pipe': pipe})
    return dev, dev.queue


_last_reason = ''


def diagnose() -> Tuple[bool, str]:
    """返回 (是否可用, 原因)。原因用于日志与 `检测-wgpu.bat`。"""
    global _last_reason
    try:
        wgpu = _wp()
    except Exception as e:
        _last_reason = ('导入 wgpu 失败：%s: %s（gpudeps 目录：%s）'
                        % (type(e).__name__, e, _GPUD))
        return False, _last_reason
    try:
        ver = getattr(wgpu, '__version__', '?')
        native = ''
        try:
            from wgpu.backends.wgpu_native import _helpers  # noqa: F401
            import wgpu.backends.wgpu_native as _bn
            native = str(getattr(_bn, '__file__', ''))
        except Exception:
            pass
        try:
            dev, _q = _device()
            a = _cache['adapter'].info
            _last_reason = ('%s (%s, %s)，wgpu %s'
                            % (a.get('device'), a.get('backend_type'),
                               a.get('adapter_type'), ver))
            return True, _last_reason
        except Exception as e:
            backs = ''
            try:
                ad = wgpu.gpu.enumerate_adapters_sync()
                backs = '；枚举到 %d 个适配器：%s' % (
                    len(ad), ', '.join('%s/%s' % (
                        (x.info or {}).get('backend_type'), (x.info or {}).get('device'))
                        for x in ad))
            except Exception:
                backs = '；适配器枚举本身也失败'
            _last_reason = ('找不到可用适配器（wgpu %s%s）%s：%s: %s'
                            % (ver, ('，' + native) if native else '', backs,
                               type(e).__name__, e))
            return False, _last_reason
    except Exception as e:                                  # 兜底
        _last_reason = '未知错误：%s: %s' % (type(e).__name__, e)
        return False, _last_reason


def available() -> bool:
    try:
        ok, _msg = diagnose()
        return bool(ok)
    except Exception:
        return False


def reason() -> str:
    return _last_reason or '未知（未调用 diagnose）'


def info() -> str:
    ok, msg = diagnose()
    if not ok:
        raise RuntimeError(msg)
    return msg


def affine_from(dx: float, dy: float, theta: float, k: float,
                plan: Dict) -> np.ndarray:
    """输出画布坐标 (X,Y) → 源像素索引 (u,v) 的仿射系数。

    与 `_xy_rows` 同一约定：源像素索引 u 的中心在画布上映射为
        X = s·(k·(ct·(u+0.5) - st·(v+0.5)) + ux) + ox ，ux = -dx 。
    反解得 u = a·X + b·Y + c，v = d·X + e·Y + f。
    """
    s = float(plan['s'])
    ox = float(plan['ox'])
    oy = float(plan['oy'])
    th = math.radians(float(theta or 0.0))
    ct, st = math.cos(th), math.sin(th)
    kk = float(k) if (k and np.isfinite(k)) else 1.0
    ux, uy = -float(dx), -float(dy)
    ks = kk * s
    a, b = ct / ks, st / ks
    d, e = -st / ks, ct / ks
    c = -(ct / ks) * ox - (st / ks) * oy - (ct * ux + st * uy) / kk - 0.5
    f = (st / ks) * ox - (ct / ks) * oy + (st * ux - ct * uy) / kk - 0.5
    return np.array([a, b, c, d, e, f], np.float64)


def _pack(ws, hs, W, H, wgt):
    p = np.zeros(16, np.float32)
    p[6:10] = np.array([ws, hs, W, H], np.uint32).view(np.float32)
    p[10] = np.float32(wgt)
    return p


def _gather_into(num, cov, src_buf, uni, bg, frames, affines, out_shape, weights,
                 dev, queue):
    H, W = int(out_shape[0]), int(out_shape[1])
    gx, gy = (W + 7) // 8, (H + 7) // 8
    for i in range(len(affines)):
        wgt = float(weights[i])
        if not np.isfinite(wgt) or wgt <= 0:
            continue
        p = _pack(frames.shape[2], frames.shape[1], W, H, wgt)
        p[:6] = np.asarray(affines[i], np.float32)
        queue.write_buffer(uni, 0, p.tobytes())
        queue.write_buffer(src_buf, 0,
                           np.ascontiguousarray(frames[i], np.float32).tobytes())
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(_cache['pipe'])
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(gx, gy)
        cp.end()
        queue.submit([enc.finish()])
    queue.submit([])


def _alloc(dev, U, shape):
    H, W = int(shape[0]), int(shape[1])
    sz = H * W * 4
    num = dev.create_buffer(size=sz, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
    cov = dev.create_buffer(size=sz, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
    return num, cov, sz


def stack_gather(frames, affines: np.ndarray, out_shape: Tuple[int, int],
                 weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """收集式叠加：返回 (num, cov)，都是 (H, W) float32。"""
    wgpu = _wp()
    dev, queue = _device()
    U = wgpu.BufferUsage
    frames = np.ascontiguousarray(frames, np.float32)
    n, hs, ws = frames.shape
    H, W = int(out_shape[0]), int(out_shape[1])
    num, cov, sz = _alloc(dev, U, (H, W))
    queue.write_buffer(num, 0, np.zeros(H * W, np.float32).tobytes())
    queue.write_buffer(cov, 0, np.zeros(H * W, np.float32).tobytes())
    src_buf = dev.create_buffer(size=max(16, hs * ws * 4), usage=U.STORAGE | U.COPY_DST)
    uni = dev.create_buffer(size=64, usage=U.UNIFORM | U.COPY_DST)
    bg = dev.create_bind_group(layout=_cache['pipe'].get_bind_group_layout(0), entries=[
        {'binding': 0, 'resource': {'buffer': uni, 'offset': 0, 'size': 64}},
        {'binding': 1, 'resource': {'buffer': src_buf, 'offset': 0,
                                    'size': max(16, hs * ws * 4)}},
        {'binding': 2, 'resource': {'buffer': num, 'offset': 0, 'size': sz}},
        {'binding': 3, 'resource': {'buffer': cov, 'offset': 0, 'size': sz}},
    ])
    w = np.ones(n, np.float64) if weights is None else np.asarray(weights, np.float64)
    _gather_into(num, cov, src_buf, uni, bg, frames, affines, (H, W), w, dev, queue)
    out = np.empty((2, H * W), np.float32)
    for j, buf in enumerate((num, cov)):
        tmp = dev.create_buffer(size=sz, usage=U.COPY_DST | U.MAP_READ)
        enc = dev.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, tmp, 0, sz)
        queue.submit([enc.finish()])
        tmp.map_sync(wgpu.MapMode.READ)
        out[j] = np.frombuffer(tmp.read_mapped(), dtype=np.float32).copy()
        tmp.unmap()
    return out[0].reshape(H, W), out[1].reshape(H, W)


def stack_plane(frames, affines, out_shape, weights=None):
    """单通道便捷封装：返回（覆盖率归一化后的图, 覆盖计数）。"""
    num, cov = stack_gather(frames, affines, out_shape, weights)
    return np.where(cov > 0, num / np.maximum(cov, 1e-20), 0.0).astype(np.float32), cov




def accumulate_stream(get_frame, idx, affines, out_shape, weights, dev, queue):
    """流式收集：逐帧取图上传，GPU 端持续累加，返回 (num, cov) 主机数组。"""
    wgpu = _wp()
    U = wgpu.BufferUsage
    H, W = int(out_shape[0]), int(out_shape[1])
    num, cov, sz = _alloc(dev, U, (H, W))
    queue.write_buffer(num, 0, np.zeros(H * W, np.float32).tobytes())
    queue.write_buffer(cov, 0, np.zeros(H * W, np.float32).tobytes())
    first = np.ascontiguousarray(get_frame(idx[0]), np.float32)
    hs, ws = first.shape
    src_buf = dev.create_buffer(size=max(16, hs * ws * 4), usage=U.STORAGE | U.COPY_DST)
    uni = dev.create_buffer(size=64, usage=U.UNIFORM | U.COPY_DST)
    bg = dev.create_bind_group(layout=_cache['pipe'].get_bind_group_layout(0), entries=[
        {'binding': 0, 'resource': {'buffer': uni, 'offset': 0, 'size': 64}},
        {'binding': 1, 'resource': {'buffer': src_buf, 'offset': 0,
                                    'size': max(16, hs * ws * 4)}},
        {'binding': 2, 'resource': {'buffer': num, 'offset': 0, 'size': sz}},
        {'binding': 3, 'resource': {'buffer': cov, 'offset': 0, 'size': sz}},
    ])
    gx, gy = (W + 7) // 8, (H + 7) // 8
    for j, i in enumerate(idx):
        w = float(weights[j])
        if not np.isfinite(w) or w <= 0:
            continue
        fr = first if j == 0 else np.ascontiguousarray(get_frame(i), np.float32)
        p = _pack(ws, hs, W, H, w)
        p[:6] = np.asarray(affines[j], np.float32)
        queue.write_buffer(uni, 0, p.tobytes())
        queue.write_buffer(src_buf, 0, fr.tobytes())
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(_cache['pipe'])
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(gx, gy)
        cp.end()
        queue.submit([enc.finish()])
    queue.submit([])
    out = np.empty((2, H * W), np.float32)
    for j, buf in enumerate((num, cov)):
        tmp = dev.create_buffer(size=sz, usage=U.COPY_DST | U.MAP_READ)
        enc = dev.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, tmp, 0, sz)
        queue.submit([enc.finish()])
        tmp.map_sync(wgpu.MapMode.READ)
        out[j] = np.frombuffer(tmp.read_mapped(), dtype=np.float32).copy()
        tmp.unmap()
    return out[0].reshape(H, W), out[1].reshape(H, W)


def stack_pws(fs, C, S, dx, dy, th, kk, plan, A, B, sky=0.0, R=None,
              log=None, n=None):
    """分支 B：用 gather 后端跑一遍 PWS 的加权叠加（无逐像素排异）。

    权重按两档跑：R=0 只看信噪比（S^B），R=1 再看清晰度（C^A·S^B），
    最后按 R 场线性混合——见模块头的说明。
    """
    wgpu = _wp()
    dev, queue = _device()
    H, W = int(plan['H']), int(plan['W'])
    nn = int(n)
    idx = [i for i in range(nn) if np.isfinite(dx[i]) and np.isfinite(dy[i])]
    if not idx:
        raise RuntimeError('没有可用帧')
    aff = [affine_from(dx[i], dy[i], th[i], kk[i], plan) for i in idx]
    s_w = np.asarray([S[i] for i in idx], np.float64)
    c_w = np.asarray([max(C[i], 1e-12) for i in idx], np.float64)
    w0 = np.power(np.maximum(s_w, 1e-12), B)
    w1 = np.power(np.maximum(c_w, 1e-12), A) * w0
    w0 = np.clip(w0 / max(w0.mean(), 1e-30), 1e-6, 1e6)
    w1 = np.clip(w1 / max(w1.mean(), 1e-30), 1e-6, 1e6)
    get = lambda i: np.asarray(fs.frame(int(i)), np.float32)
    if log:
        log('[wgpu] 设备 %s；本组 %d 帧走 wgpu' % (info(), len(idx)))
        log('[wgpu] 收集式叠加：%d 帧，画布 %d×%d，两遍权重（R=0 / R=1）' % (len(idx), W, H))
    n0, c0 = accumulate_stream(get, idx, aff, (H, W), w0, dev, queue)
    n1, c1 = accumulate_stream(get, idx, aff, (H, W), w1, dev, queue)
    d0 = np.where(c0 > 0, n0 / np.maximum(c0, 1e-20), 0.0)
    d1 = np.where(c1 > 0, n1 / np.maximum(c1, 1e-20), 0.0)
    if R is None:
        stk = d0
    else:
        rr = np.clip(np.asarray(R, np.float32), 0.0, 1.0)
        if rr.shape != (H, W):
            yy = np.clip(np.arange(H) * 0 + 0, 0, 0)  # 占位，调用方保证形状
        stk = (1.0 - rr) * d0 + rr * d1
    stk = (stk - float(sky)).astype(np.float32)
    cov = c0.astype(np.float32)
    neff = cov.copy()
    return {'stack': stk, 'neff': neff, 'coverage': cov, 'ncov': (cov > 0).astype(np.float32),
            'rej_rate': 0.0, 'wmax': float(max(w0.max(), w1.max()))}


if __name__ == '__main__':
    import time
    print('wgpu available:', available())
    if available():
        print('device:', info())
        rng = np.random.default_rng(0)
        n, hs, ws = 4, 1200, 1800
        H, W = 2500, 3600
        frames = rng.random((n, hs, ws), dtype=np.float32)
        aff = np.stack([affine_from(3.0 * i, -2.0 * i, 0.7 * i, 1.0,
                                    {'s': 2.0, 'ox': 10.0, 'oy': 20.0})
                        for i in range(n)])
        t = time.time()
        img, cov = stack_plane(frames, aff, (H, W))
        print('gather %.2fs  cov 中位 %.1f  空像素 %.2f%%' %
              (time.time() - t, np.median(cov), 100 * np.mean(cov == 0)))


def has_f32_atomic() -> bool:
    """适配器是否支持 shader-float32-atomic（真散射累加的前提）。

    wgpu 通过原生扩展把它开出来（Vulkan 的 VK_EXT_shader_atomic_float、Metal 3），
    WebGPU 规范本身还没有；拿不到时调用方应回退到分桶后端。
    """
    try:
        wgpu = _wp()
        if 'f32a' in _cache:
            return bool(_cache['f32a'])
        adapter = _cache.get('adapter')
        if adapter is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
            if adapter is None:
                _cache['f32a'] = False
                return False
            _cache['adapter'] = adapter
        ok = any(str(f) == 'shader-float32-atomic' for f in adapter.features)
        _cache['f32a'] = ok
        return ok
    except Exception:
        return False


def _device_f32():
    """带 shader-float32-atomic 的设备；不支持时抛 RuntimeError。"""
    if 'devf' in _cache:
        return _cache['devf'], _cache['queuef']
    if not has_f32_atomic():
        raise RuntimeError('适配器不支持 shader-float32-atomic')
    adapter = _cache.get('adapter')
    # 全幅画布常驻时单个缓冲可达 400+ MB，WebGPU 默认上限只有 128/256 MiB，
    # 必须显式申请适配器支持的上限，否则 create_bind_group 会报 Validation Error。
    _lim = {}
    try:
        _al = adapter.limits
        for _k in ('max-storage-buffer-binding-size', 'max_buffer_size'):
            _v = _al.get(_k) if isinstance(_al, dict) else getattr(
                _al, _k.replace('-', '_'), None)
            if _v:
                _lim[_k] = int(_v)
    except Exception:
        pass
    dev = adapter.request_device_sync(
        required_features=['shader-float32-atomic'],
        required_limits=_lim or None)
    import os as _os
    if _os.environ.get('DWT_WGPU_DEBUG') == '1':
        try:
            dev.on_uncaptured_error(lambda e: print('[wgpu-err] %s: %s'
                                                    % (getattr(e, 'reason', '?'), e), flush=True))
        except Exception:
            pass
    _cache.update({'devf': dev, 'queuef': dev.queue})
    return dev, dev.queue
