# -*- coding: utf-8 -*-
"""wgpu 后端的最小回归（分支 C）。

没有可用显卡时跳过（打印 SKIP），不阻塞其它平台的测试。
检查两件事：
  1. 冲激响应在整数对齐下应回到 1.0（Catmull-Rom 双三次的相位正确）；
  2. 平移 +0.5 像素后总量守恒（归一化没写坏）。
"""
import sys
import pathlib

import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'core'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main():
    import gpu_wgpu as g
    if not g.available():
        print('WGPU-SKIP（没有可用的 Vulkan / Metal / D3D12 设备）')
        return
    a = np.zeros((64, 64), np.float32)
    a[32, 32] = 1.0
    aff = np.stack([g.affine_from(0, 0, 0, 1.0, {'s': 1.0, 'ox': 0.0, 'oy': 0.0})])
    out, cov = g.stack_plane(a[None], aff, (64, 64))
    pk = float(out.max())
    tot = float(out.sum())
    assert abs(pk - 1.0) < 1e-3, '冲激峰值 %r（相位错了？）' % pk
    assert abs(tot - 1.0) < 1e-3, '冲激总能量 %r' % tot
    b = np.zeros((64, 64), np.float32)
    b[20:44, 20:44] = 1.0
    aff2 = np.stack([g.affine_from(0.5, -0.5, 0.0, 1.0, {'s': 1.0, 'ox': 0.0, 'oy': 0.0})])
    out2, _ = g.stack_plane(b[None], aff2, (64, 64))
    keep = float(out2.sum()) / float(b.sum())
    assert abs(keep - 1.0) < 2e-2, '平移后能量比 %r' % keep
    print('WGPU-OK  peak=%.4f  sum=%.4f  平移能量比=%.4f' % (pk, tot, keep))


if __name__ == '__main__':
    main()
