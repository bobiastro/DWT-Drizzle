"""端到端几何校验：解析星点（不做插值）→ 测位移 → drizzle ×s → 量 FWHM

若配准正确，输出 FWHM 应等于 2·2.3548·sqrt(σ² + p²/12)（p=1 的盒式模糊）。
配准错 1 像素，输出会显著变胖——这正是真机上"成品比单帧还糊"的嫌疑点。
"""
import sys
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'core'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import drizzle as D


class FakeFS:
    """最小 FrameSet 替身：解析渲染的星点，位移是纯平移，无插值。"""

    def __init__(self, imgs, sky=0.0):
        self.imgs = imgs
        self.shape = imgs[0].shape
        self.sky0 = sky

    def __len__(self):
        return len(self.imgs)

    def frame(self, i):
        return self.imgs[i]

    def strip(self, i, j0, j1):
        return self.imgs[i][j0:j1]


def render(n, stars, sigma):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    img = np.zeros((n, n), np.float32)
    for (cx, cy, amp) in stars:
        img += amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))
    return img


def fwhm_at(img, cx, cy, half=9):
    """围绕 (cx,cy) 取窗，用二阶矩估 FWHM（对平滑高斯的对照够用）。"""
    x0, y0 = int(round(cx)), int(round(cy))
    sub = img[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1].astype(np.float64)
    sub = np.maximum(sub - np.median(sub), 0)
    if sub.sum() <= 0:
        return float('nan')
    yy, xx = np.mgrid[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    mx = (sub * xx).sum() / sub.sum()
    my = (sub * yy).sum() / sub.sum()
    vx = (sub * (xx - mx) ** 2).sum() / sub.sum()
    vy = (sub * (yy - my) ** 2).sum() / sub.sum()
    sig = np.sqrt(0.5 * (vx + vy))
    return float(2.3548 * sig), float(mx), float(my)


def run(s_true, s_drz=2.0, p=1.0, sigma=1.1, n=160):
    base = [(60.25, 70.37, 1000.0), (100.6, 110.2, 400.0), (40.1, 120.9, 700.0)]
    offs = [(0.0, 0.0), (0.37, 0.62), (1.71, -2.34), (3.0, 3.0)]
    imgs = [render(n, [(x + ox, y + oy, a) for (x, y, a) in base], sigma) for (ox, oy) in offs]
    fs = FakeFS(imgs, sky=0.0)
    stars = [(int(round(x)), int(round(y))) for (x, y, a) in base]
    sh = D.measure_shifts(fs, stars, ref=0)
    dx, dy = np.asarray(sh['dx']), np.asarray(sh['dy'])
    print('measured dx', np.round(dx, 3))
    print('measured dy', np.round(dy, 3))
    print('true     dx', [o[0] for o in offs], ' dy', [o[1] for o in offs])
    plan = D.plan_output(fs.shape, dx, dy, s_drz)
    R = np.zeros((plan['H'], plan['W']), np.float32)
    res = D.stack_drizzle(fs, R, np.ones(len(fs), np.float32), np.ones(len(fs), np.float32),
                          dx, dy, plan, A=0.0, B=0.0, rej_k=0.0, rej_gate=1.0,
                          sky=0.0, pixfrac=p, band_h=64)
    Stk = res['stack']
    s = plan['s']
    for (x, y, a) in base:
        cx = x * s + plan['ox']
        cy = y * s + plan['oy']
        f, mx, my = fwhm_at(Stk, cx, cy)
        print(f'star ({x:.1f},{y:.1f}) -> out ({mx:.2f},{my:.2f}) expect ({cx:.2f},{cy:.2f}) '
              f'FWHM {f:.3f}')
    pred = s * 2.3548 * np.sqrt(sigma ** 2 + p ** 2 / 12.0)
    print(f'predicted FWHM at s={s:g}, p={p:g}: {pred:.3f} px')
    return Stk, plan


if __name__ == '__main__':
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 2.0)
