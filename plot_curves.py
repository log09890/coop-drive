'''
Learning curves of a PPO fine-tune as PNG (pygame only, no matplotlib).

  python plot_curves.py logs/train_<run>.csv [out.png]

Panels: episode return (per update, with running mean), value loss, approx. KL, clip fraction, BC anchor loss,
mean step reward. Values are the per-update rows the dashboard writes during `mode train`.
'''
import csv
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import pygame  # noqa: E402

PANELS = [('batch_return', 'avg_return', 'Return per episode'), ('value_loss', None, 'Value loss'),
          ('approx_kl', None, 'Approx. KL'), ('clip_frac', None, 'Clip fraction'),
          ('bc_loss', None, 'BC anchor loss'), ('mean_reward', None, 'Mean step reward')]
BG, FG, DIM, GRID, A, B = (255, 255, 255), (30, 34, 40), (110, 118, 128), (225, 229, 233), (11, 127, 163), (199, 119, 0)


def nice(lo, hi, n=4):
    import math
    if not hi > lo:
        lo, hi = lo - 1, hi + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    first = math.floor(lo / step) * step
    return [first + k * step for k in range(int(math.ceil((hi - first) / step)) + 1)]


def plot(rows, out, w=1500, h=900):
    pygame.init()
    s = pygame.Surface((w, h))
    s.fill(BG)
    font, small = pygame.font.SysFont('arial', 22, bold=True), pygame.font.SysFont('arial', 16)
    x = [float(r['timestep']) / 1000 for r in rows]
    n0 = sum0 = 0.0
    for r in rows:  # mean of the episodes of this update only, from the cumulative mean (last_return is one noisy episode)
        n = float(r.get('episode') or 0)
        r['batch_return'] = (float(r['avg_return']) * n - sum0) / (n - n0) if n > n0 else r['last_return']
        n0, sum0 = n, float(r['avg_return']) * n
    pw, ph = w // 3, h // 2
    for k, (key, key2, title) in enumerate(PANELS):
        ox, oy = (k % 3) * pw, (k // 3) * ph
        L, R, T, Bm = ox + 70, ox + pw - 20, oy + 40, oy + ph - 50
        ys = [float(r[key]) for r in rows]
        ys2 = [float(r[key2]) for r in rows] if key2 else []
        ticks = nice(min(ys + ys2), max(ys + ys2))
        y0, y1 = ticks[0], ticks[-1]
        X = lambda v: L + (R - L) * (v - x[0]) / max(1e-9, x[-1] - x[0])
        Y = lambda v: Bm - (Bm - T) * (v - y0) / max(1e-9, y1 - y0)
        for t in ticks:
            pygame.draw.line(s, GRID, (L, Y(t)), (R, Y(t)))
            lab = small.render(f'{t:g}', True, DIM)
            s.blit(lab, (L - 8 - lab.get_width(), Y(t) - 9))
        for t in nice(x[0], x[-1], 5):
            if x[0] <= t <= x[-1]:
                lab = small.render(f'{t:g}k', True, DIM)
                s.blit(lab, (X(t) - lab.get_width() // 2, Bm + 6))
        pygame.draw.rect(s, DIM, (L, T, R - L, Bm - T), 1)
        if len(x) > 1:
            pygame.draw.lines(s, A, False, [(X(a), Y(b)) for a, b in zip(x, ys)], 2)
            if ys2:
                pygame.draw.lines(s, B, False, [(X(a), Y(b)) for a, b in zip(x, ys2)], 3)
        s.blit(font.render(title, True, FG), (L, oy + 8))
        if key2:
            s.blit(small.render('per update', True, A), (R - 200, oy + 14))
            s.blit(small.render('running mean', True, B), (R - 100, oy + 14))
        s.blit(small.render('timestep', True, DIM), (R - 60, Bm + 26))
    pygame.image.save(s, out)


def main():
    path = sys.argv[1]
    rows = list(csv.DictReader(open(path)))
    assert rows, f'{path}: no PPO update yet'
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(path)[0] + '.png'
    plot(rows, out)
    print('wrote', out, f'({len(rows)} updates)')


if __name__ == '__main__':
    if len(sys.argv) == 1:  # self-check with synthetic rows
        import tempfile
        rows = [dict(timestep=str(1000 * k), last_return=str(k % 7 - 3), avg_return=str(k / 10), value_loss=str(1 / (k + 1)),
                     approx_kl=str(0.01), clip_frac=str(0.1), bc_loss=str(0.02), mean_reward=str(0.3)) for k in range(30)]
        p = os.path.join(tempfile.mkdtemp(), 'c.png')
        plot(rows, p)
        assert os.path.getsize(p) > 10000
        print('ok')
    else:
        main()
