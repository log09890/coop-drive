'''
Dashboard terminal model (no pygame, no CARLA): leveled, timestamped scrollback mirrored to a session log file,
command history, and "nice" chart axis ticks.
'''
import math
import os
import time

LEVELS = ('INFO', 'WARN', 'ERR', 'CMD', 'OUT', 'EVT', 'SYS')


class Terminal:
    def __init__(self, path=None, keep=3000):
        self.lines, self.keep = [], keep  # (wall 'HH:MM:SS.mmm', sim s, level, text)
        self.history, self.scroll = [], 0  # scroll: lines above the bottom
        self.path = path
        self.file = None
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self.file = open(path, 'a', encoding='utf-8', buffering=1)  # line buffered: a crash keeps the log

    def log(self, text, level='INFO', sim_t=0.0):
        assert level in LEVELS, level
        now = time.time()
        stamp = time.strftime('%H:%M:%S', time.localtime(now)) + f'.{int(now % 1 * 1000):03d}'
        for part in str(text).splitlines() or ['']:
            self.lines.append((stamp, sim_t, level, part))
            if self.file:
                self.file.write(f'{stamp} [{sim_t:9.2f}s] {level:4} {part}\n')
        del self.lines[:-self.keep]

    def command(self, line, sim_t=0.0):
        '''Record a typed command (history without repeats, newest last) and echo it.'''
        line = line.strip()
        if line:
            self.history = [h for h in self.history if h != line] + [line]
            self.log(line, 'CMD', sim_t)

    def view(self, rows):
        '''The `rows` lines to show for the current scroll position (scroll is clamped).'''
        self.scroll = max(0, min(self.scroll, len(self.lines) - rows))
        end = len(self.lines) - self.scroll
        return self.lines[max(0, end - rows):end]

    def clear(self):
        self.lines, self.scroll = [], 0

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def nice_ticks(lo, hi, n=4):
    '''Round axis ticks covering [lo, hi]: a step of 1, 2 or 5 x 10^k giving about n intervals.'''
    if not hi > lo:
        lo, hi = lo - 1, hi + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    first = math.floor(lo / step) * step
    return [round(first + k * step, 10) for k in range(int(math.ceil((hi - first) / step)) + 1)]


if __name__ == '__main__':  # self-check: python term.py
    import tempfile
    assert nice_ticks(0, 37) == [0, 10, 20, 30, 40], nice_ticks(0, 37)
    assert nice_ticks(-1, 1) == [-1, -0.5, 0, 0.5, 1], nice_ticks(-1, 1)
    assert nice_ticks(5, 5)[0] <= 4 and nice_ticks(5, 5)[-1] >= 6
    p = os.path.join(tempfile.mkdtemp(), 'logs', 's.log')
    t = Terminal(p, keep=5)
    t.command('/help'); t.command('/pause'); t.command('/help')
    assert t.history == ['/pause', '/help']
    t.log('a\nb', 'WARN', 1.5)
    for k in range(10):
        t.log(f'l{k}')
    assert len(t.lines) == 5 and t.lines[-1][3] == 'l9'
    t.scroll = 99
    assert [l[3] for l in t.view(3)] == ['l5', 'l6', 'l7'] and t.scroll == 2
    t.close()
    text = open(p, encoding='utf-8').read()
    assert 'CMD  /help' in text and 'WARN b' in text and text.count('\n') == 15, text
    print('ok')
