'''
Coop-Drive app settings (coopdrive.json next to this file) and the job queue the dashboard works through.
Pure python: no CARLA, no pygame.
'''
import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coopdrive.json')
DEFAULTS = {
    'town': 'Town02', 'vn': True, 'infra': True, 'share': True, 'mode': 'drive', 'driver': 'auto', 'model': '',
    'collect_steps': 40000, 'noise': 0.2, 'scenarios': True,
    'total_timesteps': 100000, 'episode_length': 3000, 'bc_reg': 0.5,
    'eval_name': 'eval', 'eval_episodes': 6, 'eval_steps': 2000, 'eval_expert': False, 'eval_seed': 0,
    'fullscreen': False, 'fps': 60, 'page': 'overview',
    'carla_root': 'E:/CARLA_0.9.10.1/WindowsNoEditor', 'carla_quality': 'Low',
    'job': {}, 'queue': [], 'queue_total': 0,
}
HELP = {
    'town': 'CARLA map', 'vn': 'chaotic Vietnamese traffic', 'infra': 'roadside units + V2X', 'share': 'use coop cars\' LiDAR',
    'mode': 'drive | collect | train | eval | bc', 'driver': 'auto | expert | ppo', 'model': 'policy .pth ("" = latest)',
    'collect_steps': 'expert steps to record', 'noise': 'collect steer noise σ', 'scenarios': 'collect: spawn peds / slow cars',
    'total_timesteps': 'PPO fine-tune length', 'bc_reg': 'BC anchor weight in the fine-tune (0 = off)', 'episode_length': 'max steps per training episode',
    'eval_name': 'results/<name>.json', 'eval_episodes': 'evaluation episodes', 'eval_steps': 'max steps per eval episode',
    'eval_expert': 'evaluate the waypoint expert', 'eval_seed': 'evaluation seed',
    'fullscreen': 'start fullscreen', 'fps': 'UI refresh rate', 'page': 'start page',
    'carla_root': 'CARLA folder (CarlaUE4.exe)', 'carla_quality': 'Low | Epic',
}
LIVE = {'driver', 'fullscreen', 'fps', 'page'}  # applied at once; the rest needs a restart
MODES = ('drive', 'collect', 'train', 'eval', 'bc')


def pipeline(town='Town02'):
    '''The full chain: expert data -> BC (shared LiDAR / ego + V2I only) -> evaluations -> PPO fine-tune -> evaluation.'''
    e = {'vn': True, 'infra': True}
    return [dict(mode='collect', fresh=True, **e), dict(mode='bc', noshare=False), dict(mode='bc', noshare=True),
            dict(mode='eval', eval_name='infra_expert', eval_expert=True, share=True, model='', **e),
            dict(mode='eval', eval_name='infra_bc_full', eval_expert=False, share=True, model=f'data/bc_policy_{town}.pth', **e),
            dict(mode='eval', eval_name='infra_bc_v2i', eval_expert=False, share=False,
                 model=f'preTrained_models/ppo_coop_noshare/{town}/bc_policy.pth', **e),
            dict(mode='train', share=True, model='', **e),
            dict(mode='eval', eval_name='infra_ppo_full', eval_expert=False, share=True, model='', **e)]


def defaults():
    return json.loads(json.dumps(DEFAULTS))  # deep copy: jobs and queues must never alias the defaults


def load(path=None):
    cfg, path = defaults(), path or PATH
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    return cfg


def save(cfg, path=None):
    path = path or PATH
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, path)  # atomic: a crash never leaves half a settings file


def coerce(key, text):
    '''Typed value for `set key text`, by the type of the default. Raises KeyError / ValueError.'''
    if key not in DEFAULTS or key in ('job', 'queue', 'queue_total'):
        raise KeyError(key)
    d = DEFAULTS[key]
    if isinstance(d, bool):
        t = text.lower()
        if t not in ('on', 'off', 'true', 'false', '1', '0', 'yes', 'no'):
            raise ValueError(f'{key} is on/off')
        return t in ('on', 'true', '1', 'yes')
    if isinstance(d, int):
        return int(float(text))
    if isinstance(d, float):
        return float(text)
    if key == 'mode' and text not in MODES:
        raise ValueError(f'mode is one of {" ".join(MODES)}')
    if key == 'driver' and text not in ('auto', 'expert', 'ppo'):
        raise ValueError('driver is auto, expert or ppo')
    return '' if text in ('""', "''", '-') else text


def start_job(cfg, job):
    '''Make `job` the current job: its keys override the settings for this run.'''
    cfg['job'] = dict(job)
    cfg.update({k: v for k, v in job.items() if k in DEFAULTS})


def update_job(cfg, **kv):
    '''Change the running job (e.g. collection started, training target fixed); the queued copy follows.'''
    head = bool(cfg['queue']) and cfg['queue'][0] == cfg['job']
    cfg['job'].update(kv)
    if head:
        cfg['queue'][0] = dict(cfg['job'])


def next_job(cfg):
    '''The current job is done: drop it from the queue and start the next one, or fall back to drive. Returns the new mode.'''
    if cfg['queue'] and cfg['queue'][0] == cfg['job']:
        cfg['queue'] = cfg['queue'][1:]
        if cfg['queue']:
            start_job(cfg, cfg['queue'][0])
            return cfg['mode']
    cfg['job'], cfg['mode'] = {}, 'drive'  # a manual job leaves a waiting queue alone (pipeline resume)
    return cfg['mode']


if __name__ == '__main__':  # self-check: python appcfg.py
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), 'c.json')
    c = load(p)
    c['job']['x'] = 1
    assert DEFAULTS['job'] == {} and defaults() == DEFAULTS
    c = load(p)
    assert c == DEFAULTS and coerce('vn', 'off') is False and coerce('fps', '30') == 30 and coerce('model', '-') == ''
    for bad in (('mode', 'fly'), ('vn', 'maybe'), ('fps', 'x')):
        try:
            coerce(*bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        coerce('queue', '[]'); raise AssertionError('queue')
    except KeyError:
        pass
    c['queue'] = pipeline()
    start_job(c, c['queue'][0])
    assert c['mode'] == 'collect' and c['job']['fresh']
    update_job(c, fresh=False)  # collection started: a restart resumes instead of starting over
    save(c, p)
    c = load(p)
    assert c['queue'][0]['fresh'] is False and c['mode'] == 'collect'
    modes = [next_job(c) for _ in range(len(c['queue']))]
    assert modes == ['bc', 'bc', 'eval', 'eval', 'eval', 'train', 'eval', 'drive'], modes
    assert c['queue'] == [] and c['job'] == {}
    c['job'] = {'mode': 'eval'}  # a manual job not in the queue leaves the queue alone
    c['queue'] = [{'mode': 'train'}]
    assert next_job(c) == 'drive' and c['queue'] == [{'mode': 'train'}]
    print('ok')
