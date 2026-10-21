'''
Coop-Drive research console: live multi-view dashboard for the coop PPO agent (needs COOP_SECTORS > 0 and a running
CARLA server). It can also record expert data or fine-tune PPO while you watch.

  python dashboard.py --town Town02 --vn --infra            drive (PPO if a matching model loads, else the expert)
  python dashboard.py --town Town02 --vn --infra --collect 40000   record expert data for bc.py train
  python dashboard.py --town Town02 --vn --infra --train     fine-tune PPO from the latest checkpoint

F1-F5 switch pages. Type commands straight into the terminal (help lists them); the session log is saved in logs/.
'''
import argparse
import collections
import math
import os
import pickle
import random
import subprocess
import sys
import threading
import time
import weakref
import numpy as np
import pygame
import torch
from simulation.connection import ClientConnection, carla
from simulation.environment import CarlaEnvironment
from simulation.coop import pose_matrix, to_ego
from simulation.scenarios import spawn_scenario
from simulation.vn_traffic import MOTORBIKES, BICYCLES, EVENTS
from simulation.settings import SIGNAL_DIM, INFRA_DIM, TM_PORT, COOP_SECTORS, COOP_CORRIDOR, COOP_VEHICLES, COOP_OBS, LIDAR_CHANNELS, LIDAR_POINTS_PER_SECOND
from encoder_init import EncodeState
from autoencoder.decoder import Decoder
from networks.on_policy.ppo.agent import PPOAgent
from networks.on_policy.ppo.ppo import ActorCritic
from parameters import LATENT_DIM
from bc import expert_action
from insight import closest_approach, input_sensitivity, group_sensitivity, SessionMetrics
from term import Terminal, nice_ticks
import appcfg
import evaluate

RESTART, SERVER_RESTART = 75, 76  # exit codes the launcher (app.py) acts on

W, H, HEAD, TABS, BOTTOM = 1600, 900, 30, 24, 236  # canvas; the window is scaled to fit the screen
TOP = HEAD + TABS
BIG, BEV = (800, 450), (400, 400)  # chase camera, top-down camera (40 m up, 70 deg fov -> 56 m wide)
BEV_M = 2 * 40 * math.tan(math.radians(35))  # metres across the BEV image
BG, PANEL, EDGE, HEADC, GRID = (8, 10, 14), (14, 17, 23), (40, 48, 62), (21, 26, 35), (30, 36, 48)
FG, DIM = (212, 220, 230), (108, 120, 136)
RED, ORANGE, YELLOW, GREEN, CYAN, MAG = (255, 72, 72), (255, 146, 40), (240, 206, 60), (70, 220, 120), (0, 186, 255), (210, 120, 255)
LEVEL_COLOR = {'INFO': FG, 'WARN': ORANGE, 'ERR': RED, 'CMD': CYAN, 'OUT': (170, 182, 196), 'EVT': MAG, 'SYS': DIM}
SCENARIOS = {'ped': 1, 'slow': 2, 'parked': 3}
WEATHER = {'clear': 'ClearNoon', 'cloudy': 'CloudyNoon', 'rain': 'HardRainNoon', 'sunset': 'ClearSunset'}
SERIES = 600  # 30 s of samples at 20 Hz

# camera rigs on the ego: size, mount, fov, sensor_tick (0: every frame)
CAMS = {'chase': (BIG, carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15.0)), 90, 0.0),
        'hood': ((640, 360), carla.Transform(carla.Location(x=1.6, z=1.45), carla.Rotation(pitch=-4.0)), 90, 0.0),
        'rear': ((480, 270), carla.Transform(carla.Location(x=-2.4, z=1.2), carla.Rotation(yaw=180.0)), 90, 0.0),
        'mirror_l': ((480, 270), carla.Transform(carla.Location(x=0.6, y=-1.1, z=1.1), carla.Rotation(yaw=-160.0)), 60, 0.0),
        'mirror_r': ((480, 270), carla.Transform(carla.Location(x=0.6, y=1.1, z=1.1), carla.Rotation(yaw=160.0)), 60, 0.0),
        'bev': (BEV, carla.Transform(carla.Location(z=40.0), carla.Rotation(pitch=-90.0)), 70, 0.0)}
META = {  # panel: (tag, title)
    'chase': ('CAM-01', 'CHASE · 3RD PERSON'), 'hood': ('CAM-02', 'FRONT · HOOD'), 'rear': ('CAM-03', 'REAR VIEW'),
    'mirror_l': ('CAM-04', 'MIRROR · LEFT'), 'mirror_r': ('CAM-05', 'MIRROR · RIGHT'), 'bev': ('CAM-06', 'BEV · TRAFFIC OVERLAY'),
    'rsu': ('CAM-07', 'RSU POLE · NEAREST JUNCTION'), 'semseg': ('CAM-00', 'SEMANTIC INPUT 160×80'),
    'lidar': ('LDR-FUSE', 'COOPERATIVE LIDAR · EGO FRAME'), 'radar': ('LDR-SECT', 'SECTOR RANGE · POLICY ATTENTION'),
    'vae': ('PER-VAE', 'VAE RECONSTRUCTION · LATENT z'), 'policy': ('POL-PPO', 'ACTOR · CRITIC · ATTRIBUTION'),
    'reward': ('RWD', 'REWARD TERMS · TERMINATION'), 'train': ('TRN', 'RUN CONTROL'), 'v2x': ('V2X', 'CHANNEL · RSU · SPaT · DENM'),
    'readout': ('TLM', 'VEHICLE TELEMETRY'), 'situations': ('SIT', 'SITUATION ASSESSMENT'), 'sensors': ('SNS', 'SENSOR REGISTRY'),
    'metrics': ('MET', 'SESSION METRICS · EVENT TIMELINE 60 s'),
    'ch_speed': ('CH-01', 'SPEED km/h'), 'ch_ctrl': ('CH-02', 'CONTROL OUTPUT'), 'ch_reward': ('CH-03', 'STEP REWARD'),
    'ch_safety': ('CH-04', 'FRONT GAP m · MIN TTC s'), 'ch_v2x': ('CH-05', 'V2X RATE · LATENCY'), 'ch_value': ('CH-06', 'CRITIC V(s) · π PEDAL'),
    'ch_episodes': ('CH-07', 'EPISODE RETURN'), 'ch_perf': ('CH-08', 'LOOP TIMING ms'),
    'terminal': ('TTY0', 'TERMINAL'), 'system': ('SYS', 'SYSTEM STATUS'),
    'coopcf': ('CF-PI', 'COUNTERFACTUAL DECISION · WHAT COOP CHANGES'), 'radar_cmp': ('LDR-CMP', 'SECTORS · EGO ONLY vs FUSED'),
    'ch_coop': ('CH-09', 'LANE GAP m · EGO vs SHARED · PEDAL'), 'lidar_ego': ('LDR-00', 'EGO LIDAR ONLY')}
PAGES = {  # name: [(panel, (x, y, w, h))], page area y 56..660
    'overview': [('chase', (0, 56, 800, 380)), ('lidar', (804, 56, 796, 380)), ('bev', (0, 440, 320, 220)),
                 ('readout', (324, 440, 420, 220)), ('ch_speed', (748, 440, 420, 220)), ('situations', (1172, 440, 428, 220))],
    'cameras': [('chase', (0, 56, 532, 300)), ('hood', (536, 56, 532, 300)), ('rsu', (1072, 56, 528, 300)),
                ('mirror_l', (0, 360, 396, 300)), ('rear', (400, 360, 396, 300)), ('mirror_r', (800, 360, 396, 300)),
                ('bev', (1200, 360, 400, 300))],
    'sensors': [('semseg', (0, 56, 400, 300)), ('vae', (404, 56, 400, 300)), ('radar', (808, 56, 360, 300)),
                ('v2x', (1172, 56, 428, 300)), ('sensors', (0, 360, 800, 300)), ('lidar', (804, 360, 796, 300))],
    'charts': [(n, (k % 4 * 401, 56 + k // 4 * 304, 397, 300)) for k, n in enumerate(
        ['ch_speed', 'ch_ctrl', 'ch_reward', 'ch_safety', 'ch_v2x', 'ch_value', 'ch_episodes', 'ch_perf'])],
    'coop': [('coopcf', (0, 56, 532, 300)), ('radar_cmp', (536, 56, 532, 300)), ('ch_coop', (1072, 56, 528, 300)),
             ('lidar_ego', (0, 360, 796, 300)), ('lidar', (800, 360, 800, 300))],
    'policy': [('policy', (0, 56, 532, 300)), ('reward', (536, 56, 532, 300)), ('train', (1072, 56, 528, 300)),
               ('ch_episodes', (0, 360, 532, 300)), ('ch_reward', (536, 360, 532, 300)), ('metrics', (1072, 360, 528, 300))],
}
PAGE_AREA = (0, 56, W, 604)
STRIP = [('terminal', (0, H - BOTTOM, 1100, BOTTOM)), ('system', (1104, H - BOTTOM, 496, BOTTOM))]
HELP = [
    ('set [key value]', 'list or change settings (saved; * = applied after restart)'),
    ('mode <m>', 'drive · collect · train · eval · bc   (restarts into that mode)'),
    ('pipeline <cmd>', 'start · status · resume · retry · skip · stop   (collect→BC→eval→PPO→eval)'),
    ('results', 'table of results/*.json'),
    ('restart | server restart', 'reload the dashboard · restart the CARLA server too'),
    ('page <name|1-6>', 'overview cameras sensors charts coop policy   (or F1-F6)'),
    ('focus <panel>|off', 'one panel over the whole page'),
    ('spawn <event>', 'ped slow parked ' + ' '.join(EVENTS)),
    ('chaos on|off|<sec>', 'random surprise events near the ego'),
    ('density <n>', 'number of motorbikes kept around the ego'),
    ('driver expert|ppo', 'who drives (locked while collecting / training)'),
    ('weather <name>', ' '.join(WEATHER)),
    ('v2x <key> <value>', 'latency <ms> · loss <%> · range <m> · gps <m> · cars|bikes <%> · on · off'),
    ('pause | reset', 'freeze / resume · new episode'),
    ('replay | gif [file]', 'play back / save the last ~10 s'),
    ('fullscreen [on|off]', 'toggle fullscreen (or F11), aspect kept'),
    ('snapshot [file]', 'save the window as PNG'),
    ('stats | history | log', 'session metrics · typed commands · log file path'),
    ('cls | quit', 'clear the terminal · exit'),
    ('keys', 'Enter run · ↑↓ history · Tab complete · PgUp/PgDn or wheel scroll · Esc clear'),
]
COMPLETE = {'spawn': sorted(SCENARIOS) + sorted(EVENTS), 'weather': sorted(WEATHER), 'driver': ['expert', 'ppo'],
            'focus': sorted(META) + ['off'], 'chaos': ['on', 'off'], 'fullscreen': ['on', 'off'], 'page': list(PAGES),
            'v2x': ['latency', 'loss', 'range', 'gps', 'cars', 'bikes', 'on', 'off'],
            'set': [k for k in appcfg.DEFAULTS if k not in ('job', 'queue')], 'mode': list(appcfg.MODES),
            'pipeline': ['start', 'status', 'resume', 'retry', 'skip', 'stop'], 'server': ['restart']}
COMMANDS = ['help', 'set', 'mode', 'pipeline', 'results', 'restart', 'server', 'page', 'focus', 'spawn', 'chaos', 'density', 'driver', 'weather', 'v2x', 'pause', 'reset', 'replay',
            'gif', 'fullscreen', 'snapshot', 'stats', 'history', 'log', 'cls', 'clear', 'quit']


class ChaseCamera:
    '''RGB camera on a vehicle (or fixed in the world when vehicle is None); the callback only stores the frame.'''

    def __init__(self, vehicle, size=BIG, transform=None, fov=None, tick=0.0, world=None):
        world = world or vehicle.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(size[0]))
        bp.set_attribute('image_size_y', str(size[1]))
        bp.set_attribute('sensor_tick', str(tick))
        if fov:
            bp.set_attribute('fov', str(fov))
        self.frame, self.n, self.ts, self.size, self.tick = None, 0, None, size, tick
        transform = transform or carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15.0))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle) if vehicle else world.spawn_actor(bp, transform)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: ChaseCamera._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if self:
            a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            self.frame = a[:, :, 2::-1].swapaxes(0, 1).copy()  # BGRA -> RGB, (w, h, 3) for pygame
            self.n, self.ts = self.n + 1, image.timestamp


def lidar_render(clouds, poses, ego_pose, size, cam_back=28.0, cam_up=38.0, tilt=55.0, focal=None):
    '''CODD-style fused point cloud: oblique perspective from behind/above the ego, red near any sensor -> yellow far.'''
    w, h = size
    focal = focal or 330.0 * w / 640
    img = np.zeros((w, h, 3), np.uint8)
    ego_inv = np.linalg.inv(pose_matrix(*ego_pose))
    pts = [to_ego(c, p, ego_inv) for c, p in zip(clouds, poses)]
    sensors = np.array([to_ego([[0, 0, 0.001]], p, ego_inv)[0] for p in poses])  # 0.001: not dropped as padding
    if not pts:
        return img, []
    p = np.concatenate(pts)
    near = np.full(len(p), np.inf)
    for s in sensors:
        near = np.minimum(near, np.hypot(p[:, 0] - s[0], p[:, 1] - s[1]))
    t = math.radians(tilt)
    fwd, up = np.array([math.cos(t), 0, -math.sin(t)]), np.array([math.sin(t), 0, math.cos(t)])

    def project(q):
        q = q - np.array([-cam_back, 0, cam_up])
        depth = q @ fwd
        ok = depth > 1.0
        u = w / 2 + focal * q[:, 1] / np.where(ok, depth, 1)
        v = h / 2 - focal * (q @ up) / np.where(ok, depth, 1)
        return u, v, ok & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)

    u, v, ok = project(p)
    k = np.clip(near[ok] / 30.0, 0, 1)  # 0 m red, 30 m+ yellow
    color = np.stack([np.interp(k, [0, .4, 1], [255, 255, 235]), np.interp(k, [0, .4, 1], [40, 140, 205]),
                      np.interp(k, [0, .4, 1], [20, 0, 40])], 1).astype(np.uint8)
    ui, vi = u[ok].astype(int), v[ok].astype(int)
    img[ui, vi] = color
    img[ui + 1, vi] = color  # 2 px dots read better at this scale
    su, sv, sok = project(sensors)
    return img, [(int(a), int(b), i) for i, (a, b, o) in enumerate(zip(su, sv, sok)) if o]  # i: index of the sensor


def detect_situations(env):
    '''Rule-based labels for what is happening around the ego right now.'''
    out = []
    t = env.vehicle.get_transform()
    loc, fwd = t.location, t.get_forward_vector()
    vehicles = [a for a in env.world.get_actors().filter('vehicle.*') if a.id != env.vehicle.id]
    two = sum(1 for a in vehicles if a.type_id in MOTORBIKES + BICYCLES and a.get_location().distance(loc) < 15)
    boxed = 0
    for a in vehicles:
        if a.type_id in MOTORBIKES + BICYCLES:
            d = a.get_location() - loc
            boxed += abs(d.x * fwd.x + d.y * fwd.y) < 7 and abs(-d.x * fwd.y + d.y * fwd.x) < 3.5
    if boxed >= 4:
        out.append((f'Boxed in: {boxed} bikes around', RED))
    elif two:
        out.append((f'2-wheelers <15 m: {two}', RED if two >= 5 else ORANGE))
    lead = oncoming = None
    for a in vehicles:
        d = a.get_location() - loc
        ahead, side = d.x * fwd.x + d.y * fwd.y, abs(-d.x * fwd.y + d.y * fwd.x)
        if 0 < ahead < 25 and side < 2.0:
            f2 = a.get_transform().get_forward_vector()
            if f2.x * fwd.x + f2.y * fwd.y < -0.5:
                oncoming = min(oncoming or ahead, ahead)
            elif lead is None or ahead < lead[0]:
                lead = (ahead, math.hypot(a.get_velocity().x, a.get_velocity().y) * 3.6)
    if oncoming:
        out.append((f'Oncoming in lane {oncoming:.0f} m', RED))
    if lead:
        out.append((f'Lead {lead[0]:.0f} m · {lead[1]:.0f} km/h', RED if lead[0] < 10 else ORANGE))
    peds = [a.get_location().distance(loc) for a in env.world.get_actors().filter('walker.pedestrian.*')]
    if peds and min(peds) < 15:
        out.append((f'Pedestrian {min(peds):.0f} m', RED if min(peds) < 6 else ORANGE))
    if env.front_obstacle < COOP_CORRIDOR:
        out.append((f'Lane blocked {env.front_obstacle:.1f} m', RED if env.front_obstacle < 10 else ORANGE))
    if env.vehicle.is_at_traffic_light():
        out.append((f'Light {env.vehicle.get_traffic_light_state()}', YELLOW))
    if env.map.get_waypoint(loc).is_junction:
        out.append(('Junction', YELLOW))
    if env.distance_from_center > 1.5:
        out.append((f'Off-center {env.distance_from_center:.1f} m', RED if env.distance_from_center > 2.5 else ORANGE))
    return out or [('Clear', GREEN)]


def ego_frame(env):
    '''Ego transform and a rotation of world xy offsets into (forward, right) metres.'''
    t = env.vehicle.get_transform()
    c, sn = math.cos(math.radians(t.rotation.yaw)), math.sin(math.radians(t.rotation.yaw))
    return t, lambda dx, dy: (dx * c + dy * sn, -dx * sn + dy * c)


def conflicts(env, radius=30.0):
    '''Every road user within radius m in the ego frame: (actor, (fwd, right), velocity, rel. velocity, ttc, miss distance).'''
    t, rot = ego_frame(env)
    ev = env.vehicle.get_velocity()
    out = []
    for a in list(env.world.get_actors().filter('vehicle.*')) + list(env.world.get_actors().filter('walker.pedestrian.*')):
        if a.id == env.vehicle.id:
            continue
        l, v = a.get_location(), a.get_velocity()
        p = rot(l.x - t.location.x, l.y - t.location.y)
        if math.hypot(*p) > radius:
            continue
        ttc, dist = closest_approach([p], [rot(v.x - ev.x, v.y - ev.y)])
        out.append((a, p, rot(v.x, v.y), rot(v.x - ev.x, v.y - ev.y), float(ttc[0]), float(dist[0])))
    return out


class Tee:
    '''Copies stdout (env, agent and CARLA prints) into the dashboard terminal.'''

    def __init__(self, stream, sink):
        self.stream, self.sink, self.buf = stream, sink, ''

    def write(self, s):
        self.stream.write(s)
        self.buf += s
        *done, self.buf = self.buf.split('\n')
        for line in done:
            if line.strip():
                self.sink(line.rstrip(), 'SYS')

    def flush(self):
        self.stream.flush()


class Dashboard:
    def __init__(self, agent, policy_ok, mode, config, cfg, app):
        try:  # without this Windows magnifies the window by the display scale (200 %) and it no longer fits
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        pygame.init()
        pygame.key.set_repeat(400, 30)
        info = pygame.display.Info()
        k = min(1.0, info.current_w * 0.96 / W, (info.current_h - 100) * 0.96 / H)
        self.windowed, self.fullscreen, self.toggled = (int(W * k), int(H * k)), False, 0.0
        self.window = pygame.display.set_mode(self.windowed, pygame.RESIZABLE)
        try:
            pygame.display.set_icon(pygame.image.load(os.path.join('assets', 'coopdrive.png')))
        except Exception:
            pass
        self.screen = pygame.Surface((W, H))  # everything is drawn here at a fixed size
        self.run_id = time.strftime('%Y%m%d-%H%M%S')
        pygame.display.set_caption(f'Coop-Drive research console · run {self.run_id}')
        f = 'consolas,dejavusansmono,couriernew'
        self.mono = pygame.font.SysFont(f, 13)
        self.small = pygame.font.SysFont(f, 11)
        self.bold = pygame.font.SysFont(f, 13, bold=True)
        self.large = pygame.font.SysFont(f, 22, bold=True)
        self.agent, self.policy_ok, self.mode, self.config = agent, policy_ok, mode, config
        self.cfg, self.app, self.exit_code, self.pending, self.cache = cfg, app, 0, [], {}
        self.status = ''  # job progress line for the run-control panel
        self.decoder = Decoder(LATENT_DIM).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.decoder.load()
        self.decoder.eval()
        self.term = Terminal(os.path.join('logs', f'session_{self.run_id}.log'))
        self.cams, self.rsu_cam, self.rsu_j, self.collision = {}, None, None, None
        self.episode, self.ep_reward, self.ep_hist = 0, 0.0, []  # ep_hist: returns of finished episodes
        self.driver = 'ppo' if policy_ok and mode != 'collect' else 'expert'
        self.line, self.hist_i, self.page, self.focus = '', 0, cfg['page'] if cfg['page'] in PAGES else 'overview', None
        self.need_capture, self.remote_t, self.composing = False, 0.0, ''
        self.paused = self.quit = self.want_reset = False
        self.metrics, self.timeline, self.sim_t = SessionMetrics(), [], 0.0  # timeline: (sim time s, kind, label)
        self.frames = collections.deque(maxlen=120)  # ~10+ s of half-size window frames for the instant replay
        self.replay, self.replay_i = None, 0
        self.hits, self.hit_ids, self.series = [], [], collections.defaultdict(lambda: collections.deque(maxlen=SERIES))
        self.ms = collections.defaultdict(float)  # latest timings: step, vae, policy, draw
        self.mean, self.value, self.sens, self.sens_step = np.zeros(2), 0.0, None, -1
        self.cf, self.coop_stats = None, collections.Counter(steps=0)
        self.worst, self.prev_speed, self.accel, self.yaw_rate, self.prev_yaw = None, 0.0, 0.0, 0.0, None
        self.train = {'timestep': 0, 'target': 0, 'episode': 0, 'avg': 0.0, 'updates': 0, 'update_s': 0.0, 'saves': 0,
                      'start': time.time(), 'steps0': 0, 'std': getattr(agent, 'action_std', 0.0), 'note': ''}
        self.fps, self.last_frame, self.last_input, self.step_wall, self.prev_step_t = 0.0, time.time(), 0.0, 0.05, time.time()
        self.log(f'run {self.run_id} · mode {mode} · {config}', 'INFO')
        self.log(f'session log → {self.term.path}', 'INFO')

    # ------------------------------------------------------------------ state
    def log(self, text, level='INFO'):
        self.term.log(text, level, self.sim_t)
        self.term.scroll = 0 if self.term.scroll == 0 else self.term.scroll + 1  # stay where the user scrolled to

    def new_episode(self, env):
        for name, (size, tr, fov, tick) in CAMS.items():
            self.cams[name] = ChaseCamera(env.vehicle, size, tr, fov, tick)
        self.rsu_cam, self.rsu_j = None, None
        bp = env.world.get_blueprint_library().find('sensor.other.collision')
        self.collision = env.world.spawn_actor(bp, carla.Transform(), attach_to=env.vehicle)
        self.hits = []
        self.hits, self.hit_ids = [], []
        self.collision.listen(lambda e: (self.hits.append(e.other_actor.type_id), self.hit_ids.append(e.other_actor.id)))
        env.sensor_list += [c.sensor for c in self.cams.values()] + [self.collision]
        self.episode += 1
        self.ep_reward = 0.0
        self.metrics.episodes += 1
        self.mark('episode', f'ep {self.episode}')
        self.log(f'episode {self.episode} start · route {len(env.route_waypoints)} wp', 'INFO')

    def mark(self, kind, label=''):
        self.timeline = [e for e in self.timeline if e[0] > self.sim_t - 60] + [(self.sim_t, kind, label)]

    def after_step(self, env, o, reward):
        '''Per simulation step: telemetry series, policy outputs, session metrics, timeline events and collisions.'''
        self.sim_t += 0.05
        self.need_capture = True
        now = time.time()
        self.step_wall, self.prev_step_t = 0.8 * self.step_wall + 0.2 * (now - self.prev_step_t), now
        with torch.no_grad():
            x = o.cpu().unsqueeze(0)
            self.mean = self.agent.old_policy.actor(x)[0].numpy()
            self.value = float(self.agent.old_policy.critic(x)[0])
        self.counterfactual(env, o)
        ctrl = env.vehicle.get_control()
        yaw = env.vehicle.get_transform().rotation.yaw
        if self.prev_yaw is not None:
            self.yaw_rate = ((yaw - self.prev_yaw + 180) % 360 - 180) / 0.05
        self.accel = 0.8 * self.accel + 0.2 * (env.velocity - self.prev_speed) / 3.6 / 0.05  # smoothed: physics jitter
        self.prev_speed, self.prev_yaw = env.velocity, yaw
        if env.timesteps % 2 == 0:  # every other step is enough for the conflict scan
            near = conflicts(env)
            hot = [(n[4], n[0].type_id.split('.')[-1]) for n in near if n[5] < 2.5 and 0 < n[4] < 3]
            self.worst = min(hot) if hot else None
        v2x = env.infra.channel.summary(env.infra.t) if env.infra else {'msg_per_s': 0, 'latency_ms': 0, 'loss': 0}
        s = self.series
        for k, v in (('t', self.sim_t), ('speed', env.velocity), ('target', env.target_speed), ('steer', ctrl.steer),
                     ('throttle', ctrl.throttle), ('brake', ctrl.brake), ('reward', reward),
                     ('front', min(env.front_obstacle, COOP_CORRIDOR)), ('ttc', self.worst[0] if self.worst else 3.0),
                     ('msg', v2x['msg_per_s']), ('lat', v2x['latency_ms']), ('loss', 100 * v2x['loss']),
                     ('value', self.value), ('pi_pedal', self.mean[1]), ('pi_steer', self.mean[0]),
                     ('horn', float(getattr(env, 'horn', False))),
                     ('step_ms', self.ms['step']), ('vae_ms', self.ms['vae']), ('pi_ms', self.ms['policy']), ('draw_ms', self.ms['draw'])):
            s[k].append(float(v))
        if env.infra and env.timesteps % 20 == 1:
            self.update_rsu_cam(env)
        w = env.infra.warning if env.infra else None
        if w and w['t'] - getattr(self, 'last_denm', -9) > 1.0:  # one mark per warning burst
            self.last_denm = w['t']
            self.mark('denm', w['kind'])
            self.log(f"DENM {w['kind']} from {w['from']}", 'EVT')
        for kind in self.metrics.step(env.velocity, env.front_obstacle, env.red_moving):
            self.mark(kind)
            if kind == 'red_light':
                self.log('red light passed while moving', 'WARN')
        if self.hits:
            what = self.hits[0].split('.')[-1]
            self.hits = []
            self.metrics.collisions += 1
            self.mark('collision', what)
            self.log(f'collision with {what}', 'ERR')
            return True
        return False

    def counterfactual(self, env, o):
        '''The policy's action on the same moment with less information: no shared LiDAR, no infrastructure, ego only.'''
        if not env.lidar_objs or env.lidar_objs[0].latest is None:  # episode over: the env has already removed its sensors
            return
        alt = torch.tensor(env.coop_obs_other(), dtype=torch.float32)  # the coop part as the ego's own LiDAR alone sees it
        o = o.cpu().float()
        o_alt = torch.cat([o[:100], alt])
        inf = slice(100 + 2 * (COOP_SECTORS + 1) + SIGNAL_DIM, 100 + 2 * (COOP_SECTORS + 1) + SIGNAL_DIM + INFRA_DIM)
        no_inf = lambda x: torch.cat([x[:inf.start], torch.zeros(INFRA_DIM), x[inf.stop:]])
        with torch.no_grad():
            mu = self.agent.old_policy.actor(torch.stack([o, o_alt, no_inf(o), no_inf(o_alt)])).numpy()
        self.cf = {'full': mu[0], 'no_v2v': mu[1], 'no_v2i': mu[2], 'ego': mu[3],
                   'gap_shared': min(env.front_obstacle, COOP_CORRIDOR), 'gap_ego': float(alt[COOP_SECTORS]) * COOP_CORRIDOR,
                   'sect_ego': alt[:COOP_SECTORS].numpy(), 'sect_full': o[100:100 + COOP_SECTORS].numpy()}
        c = self.coop_stats
        c['steps'] += 1
        c['changed'] += abs(mu[0][1] - mu[3][1]) > 0.2  # coop moves the pedal decision by more than 0.2
        c['brake_only_coop'] += mu[0][1] < -0.3 <= mu[3][1]  # brakes with coop, would not brake without
        c['hidden'] += self.cf['gap_shared'] < COOP_CORRIDOR - 0.01 and self.cf['gap_ego'] >= COOP_CORRIDOR - 0.01
        c['denm'] += env.infra is not None and env.infra.warning_now() is not None
        for k in ('gap_shared', 'gap_ego'):
            self.series[k].append(self.cf[k])
        self.series['ped_full'].append(float(mu[0][1]))
        self.series['ped_ego'].append(float(mu[3][1]))

    def update_rsu_cam(self, env):
        '''Fixed camera on the pole of the nearest RSU (within 90 m), aimed at the ego's approach.'''
        e = env.vehicle.get_location()
        near = min(env.infra.rsus, key=lambda r: math.hypot(r['xy'][0] - e.x, r['xy'][1] - e.y), default=None)
        if near is None or math.hypot(near['xy'][0] - e.x, near['xy'][1] - e.y) > 90 or near['junction'] == self.rsu_j:
            return
        if self.rsu_cam and self.rsu_cam.sensor.is_alive:
            if self.rsu_cam.sensor in env.sensor_list:
                env.sensor_list.remove(self.rsu_cam.sensor)
            self.rsu_cam.sensor.destroy()
        yaw = math.degrees(math.atan2(e.y - near['xy'][1], e.x - near['xy'][0]))
        tr = carla.Transform(carla.Location(near['xy'][0], near['xy'][1], near['ground'] + 7.0), carla.Rotation(pitch=-28.0, yaw=yaw))
        self.rsu_cam, self.rsu_j = ChaseCamera(None, (640, 360), tr, 90, 0.0, env.world), near['junction']
        env.sensor_list.append(self.rsu_cam.sensor)
        self.log(f"RSU camera → junction {near['junction']}", 'SYS')

    def sensitivity(self, env, o):
        if self.sens_step != env.timesteps:
            self.sens, self.sens_step = input_sensitivity(self.agent.old_policy.actor, o), env.timesteps
        return self.sens

    def set_fullscreen(self, on):
        self.fullscreen, self.toggled = on, time.time()
        self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if on else pygame.display.set_mode(self.windowed, pygame.RESIZABLE)
        self.log(f"fullscreen {'on' if on else 'off'} · {self.window.get_width()}×{self.window.get_height()} · F11 toggles", 'OUT')

    def present(self, surf):
        '''Show the fixed-size canvas in the window, scaled with the aspect kept (letterboxed in fullscreen).'''
        ww, wh = self.window.get_size()
        if (ww, wh) == surf.get_size():
            self.window.blit(surf, (0, 0))
        else:
            k = min(ww / surf.get_width(), wh / surf.get_height())
            size = (int(surf.get_width() * k), int(surf.get_height() * k))
            self.window.fill(BG)
            self.window.blit(pygame.transform.smoothscale(surf, size), ((ww - size[0]) // 2, (wh - size[1]) // 2))
        pygame.display.flip()

    def start_replay(self):
        if len(self.frames) < 5:
            return self.log('nothing to replay yet', 'WARN')
        self.replay, self.replay_i = list(self.frames), 0
        self.log(f'replay {len(self.replay)} frames · Esc to stop', 'OUT')

    def save_gif(self, path=None):
        from PIL import Image
        if len(self.frames) < 5:
            return self.log('nothing to save yet', 'WARN')
        os.makedirs('replays', exist_ok=True)
        path = path or os.path.join('replays', time.strftime('replay_%H%M%S.gif'))
        imgs = [Image.frombytes('RGB', f.get_size(), pygame.image.tostring(f, 'RGB')).resize((560, 315))
                for f in list(self.frames)[::2]]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        self.log(f'saved {path}', 'OUT')

    # ------------------------------------------------------------------ app: settings, jobs, restarts
    def cached(self, key, env, fn):
        '''fn() once per simulation step: the UI redraws several times per step and must not redo the heavy parts.'''
        k = (key, env.timesteps)
        if k not in self.cache:
            self.cache = {kk: v for kk, v in self.cache.items() if kk[1] == env.timesteps}
            self.cache[k] = fn()
        return self.cache[k]

    def run_pending(self, env):
        while self.pending:
            self.run(env, self.pending.pop(0))

    def save_cfg(self):
        if self.app:
            appcfg.save(self.cfg)

    def restart(self, why, code=RESTART):
        if not self.app:
            return self.log('restart needs the app (CoopDrive.bat); this session was started with command-line flags', 'WARN')
        self.save_cfg()
        self.log(f'{why} · restarting…', 'INFO')
        self.exit_code, self.quit = code, True

    def finish_job(self, summary=''):
        '''The current mode finished its work: next queued job, or back to drive.'''
        self.log(f'job {self.mode} complete' + (f' · {summary}' if summary else ''), 'INFO')
        if not self.app:
            self.quit = True
            return
        self.cfg['queue'] = appcfg.load()['queue']  # jobs queued from outside while this one ran (pipeline extensions)
        nxt = appcfg.next_job(self.cfg)
        self.restart(f'next: {nxt}' + (f" ({len(self.cfg['queue'])} queued)" if self.cfg['queue'] else ''))

    def fail_job(self, why):
        self.log(f'job {self.mode} failed: {why}', 'ERR')
        self.log('pipeline retry · pipeline skip · pipeline stop · mode drive', 'WARN')
        self.status = 'FAILED · ' + why

    def cmd_set(self, args):
        cfg = self.cfg
        if not args:
            for k in appcfg.DEFAULTS:
                if k not in ('job', 'queue', 'queue_total'):
                    self.log(f"  {k:16} {str(cfg[k]):34} {'' if k in appcfg.LIVE else '*'} {appcfg.HELP.get(k, '')}", 'OUT')
            return self.log(f"  file {appcfg.PATH if self.app else '(command-line session: not saved)'}", 'OUT')
        try:
            val = appcfg.coerce(args[0], ' '.join(args[1:])) if len(args) > 1 else None
            cur = cfg[args[0]] if args[0] not in ('job', 'queue') else None
        except KeyError:
            return self.log(f'no setting {args[0]} · set lists them', 'WARN')
        except ValueError as e:
            return self.log(str(e) or f'bad value for {args[0]}', 'WARN')
        key = args[0]
        if val is None:
            return self.log(f'{key} = {cur}   {appcfg.HELP.get(key, "")}', 'OUT')
        if key == 'mode':
            return self.cmd_mode([val])
        if key == 'driver' and val == 'ppo' and not self.policy_ok:
            return self.log('no matching PPO model', 'WARN')
        cfg[key] = val
        self.save_cfg()
        if key == 'fullscreen':
            self.set_fullscreen(val)
        elif key == 'page':
            self.page = val if val in PAGES else self.page
        elif key == 'driver' and self.mode == 'drive':
            self.driver = 'ppo' if val == 'ppo' or (val == 'auto' and self.policy_ok) else 'expert'
        self.log(f'{key} = {val}' + ('' if key in appcfg.LIVE else '  · type restart to apply'), 'OUT')

    def cmd_mode(self, args):
        if not args or args[0] not in appcfg.MODES:
            return self.log(f'mode {self.mode} · modes: {" ".join(appcfg.MODES)}', 'OUT')
        self.cfg['job'], self.cfg['mode'] = {'mode': args[0]}, args[0]  # a manual job: the pipeline queue waits
        self.restart(f'mode {args[0]}')

    def cmd_pipeline(self, args):
        cfg, a = self.cfg, args[0] if args else 'status'
        if a == 'start':
            cfg['queue'] = appcfg.pipeline(cfg['town'])
            cfg['queue_total'] = len(cfg['queue'])
            appcfg.start_job(cfg, cfg['queue'][0])
            return self.restart(f"pipeline of {len(cfg['queue'])} jobs")
        if a == 'resume':
            if not cfg['queue']:
                return self.log('nothing queued · pipeline start', 'WARN')
            appcfg.start_job(cfg, cfg['queue'][0])
            return self.restart('pipeline resumed')
        if a == 'retry':
            return self.restart(f'retry {self.mode}')
        if a == 'skip':
            return self.restart(f'skipped {self.mode} · next {appcfg.next_job(cfg)}')
        if a == 'stop':
            cfg['queue'], cfg['job'], cfg['mode'] = [], {}, 'drive'
            return self.restart('pipeline stopped')
        if not cfg['queue']:
            return self.log('pipeline idle · pipeline start runs collect → BC → eval → PPO fine-tune → eval', 'OUT')
        for k, j in enumerate(cfg['queue']):
            what = ' '.join(f'{x}={v}' for x, v in j.items() if x not in ('mode', 'vn', 'infra'))
            self.log(f"  {'▶' if k == 0 and j == cfg['job'] else ' '} {k + 1}. {j['mode']:8} {what}", 'OUT')

    def cmd_results(self):
        import glob
        import json
        rows = []
        for f in sorted(glob.glob('results/*.json')):
            try:
                d = json.load(open(f))
            except ValueError:
                continue
            m = d.get('summary', {})
            if 'collisions_per_km' in m:
                rows.append(f"  {d.get('name', f):22} {m['km']:6.2f} km  coll/km {m['collisions_per_km']:6.2f}  near-miss s/km "
                            f"{m['near_miss_s_per_km']:6.1f}  red/km {m['red_light_runs_per_km']:5.2f}  route {m['route_done_%']:5.1f}%  "
                            f"{m['mean_speed_kmh']:5.1f} km/h")
        for r in rows or ['  no results yet']:
            self.log(r, 'OUT')

    # ------------------------------------------------------------------ commands
    def run(self, env, line):
        parts = line.strip().lstrip('/').split()
        if not parts:
            return
        cmd, args = parts[0].lower(), parts[1:]
        out = lambda text, level='OUT': self.log(text, level)
        if cmd == 'help':
            for c, d in HELP:
                out(f'  {c:26} {d}')
            return
        if cmd == 'set':
            return self.cmd_set(args)
        if cmd == 'mode':
            return self.cmd_mode(args)
        if cmd == 'pipeline':
            return self.cmd_pipeline(args)
        if cmd == 'results':
            return self.cmd_results()
        if cmd == 'restart':
            return self.restart('restart')
        if cmd == 'server':
            return self.restart('CARLA server restart', SERVER_RESTART) if args[:1] == ['restart'] else out('server restart', 'WARN')
        if cmd == 'page':
            names = list(PAGES)
            a = args[0].lower() if args else ''
            name = names[int(a) - 1] if a.isdigit() and 1 <= int(a) <= len(names) else a
            if name not in PAGES:
                return out(f'pages: {" ".join(names)}', 'WARN')
            self.page, self.focus = name, None
            return out(f'page {name}')
        if cmd == 'focus':
            name = args[0].lower() if args else 'off'
            if name != 'off' and name not in META:
                return out(f'panels: {" ".join(META)}', 'WARN')
            self.focus = None if name == 'off' else name
            return out(f'focus {name}')
        if cmd == 'spawn' and args:
            name = args[0].lower()
            if name in SCENARIOS:
                ok = spawn_scenario(env, SCENARIOS[name])
            elif name in EVENTS:
                ok = bool(EVENTS[name](env))
            else:
                return out(f'unknown event {name}', 'WARN')
            if ok:
                self.mark('chaos', name)
            return out(f'spawned {name}' if ok else f'{name}: no free spot, try again', 'OUT' if ok else 'WARN')
        if cmd == 'chaos':
            if not env.traffic:
                return out('chaos needs --vn', 'WARN')
            a = args[0].lower() if args else ''
            env.traffic.chaos = 0.0 if a == 'off' else 5.0 if a in ('on', '') else max(1.0, float(a)) if a.replace('.', '', 1).isdigit() else env.traffic.chaos
            env.traffic.next_event = env.traffic.ticks + int(env.traffic.chaos * 20)
            return out(f'chaos {"off" if not env.traffic.chaos else f"~every {env.traffic.chaos:.0f} s"}')
        if cmd == 'density' and args and args[0].isdigit():
            if not env.traffic:
                return out('density needs --vn', 'WARN')
            return out(f'motorbikes {env.traffic.set_motorbikes(env, int(args[0]))}')
        if cmd == 'driver' and args and args[0] in ('expert', 'ppo'):
            if self.mode != 'drive':
                return out(f'driver is locked while in {self.mode} mode', 'WARN')
            self.cfg['driver'] = args[0]
            self.save_cfg()
            if args[0] == 'ppo' and not self.policy_ok:
                return out('no matching PPO model', 'WARN')
            self.driver = args[0]
            return out(f'driver {self.driver}')
        if cmd == 'weather' and args and args[0] in WEATHER:
            env.world.set_weather(getattr(carla.WeatherParameters, WEATHER[args[0]]))
            return out(f'weather {args[0]}')
        if cmd == 'pause':
            self.paused = not self.paused
            return out('paused' if self.paused else 'resumed')
        if cmd == 'reset':
            self.want_reset = True
            return out('reset requested')
        if cmd == 'v2x':
            if not env.infra:
                return out('v2x needs --infra', 'WARN')
            ch, key = env.infra.channel, args[0].lower() if args else ''
            val = float(args[1]) if len(args) > 1 and args[1].replace('.', '', 1).isdigit() else None
            if key in ('on', 'off'):
                env.infra.enabled = key == 'on'
            elif key == 'latency' and val is not None:
                ch.latency = val / 1000
            elif key == 'loss' and val is not None:
                ch.loss = min(val, 100) / 100
            elif key == 'range' and val is not None:
                ch.range = val
            elif key == 'gps' and val is not None:
                env.infra.gps_sigma = val
            elif key in ('cars', 'bikes') and val is not None:  # share of vehicles fitted with V2X (send CAM)
                setattr(env.infra, 'cam_' + key, min(val, 100) / 100)
                env.infra.connected = {}
            return out(f"v2x {'on' if env.infra.enabled else 'off'} · {ch.latency * 1000:.0f} ms · loss {ch.loss * 100:.0f}% · "
                       f"range {ch.range:.0f} m · gps {env.infra.gps_sigma:.1f} m · fitted cars {env.infra.cam_cars * 100:.0f}% "
                       f"bikes {env.infra.cam_bikes * 100:.0f}%")
        if cmd == 'fullscreen':
            a = args[0].lower() if args else ''
            return self.set_fullscreen(not self.fullscreen if a not in ('on', 'off') else a == 'on')
        if cmd == 'replay':
            return self.start_replay()
        if cmd == 'gif':
            return self.save_gif(args[0] if args else None)
        if cmd == 'snapshot':
            path = args[0] if args else time.strftime('snapshot_%H%M%S.png')
            pygame.image.save(self.screen, path)
            return out(f'saved {path}')
        if cmd == 'stats':
            m = self.metrics.summary()
            return out(f"distance {m['km'] * 1000:.0f} m · collisions/km {m['collisions_per_km']:.2f} · near-miss s/km "
                       f"{m['near_miss_s_per_km']:.1f} · red/km {m['red_runs_per_km']:.2f} · mean {m['mean_speed']:.1f} km/h · "
                       f"episodes {self.metrics.episodes}")
        if cmd == 'history':
            for k, h in enumerate(self.term.history[-20:]):
                out(f'  {k + 1:3d}  {h}')
            return
        if cmd == 'log':
            return out(f'{os.path.abspath(self.term.path)} · {len(self.term.lines)} lines in view')
        if cmd in ('cls', 'clear'):
            return self.term.clear()
        if cmd in ('quit', 'exit'):
            self.quit = True
            return
        out(f"unknown command '{cmd}' · type help", 'WARN')

    def complete(self, line):
        '''Tab: complete the command, then its argument; several matches -> longest common prefix.'''
        slash = '/' if line.startswith('/') else ''
        parts = line.lstrip('/').split(' ')
        options, word = (COMMANDS, parts[0]) if len(parts) == 1 else (COMPLETE.get(parts[0], []), parts[-1])
        hits = [o for o in options if o.startswith(word)]
        if not hits:
            return line
        if len(hits) > 1:
            self.log('  '.join(hits), 'OUT')
        return slash + ' '.join(parts[:-1] + [os.path.commonprefix(hits) + (' ' if len(hits) == 1 else '')])

    def handle_events(self, env):
        '''Keyboard / mouse / window events; True if there were any (the UI then redraws at full fps).'''
        rows = self.term_rows()
        if time.time() - self.remote_t > 1.0:  # logs/remote.txt: commands from scripts, run like typed ones (one per line)
            self.remote_t = time.time()
            path = os.path.join('logs', 'remote.txt')
            if os.path.exists(path):
                try:
                    lines = open(path, encoding='utf-8').read().splitlines()
                    os.remove(path)
                except OSError:
                    lines = []
                for line in lines:
                    if line.strip():
                        self.term.command(line.strip(), self.sim_t)
                        self.pending.append(line)
        events = pygame.event.get()
        if any(e.type in (pygame.KEYDOWN, pygame.TEXTINPUT, pygame.TEXTEDITING, pygame.MOUSEWHEEL) for e in events):
            self.last_input = time.time()
        for e in events:
            if e.type == pygame.QUIT:
                self.quit = True
            elif e.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.windowed = e.size
                self.window = pygame.display.set_mode(e.size, pygame.RESIZABLE)
            elif e.type == pygame.MOUSEWHEEL:
                self.term.scroll += 3 * e.y
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_F11:
                    if time.time() - self.toggled > 0.8:  # a mode switch takes a while: key repeat flipped it back and forth
                        self.set_fullscreen(not self.fullscreen)
                elif pygame.K_F1 <= e.key <= pygame.K_F1 + len(PAGES) - 1:
                    self.page, self.focus = list(PAGES)[e.key - pygame.K_F1], None
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    line, self.line, self.composing = self.line + self.composing, '', ''  # an IME word not committed yet counts
                    self.term.scroll = 0
                    if line.strip():
                        self.term.command(line.strip(), self.sim_t)
                        self.pending.append(line)  # run between simulation steps (the sim may be mid-tick)
                    self.hist_i = len(self.term.history)
                elif e.key == pygame.K_ESCAPE:
                    self.line, self.composing, self.replay = '', '', None
                elif e.key in (pygame.K_UP, pygame.K_DOWN) and self.term.history:
                    hist = self.term.history
                    self.hist_i = max(0, min(len(hist), self.hist_i + (-1 if e.key == pygame.K_UP else 1)))
                    self.line = hist[self.hist_i] if self.hist_i < len(hist) else ''
                elif e.key == pygame.K_PAGEUP:
                    self.term.scroll += rows - 1
                elif e.key == pygame.K_PAGEDOWN:
                    self.term.scroll = max(0, self.term.scroll - rows + 1)
                elif e.key == pygame.K_TAB:
                    self.line, self.composing = self.complete(self.line + self.composing), ''
                elif e.key == pygame.K_BACKSPACE and not self.composing:  # while composing, the IME edits its own text
                    self.line = self.line[:-1]
            elif e.type == pygame.TEXTEDITING:  # Vietnamese IMEs (Telex/VNI) hold a word here until space: show it live
                self.composing = e.text
            elif e.type == pygame.TEXTINPUT:  # committed text after the keyboard layout / IME
                self.line += e.text
                self.composing = ''
        if env.traffic and env.traffic.log:  # surprise events fired by the chaos timer
            for name in env.traffic.log:
                self.log(f'chaos event: {name}', 'EVT')
                self.mark('chaos', name)
            env.traffic.log.clear()
        return bool(events)

    # ------------------------------------------------------------------ drawing helpers
    def text(self, s, msg, pos, color=FG, font=None, right=False):
        img = (font or self.mono).render(msg, True, color)
        s.blit(img, (pos[0] - img.get_width(), pos[1]) if right else pos)
        return img.get_width()

    def led(self, s, pos, color):
        pygame.draw.circle(s, color, pos, 4)
        pygame.draw.circle(s, tuple(min(255, c + 60) for c in color), (pos[0] - 1, pos[1] - 1), 1)

    def frame(self, name, rect):
        '''Panel chrome (border, header strip with tag + title); returns the content subsurface.'''
        x, y, w, h = rect
        s = self.screen
        pygame.draw.rect(s, PANEL, rect)
        pygame.draw.rect(s, EDGE, rect, 1)
        pygame.draw.rect(s, HEADC, (x + 1, y + 1, w - 2, 17))
        tag, title = META[name]
        tw = self.text(s, tag, (x + 6, y + 2), CYAN, self.bold)
        self.text(s, title, (x + 14 + tw, y + 3), FG, self.small)
        return s.subsurface((x + 1, y + 19, w - 2, h - 20))

    def cover(self, s, frame, flip=False):
        '''Blit a camera frame scaled to cover s (center crop). Returns the scale factor.'''
        w, h = s.get_size()
        img = pygame.surfarray.make_surface(frame)
        k = max(w / img.get_width(), h / img.get_height())
        img = pygame.transform.scale(img, (int(img.get_width() * k + 1), int(img.get_height() * k + 1)))
        if flip:
            img = pygame.transform.flip(img, True, False)
        s.blit(img, ((w - img.get_width()) // 2, (h - img.get_height()) // 2))
        return k

    def caption(self, s, msg):
        '''Small readout in the bottom-left corner of an image panel, on a dark plate so it reads on any picture.'''
        img = self.small.render(msg, True, CYAN)
        y = s.get_height() - img.get_height() - 6
        pygame.draw.rect(s, (6, 8, 11), (6, y - 2, img.get_width() + 8, img.get_height() + 4))
        s.blit(img, (10, y))

    def reticle(self, s):
        w, h = s.get_size()
        for (cx, cy), (dx, dy) in (((8, 8), (1, 1)), ((w - 9, 8), (-1, 1)), ((8, h - 9), (1, -1)), ((w - 9, h - 9), (-1, -1))):
            pygame.draw.line(s, CYAN, (cx, cy), (cx + 12 * dx, cy), 1)
            pygame.draw.line(s, CYAN, (cx, cy), (cx, cy + 12 * dy), 1)

    def hbar(self, s, x, y, w, value, label, color, lo=-1.0):
        '''Horizontal gauge from lo..1 (zero line drawn when lo < 0).'''
        pygame.draw.rect(s, GRID, (x, y, w, 9))
        value = float(np.clip(value, lo, 1))
        zero = x + w * (-lo) / (1 - lo)
        end = x + w * (value - lo) / (1 - lo)
        pygame.draw.rect(s, color, (min(zero, end), y, max(abs(end - zero), 2), 9))
        if lo < 0:
            pygame.draw.line(s, DIM, (zero, y - 2), (zero, y + 10))
        self.text(s, f'{label:9}{value:+.3f}', (x + w + 8, y - 2), FG, self.small)

    def chart(self, s, keys, colors, names, unit='', lo=None, hi=None, window=30.0):
        '''Time series of the last `window` s with nice y ticks, x grid every 5 s and a legend with current values.'''
        w, h = s.get_size()
        L, R, T, B = 46, 10, 20, 18
        pw, ph = w - L - R, h - T - B
        t = np.array(self.series['t'])
        pygame.draw.rect(s, (10, 12, 17), (L, T, pw, ph))
        if len(t) < 2:
            self.text(s, 'awaiting samples…', (L + 8, T + 8), DIM, self.small)
            return
        keep = t > t[-1] - window
        data = [np.array(self.series[k])[keep] for k in keys]
        vals = np.concatenate(data)
        ticks = nice_ticks(vals.min() if lo is None else lo, vals.max() if hi is None else hi)
        y0, y1 = ticks[0], ticks[-1]
        ypx = lambda v: T + ph * (1 - (np.clip(v, y0, y1) - y0) / (y1 - y0))
        for v in ticks:
            pygame.draw.line(s, GRID, (L, ypx(v)), (L + pw, ypx(v)))
            self.text(s, f'{v:g}', (L - 5, ypx(v) - 6), DIM, self.small, right=True)
        for sec in range(0, int(window) + 1, 5):
            px = L + pw * (1 - sec / window)
            pygame.draw.line(s, GRID, (px, T), (px, T + ph))
            self.text(s, f'-{sec}s' if sec else 't0', (px - 10, T + ph + 3), DIM, self.small)
        if y0 < 0 < y1:
            pygame.draw.line(s, DIM, (L, ypx(0)), (L + pw, ypx(0)))
        xs = L + pw * (1 - (t[-1] - t[keep]) / window)
        x = L + 4
        for d, c, n in zip(data, colors, names):
            if len(d) > 1:
                pygame.draw.lines(s, c, False, list(zip(xs, ypx(d))), 1)
            pygame.draw.rect(s, c, (x, 6, 8, 8))
            x += 12 + self.text(s, f'{n} {d[-1]:.2f}{unit}', (x + 12, 3), c, self.small) + 10
        pygame.draw.rect(s, EDGE, (L, T, pw, ph), 1)

    # ------------------------------------------------------------------ panels (draw into the content surface s)
    def p_camera(self, name):
        def draw(s, env, o):
            cam = self.rsu_cam if name == 'rsu' else self.cams.get(name)
            if cam is None or cam.frame is None:
                self.text(s, 'NO SIGNAL' if name != 'rsu' else 'no RSU within 90 m' if env.infra else 'RSUs need --infra',
                          (s.get_width() // 2 - 60, s.get_height() // 2 - 8), DIM, self.bold)
                return 'offline'
            self.cover(s, cam.frame, flip=name.startswith('mirror'))
            self.reticle(s)
            age = (self.sim_now - cam.ts) * 1000 if cam.ts else 0
            self.caption(s, f'{cam.size[0]}×{cam.size[1]}  {1 / cam.tick if cam.tick else 20:.0f} Hz  #{cam.n}  age {age:.0f} ms')
            if name == 'chase':
                self.text(s, f'{env.velocity:5.1f} km/h', (s.get_width() - 12, 12), FG, self.large, right=True)
                b = getattr(env, 'blinker', 0)
                if b and time.time() % 0.8 < 0.4:  # blinking arrow, like the dashboard of a real car
                    self.text(s, '<<' if b < 0 else '>>', (s.get_width() // 2 + (-60 if b < 0 else 40), 12), ORANGE, self.large)
                if getattr(env, 'horn', False):
                    self.text(s, 'HORN', (s.get_width() // 2 - 28, 44), MAG, self.large)
            return 'live'
        return draw

    def p_bev(self, s, env, o):
        cam = self.cams.get('bev')
        w, h = s.get_size()
        mx, my = w // 2, h // 2
        k = self.cover(s, cam.frame) if cam and cam.frame is not None else 1.0
        scale = k * BEV[0] / BEV_M
        t, rot = ego_frame(env)

        def px(dx, dy):
            f, r = rot(dx, dy)
            return mx + r * scale, my - f * scale
        route = [px(wp.transform.location.x - t.location.x, wp.transform.location.y - t.location.y)
                 for wp in env.route_waypoints[env.current_waypoint_index::2][:40]]
        if len(route) > 1:
            pygame.draw.lines(s, (120, 120, 170), False, route, 2)
        near = self.cached('conflicts', env, lambda: conflicts(env))
        for a, p, va, vr, ttc, dist in near:
            danger = dist < 2.5 and 0 < ttc < 3
            u = (mx + p[1] * scale, my - p[0] * scale)
            col = RED if danger else YELLOW if a.type_id.startswith('walker') else ORANGE if a.type_id in MOTORBIKES + BICYCLES else CYAN
            if math.hypot(*va) > 0.5:  # where it will be in 2 s
                pygame.draw.line(s, col, u, (mx + (p[1] + 2 * va[1]) * scale, my - (p[0] + 2 * va[0]) * scale), 2)
            pygame.draw.circle(s, col, (int(u[0]), int(u[1])), 6 if danger else 3, 2 if danger else 0)
            if danger:
                pygame.draw.circle(s, RED, (int(mx + (p[1] + vr[1] * ttc) * scale), int(my - (p[0] + vr[0] * ttc) * scale)), 4)
        if env.infra:
            for r in env.infra.rsus:
                u = px(r['xy'][0] - t.location.x, r['xy'][1] - t.location.y)
                pygame.draw.rect(s, YELLOW, (u[0] - 5, u[1] - 5, 10, 10), 2)
            for ob in env.infra.known.values():
                age = env.infra.t - ob['t']
                u = px(ob['x'] + ob['vx'] * age - t.location.x, ob['y'] + ob['vy'] * age - t.location.y)
                pygame.draw.rect(s, (120, 255, 255), (u[0] - 4, u[1] - 4, 8, 8), 1)
            if env.infra.warning_now() is not None:
                wn = env.infra.warning
                cx, cy = px(wn['x'] - t.location.x, wn['y'] - t.location.y)
                pygame.draw.line(s, RED, (cx - 7, cy - 7), (cx + 7, cy + 7), 3)
                pygame.draw.line(s, RED, (cx - 7, cy + 7), (cx + 7, cy - 7), 3)
        pygame.draw.polygon(s, FG, [(mx, my - 8), (mx - 5, my + 6), (mx + 5, my + 6)])
        for m in (10, 20):  # range rings
            pygame.draw.circle(s, GRID, (mx, my), int(m * scale), 1)
            self.text(s, f'{m} m', (mx + int(m * scale) - 26, my + 2), DIM, self.small)
        return f'{len(near)} users <30 m'

    def p_lidar(self, s, env, o):
        scans = [l.latest for l in env.lidar_objs if l.latest is not None]
        n_cars = len(scans)
        if env.infra:
            scans += [r['lidar'].latest for r in env.infra.rsus if r['lidar'].latest is not None]
        img, marks = self.cached(('lidar', s.get_size()), env, lambda: lidar_render(
            [sc[0] for sc in scans], [sc[1] for sc in scans], env.lidar_objs[0].latest[1], s.get_size()))
        s.blit(pygame.surfarray.make_surface(img), (0, 0))
        for u, v, i in marks:
            if i >= n_cars:
                pygame.draw.rect(s, YELLOW, (u - 5, v - 5, 10, 10), 2)
            else:
                pygame.draw.circle(s, (255, 255, 255) if i == 0 else CYAN, (u, v), 6 if i == 0 else 4, 2)
        pts = sum(len(sc[0]) for sc in scans)
        self.caption(s, f'○ ego  ○ coop car  □ RSU      {pts:,} pts fused')
        return f'{n_cars} veh + {len(scans) - n_cars} RSU'

    def p_semseg(self, s, env, o):
        sem = env.image_obs.reshape(80, 160, 3)[:, :, ::-1]  # buffer order is H x W, BGR
        w, h = s.get_size()
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(sem.swapaxes(0, 1).copy()), (w - 16, (w - 16) // 2)), (8, 8))
        self.text(s, 'semantic segmentation · fov 90 · 160×80 → VAE encoder', (8, (w - 16) // 2 + 14), DIM, self.small)
        self.text(s, f'encode {self.ms["vae"]:.1f} ms', (8, (w - 16) // 2 + 30), CYAN, self.small)
        return 'obs[0:95]'

    def p_vae(self, s, env, o):
        w, h = s.get_size()
        def decode():
            with torch.no_grad():
                rec = self.decoder(o[:LATENT_DIM].unsqueeze(0).to(next(self.decoder.parameters()).device))[0]
            return (rec.permute(2, 1, 0).cpu().numpy() * 255).astype(np.uint8)
        recimg = self.cached('vae', env, decode)
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(recimg), (w - 16, (w - 16) // 2)), (8, 8))
        z = o[:LATENT_DIM].cpu().numpy()
        y0, bw = (w - 16) // 2 + 18, (w - 16) / LATENT_DIM
        mid = y0 + 40
        pygame.draw.line(s, GRID, (8, mid), (w - 8, mid))
        for k, val in enumerate(z):
            hgt = float(np.clip(val, -3, 3)) * 12
            pygame.draw.rect(s, CYAN if val >= 0 else MAG, (8 + k * bw, min(mid, mid - hgt), max(bw - 1, 1), abs(hgt) + 1))
        self.text(s, f'z ∈ R^{LATENT_DIM}   μ {z.mean():+.2f}   σ {z.std():.2f}   |z|max {abs(z).max():.2f}', (8, mid + 42), DIM, self.small)
        return 'decoder'

    def p_radar(self, s, env, o):
        w, h = s.get_size()
        sectors = o[100:100 + COOP_SECTORS].cpu().numpy()
        cx, cy, r = w // 2, h // 2 + 4, min(w, h) // 2 - 26
        n = len(sectors)
        for ring in (0.25, 0.5, 0.75, 1.0):
            pygame.draw.circle(s, GRID, (cx, cy), int(12 + (r - 12) * ring), 1)
        for k in range(n):
            a0, a1 = -math.pi + k * 2 * math.pi / n, -math.pi + (k + 1) * 2 * math.pi / n
            rr = 12 + (r - 12) * sectors[k]
            col = RED if sectors[k] < .15 else ORANGE if sectors[k] < .4 else GREEN
            pygame.draw.polygon(s, col, [(cx, cy)] + [(cx + rr * math.sin(a), cy - rr * math.cos(a)) for a in np.linspace(a0, a1, 4)])
        pygame.draw.polygon(s, FG, [(cx, cy - 9), (cx - 5, cy + 6), (cx + 5, cy + 6)])
        # outer ring: how much each direction drives the pedal right now (now + motion features of that sector)
        _, pedal_s = self.sensitivity(env, o)
        c0 = LATENT_DIM + 5
        sect = pedal_s[c0:c0 + n] + pedal_s[c0 + n + 1:c0 + 2 * n + 1]
        top = sect.max() or 1.0
        for k in range(n):
            a0, a1 = -math.pi + k * 2 * math.pi / n, -math.pi + (k + 1) * 2 * math.pi / n
            wk = sect[k] / top
            if wk > 0.5:  # only the directions that clearly dominate the decision
                pts = [(cx + (r + 6) * math.sin(a), cy - (r + 6) * math.cos(a)) for a in np.linspace(a0, a1, 5)]
                pygame.draw.lines(s, (int(70 + 185 * wk), int(200 * wk), 255), False, pts, int(2 + 6 * wk))
        self.text(s, f'{n} sectors · {COOP_VEHICLES} coop · range 50 m', (6, 4), DIM, self.small)
        self.text(s, 'outer arc: ∂π_pedal/∂sector', (6, h - 16), (120, 160, 255), self.small)
        return f'min {50 * sectors.min():.1f} m'

    def p_coopcf(self, s, env, o):
        cf, c = self.cf, self.coop_stats
        if cf is None:
            self.text(s, 'awaiting the first step…', (10, 10), DIM, self.small)
            return '-'
        self.text(s, 'pedal the policy chooses on this moment, given…', (10, 6), DIM, self.small)
        rows = [('everything', cf['full'], CYAN), ('no shared LiDAR', cf['no_v2v'], ORANGE), ('no infrastructure', cf['no_v2i'], MAG),
                ('ego sensors only', cf['ego'], DIM)]
        for k, (name, mu, col) in enumerate(rows):
            self.text(s, f'{name:18}', (10, 30 + k * 24), FG, self.mono)
            self.hbar(s, 190, 34 + k * 24, 170, mu[1], 'pedal', col)
        d = cf['full'][1] - cf['ego'][1]
        verdict = ('coop makes it BRAKE harder' if d < -0.2 else 'coop lets it GO' if d > 0.2 else 'same decision without coop')
        self.text(s, f'Δ pedal (coop − ego) {d:+.2f}  →  {verdict}', (10, 132), RED if d < -0.2 else GREEN if d > 0.2 else DIM, self.bold)
        n = max(1, c['steps'])
        stats = [('decision changed by coop', f"{100 * c['changed'] / n:5.1f} % of steps"),
                 ('brakes only thanks to coop', f"{c['brake_only_coop']} steps"),
                 ('obstacle seen only via shared LiDAR', f"{100 * c['hidden'] / n:5.1f} % of steps"),
                 ('DENM warning active', f"{100 * c['denm'] / n:5.1f} % of steps"),
                 ('lane gap ego / shared', f"{cf['gap_ego']:4.1f} / {cf['gap_shared']:4.1f} m")]
        for k, (a, b) in enumerate(stats):
            self.text(s, f'{a:34}', (10, 160 + k * 20), DIM, self.small)
            self.text(s, b, (290, 160 + k * 20), FG, self.small)
        return f"{c['steps']} steps"

    def p_radar_cmp(self, s, env, o):
        cf = self.cf
        if cf is None:
            return '-'
        w, h = s.get_size()
        n = COOP_SECTORS
        for k, (title, sect, other) in enumerate((('EGO LIDAR ONLY', cf['sect_ego'], None), ('FUSED (V2V)', cf['sect_full'], cf['sect_ego']))):
            cx, cy, r = w // 4 + k * w // 2, h // 2 + 10, min(w // 4, h // 2) - 30
            for ring in (0.5, 1.0):
                pygame.draw.circle(s, GRID, (cx, cy), int(12 + (r - 12) * ring), 1)
            for i in range(n):
                a0, a1 = -math.pi + i * 2 * math.pi / n, -math.pi + (i + 1) * 2 * math.pi / n
                rr = 12 + (r - 12) * sect[i]
                revealed = other is not None and sect[i] < other[i] - 0.02  # the fused view sees closer here
                col = YELLOW if revealed else RED if sect[i] < .15 else ORANGE if sect[i] < .4 else GREEN
                pygame.draw.polygon(s, col, [(cx, cy)] + [(cx + rr * math.sin(a), cy - rr * math.cos(a)) for a in np.linspace(a0, a1, 4)])
            pygame.draw.polygon(s, FG, [(cx, cy - 8), (cx - 5, cy + 6), (cx + 5, cy + 6)])
            self.text(s, title, (cx - 70, 6), FG, self.small)
        self.text(s, '■ yellow: sector the coop cars see closer than the ego', (8, h - 18), YELLOW, self.small)
        revealed = int((cf['sect_full'] < cf['sect_ego'] - 0.02).sum())
        return f'{revealed} sectors revealed'

    def p_ch_coop(self, s, env, o):
        self.chart(s, ['gap_ego', 'gap_shared', 'ped_full', 'ped_ego'], [DIM, CYAN, GREEN, ORANGE], ['gap ego', 'gap shared', 'π full', 'π ego'],
                   '', -1, COOP_CORRIDOR)
        return '30 s window'

    def p_lidar_ego(self, s, env, o):
        l = env.lidar_objs[0].latest
        if l is None:
            return 'no scan'
        img, marks = self.cached(('lidar_ego', s.get_size()), env, lambda: lidar_render([l[0]], [l[1]], l[1], s.get_size()))
        s.blit(pygame.surfarray.make_surface(img), (0, 0))
        self.caption(s, f'○ ego only     {len(l[0]):,} pts — compare with the fused view on the right')
        return '1 sensor'

    def p_policy(self, s, env, o):
        steer_s, pedal_s = self.sensitivity(env, o)
        self.text(s, f"driver {self.driver.upper()}   mode {self.mode.upper()}   σ_action {getattr(self.agent, 'action_std', 0):.3f}",
                  (10, 6), CYAN if self.policy_ok else ORANGE, self.mono)
        self.hbar(s, 10, 32, 200, self.mean[0], 'μ steer', ORANGE)
        self.hbar(s, 10, 50, 200, self.mean[1], 'μ pedal', GREEN)
        if len(self.mean) > 2:
            self.hbar(s, 280, 32, 120, self.mean[2], 'μ horn', MAG)
        self.hbar(s, 10, 68, 200, np.tanh(self.value / 10), 'V(s)/10', CYAN)
        self.text(s, f'V(s) {self.value:+.3f}   obs dim {100 + COOP_OBS}   actor ∂ via autograd', (10, 86), DIM, self.small)
        groups = group_sensitivity(pedal_s, LATENT_DIM, COOP_SECTORS)
        total = sum(groups.values()) or 1.0
        verb = 'BRAKE' if self.mean[1] < -0.05 else 'ACCEL' if self.mean[1] > 0.05 else 'HOLD'
        self.text(s, f'pedal attribution → {verb}', (10, 108), FG, self.bold)
        for k, (name, val) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1])[:9]):
            y = 130 + k * 16
            pygame.draw.rect(s, GRID, (10, y, 220, 9))
            pygame.draw.rect(s, GREEN if k < 3 else (60, 140, 90), (10, y, int(220 * val / total), 9))
            self.text(s, f'{100 * val / total:5.1f}%  {name}', (240, y - 3), FG if k < 3 else DIM, self.small)
        return 'loaded' if self.policy_ok else 'no model'

    def p_reward(self, s, env, o):
        centering = max(1 - env.distance_from_center / env.max_distance_from_center, 0)
        angle_f = max(1 - abs(env.angle / np.deg2rad(20)), 0)
        r = self.series['reward'][-1] if self.series['reward'] else 0.0
        rows = [('speed', f'{env.velocity:6.2f} km/h', f'band [{env.min_speed:.0f}, {env.target_speed:.0f}] max {env.max_speed:.0f}'),
                ('centering', f'{env.distance_from_center:6.2f} m', f'factor ×{centering:.3f}'),
                ('heading', f'{math.degrees(env.angle):+6.2f} °', f'factor ×{angle_f:.3f}'),
                ('front gap', f'{min(env.front_obstacle, 99):6.2f} m', 'lane corridor (coop LiDAR)'),
                ('red & moving', f'{"YES" if env.red_moving else "no":>6}', 'safety penalty term'),
                ('step reward', f'{r:+6.3f}', f'episode Σ {self.ep_reward:+.1f}'),
                ('waypoint', f'{env.current_waypoint_index:6d}', f'of {len(env.route_waypoints)}')]
        for k, (a, b, c) in enumerate(rows):
            col = (GREEN if r > 0 else RED) if a == 'step reward' else RED if a == 'red & moving' and env.red_moving else FG
            self.text(s, f'{a:13}', (10, 8 + k * 19), DIM, self.mono)
            self.text(s, b, (130, 8 + k * 19), col, self.mono)
            self.text(s, c, (230, 10 + k * 19), DIM, self.small)
        flags = [('collision', bool(env.collision_history)), (f'off-lane > {env.max_distance_from_center:.0f} m', env.distance_from_center > env.max_distance_from_center),
                 (f'speed > {env.max_speed:.0f}', env.velocity > env.max_speed), ('route complete', env.current_waypoint_index >= len(env.route_waypoints) - 2)]
        self.text(s, 'termination', (10, 150), FG, self.bold)
        for k, (name, on) in enumerate(flags):
            self.led(s, (16 + (k % 2) * 230, 178 + (k // 2) * 20), RED if on else (40, 60, 50))
            self.text(s, name, (28 + (k % 2) * 230, 171 + (k // 2) * 20), RED if on else DIM, self.small)
        return f'Σ {self.ep_reward:+.0f}'

    def p_train(self, s, env, o):
        tr = self.train
        el = time.time() - tr['start']
        rate = (tr['timestep'] - tr['steps0']) / el if el > 1 else 0.0
        eta = (tr['target'] - tr['timestep']) / rate if rate > 0 and tr['target'] else 0
        frac = tr['timestep'] / tr['target'] if tr['target'] else 0
        self.text(s, f"MODE  {self.mode.upper()}  {self.cfg['eval_name'] if self.mode == 'eval' else ''}", (10, 6), CYAN, self.large)
        pygame.draw.rect(s, GRID, (10, 40, s.get_width() - 20, 12))
        pygame.draw.rect(s, GREEN, (10, 40, int((s.get_width() - 20) * min(frac, 1)), 12))
        self.text(s, f'{100 * frac:5.1f} %', (s.get_width() - 12, 56), FG, self.small, right=True)
        mem = len(self.agent.memory.rewards) if self.mode == 'train' else 0
        rows = [('timestep', f"{tr['timestep']:,} / {tr['target']:,}" if tr['target'] else f"{tr['timestep']:,}"),
                ('episode', f"{tr['episode']}  (this session {self.episode})"),
                ('avg return', f"{tr['avg']:+.2f}"), ('last return', f"{self.ep_hist[-1]:+.2f}" if self.ep_hist else '-'),
                ('action σ', f"{tr['std']:.3f}"), ('rollout buffer', f'{mem:,} steps'),
                ('PPO updates', f"{tr['updates']} · last {tr['update_s']:.1f} s · {tr['saves']} saved"),
                ('last update', f"KL {tr['ppo']['approx_kl']:.4f} · clip {100 * tr['ppo']['clip_frac']:.0f}% · V {tr['ppo']['value_loss']:.3f} · BC {tr['ppo']['bc_loss']:.3f}"
                 if tr.get('ppo') else '-'),
                ('throughput', f'{rate:.1f} steps/s   RTF {rate * 0.05:.2f}×'),
                ('elapsed · ETA', f'{el / 60:.1f} min · {eta / 60:.0f} min' if eta else f'{el / 60:.1f} min'),
                ('job', self.status or '-'),
                ('queue', ' → '.join(j['mode'] for j in self.cfg['queue'][:6]) or 'empty')]
        for k, (a, b) in enumerate(rows):
            self.text(s, f'{a:15}', (10, 72 + k * 18), DIM, self.mono)
            self.text(s, b, (150, 72 + k * 18), FG, self.mono)
        return self.mode

    def p_v2x(self, s, env, o):
        if not env.infra:
            self.text(s, 'infrastructure disabled · start with --infra', (10, 10), DIM, self.mono)
            return 'off'
        inf, ch = env.infra, env.infra.channel
        v = ch.summary(inf.t)
        e = env.vehicle.get_location()
        self.led(s, (16, 14), GREEN if inf.enabled else RED)
        self.text(s, f"{'ONLINE' if inf.enabled else 'OFFLINE'}  {v['msg_per_s']:5.0f} msg/s  {v['latency_ms']:4.0f} ms  loss {100 * v['loss']:3.0f}%",
                  (28, 7), FG, self.mono)
        self.text(s, f'cfg latency {ch.latency * 1000:.0f}±{ch.jitter * 1000:.0f} ms · loss {ch.loss * 100:.0f}% · range {ch.range:.0f} m · '
                     f'gps σ {inf.gps_sigma:.1f} m', (10, 28), DIM, self.small)
        self.text(s, f'fitted CAM: cars {inf.cam_cars * 100:.0f}% · bikes {inf.cam_bikes * 100:.0f}%', (10, 42), DIM, self.small)
        src = collections.Counter(ob.get('src', '?') for ob in inf.known.values())
        self.text(s, f"known objects {len(inf.known)}  " + '  '.join(f'{k} {n}' for k, n in sorted(src.items())), (10, 60), CYAN, self.mono)
        la = inf.light_ahead
        self.text(s, f'SPaT  {la[1].upper()} · {la[2]:.1f} s left · {la[0]:.0f} m' if la else 'SPaT  no light on route',
                  (10, 78), RED if la and la[1] == 'Red' else YELLOW if la else DIM, self.mono)
        denm = inf.warning_now()
        self.text(s, f"DENM  {inf.warning['kind']} · meets ego in {denm:.1f} s · {inf.warning['from']}" if denm is not None else 'DENM  idle',
                  (10, 96), RED if denm is not None else DIM, self.mono)
        self.text(s, f"{'RSU':6}{'junction':>9}{'dist m':>9}{'pts':>8}", (10, 120), DIM, self.small)
        rs = sorted(inf.rsus, key=lambda r: math.hypot(r['xy'][0] - e.x, r['xy'][1] - e.y))
        for k, r in enumerate(rs[:8]):
            d = math.hypot(r['xy'][0] - e.x, r['xy'][1] - e.y)
            pts = len(r['lidar'].latest[0]) if r['lidar'].latest is not None else 0
            self.led(s, (14, 140 + k * 15), GREEN if d < ch.range else (50, 60, 70))
            self.text(s, f"RSU-{k:<2}{r['junction']:>9}{d:9.0f}{pts:8,}", (22, 134 + k * 15), FG if d < 80 else DIM, self.small)
        return f'{len(inf.rsus)} RSU'

    def p_readout(self, s, env, o):
        ctrl = env.vehicle.get_control()
        loc = env.vehicle.get_location()
        cells = [('SPEED', f'{env.velocity:5.1f}', 'km/h', FG), ('ACCEL', f'{self.accel:+5.2f}', 'm/s²', FG),
                 ('YAW RATE', f'{self.yaw_rate:+5.1f}', '°/s', FG), ('STEER', f'{ctrl.steer:+.3f}', '', ORANGE),
                 ('THROTTLE', f'{ctrl.throttle:.3f}', '', GREEN), ('BRAKE', f'{ctrl.brake:.3f}', '', RED if ctrl.brake > 0 else FG),
                 ('LANE OFF', f'{env.distance_from_center:4.2f}', 'm', FG), ('FRONT GAP', f'{min(env.front_obstacle, 99):4.1f}', 'm', FG),
                 ('MIN TTC', f'{self.worst[0]:.1f}' if self.worst else ' --', 's', RED if self.worst else DIM)]
        cw = (s.get_width() - 12) // 3
        for k, (lab, val, unit, col) in enumerate(cells):
            x, y = 6 + (k % 3) * cw, 4 + (k // 3) * 58
            pygame.draw.rect(s, (10, 12, 17), (x, y, cw - 6, 54))
            pygame.draw.rect(s, GRID, (x, y, cw - 6, 54), 1)
            self.text(s, lab, (x + 6, y + 3), DIM, self.small)
            vw = self.text(s, val, (x + 6, y + 20), col, self.large)
            self.text(s, unit, (x + 10 + vw, y + 30), DIM, self.small)
        self.text(s, f'GNSS x {loc.x:8.2f}  y {loc.y:8.2f}  z {loc.z:5.2f}   yaw {env.vehicle.get_transform().rotation.yaw:+7.1f}°',
                  (8, s.get_height() - 16), DIM, self.small)
        return f'{env.timesteps} steps'

    def p_situations(self, s, env, o):
        denm = env.infra.warning_now() if env.infra else None
        labels = ([(f"DENM {env.infra.warning['kind']} meets ego in {denm:.1f} s ({env.infra.warning['from']})", RED)] if denm is not None else []) \
            + ([(f'Conflict in {self.worst[0]:.1f} s: {self.worst[1]}', RED)] if self.worst else []) + self.cached('sit', env, lambda: detect_situations(env))
        for k, (label, col) in enumerate(labels[:10]):
            self.led(s, (14, 14 + k * 19), col)
            self.text(s, label, (26, 7 + k * 19), col, self.mono)
        sev = 'HIGH' if any(c == RED for _, c in labels) else 'MED' if any(c == ORANGE for _, c in labels) else 'LOW'
        return f'risk {sev}'

    def p_sensors(self, s, env, o):
        self.text(s, f"{'ID':10}{'TYPE':14}{'MOUNT':16}{'CONFIG':22}{'RATE':>8}{'OUTPUT':>14}{'AGE ms':>9}", (24, 4), DIM, self.small)
        rows = [('CAM-00', 'semantic', 'ego roof', '160×80 fov 90', '20 Hz', 'obs', 0 if env.image_obs is not None else None)]
        for k, name in enumerate(CAMS):
            c = self.cams.get(name)
            if c:
                rows.append((META[name][0], 'rgb', name, f'{c.size[0]}×{c.size[1]}', f'{1 / c.tick if c.tick else 20:.0f} Hz', f'#{c.n}',
                             (self.sim_now - c.ts) * 1000 if c.ts else None))
        if self.rsu_cam:
            c = self.rsu_cam
            rows.append(('CAM-07', 'rgb', f'pole J{self.rsu_j}', '640×360', '20 Hz', f'#{c.n}', (self.sim_now - c.ts) * 1000 if c.ts else None))
        for k, l in enumerate(env.lidar_objs):
            rows.append((f'LDR-{k:02d}', 'lidar ray', 'ego roof' if k == 0 else f'coop car {k}', f'{LIDAR_CHANNELS} ch {LIDAR_POINTS_PER_SECOND // 1000}k/s',
                         '20 Hz', f'{len(l.latest[0]):,} pts' if l.latest is not None else '-', 0 if l.latest is not None else None))
        if env.infra:
            for k, r in enumerate(env.infra.rsus):
                rows.append((f'RSU-{k:02d}', 'lidar ray', f"pole J{r['junction']}", '64 ch 200k/s', '20 Hz',
                             f"{len(r['lidar'].latest[0]):,} pts" if r['lidar'].latest is not None else '-', 0 if r['lidar'].latest is not None else None))
        rows.append(('COL-00', 'collision', 'ego', 'event', 'async', f'{self.metrics.collisions} hits', 0))
        lh = max(10, min(14, (s.get_height() - 22) // max(1, len(rows))))
        for k, (i, typ, mount, cfg, rate, outp, age) in enumerate(rows):
            y = 20 + k * lh
            ok = age is not None and age < 400
            self.led(s, (12, y + 6), GREEN if ok else ORANGE if age is not None else RED)
            self.text(s, f'{i:10}{typ:14}{mount:16}{cfg:22}{rate:>8}{outp:>14}{(f"{age:.0f}" if age is not None else "--"):>9}',
                      (24, y), FG if ok else DIM, self.small)
        return f'{len(rows)} devices'

    def p_metrics(self, s, env, o):
        m = self.metrics.summary()
        rows = [('distance', f"{m['km'] * 1000:,.0f} m"), ('collisions', f"{self.metrics.collisions}  ({m['collisions_per_km']:.2f}/km)"),
                ('near-miss', f"{m['near_miss_s_per_km']:.1f} s/km"), ('red-light runs', f"{self.metrics.red_runs}  ({m['red_runs_per_km']:.2f}/km)"),
                ('mean speed', f"{m['mean_speed']:.1f} km/h"), ('episodes', f'{self.metrics.episodes}')]
        for k, (a, b) in enumerate(rows):
            self.text(s, f'{a:15}', (10, 6 + k * 18), DIM, self.mono)
            self.text(s, b, (150, 6 + k * 18), FG, self.mono)
        w, h = s.get_size()
        tx0, tx1, ty = 14, w - 14, h - 34
        pygame.draw.line(s, DIM, (tx0, ty), (tx1, ty), 1)
        for sec in range(0, 61, 10):
            px = tx1 - (tx1 - tx0) * sec / 60
            pygame.draw.line(s, DIM, (px, ty - 3), (px, ty + 3), 1)
            self.text(s, f'-{sec}s' if sec else 'now', (px - 10, ty + 6), DIM, self.small)
        styles = {'denm': (RED, 10), 'chaos': (MAG, 9), 'near_miss': (ORANGE, 5), 'red_light': (YELLOW, 7), 'collision': (RED, 14), 'episode': (FG, 18)}
        for k, (ts, kind, label) in enumerate(self.timeline):
            px = tx1 - (tx1 - tx0) * (self.sim_t - ts) / 60
            col, hgt = styles.get(kind, (DIM, 5))
            pygame.draw.line(s, col, (px, ty - hgt), (px, ty), 3 if kind == 'collision' else 1)
            if label and kind != 'near_miss':
                self.text(s, label, (px - 4, ty - hgt - 13 - 12 * (k % 3)), col, self.small)
        return f"{m['km'] * 1000:.0f} m"

    def p_episodes(self, s, env, o):
        w, h = s.get_size()
        L, R, T, B = 46, 10, 20, 18
        pw, ph = w - L - R, h - T - B
        pygame.draw.rect(s, (10, 12, 17), (L, T, pw, ph))
        hist = self.ep_hist[-60:]
        if not hist:
            self.text(s, 'no finished episode yet', (L + 8, T + 8), DIM, self.small)
            return '0 ep'
        ticks = nice_ticks(min(0, min(hist)), max(0, max(hist)))
        y0, y1 = ticks[0], ticks[-1]
        ypx = lambda v: T + ph * (1 - (v - y0) / (y1 - y0))
        for v in ticks:
            pygame.draw.line(s, GRID, (L, ypx(v)), (L + pw, ypx(v)))
            self.text(s, f'{v:g}', (L - 5, ypx(v) - 6), DIM, self.small, right=True)
        bw = pw / 60
        for k, r in enumerate(hist):
            x = L + k * bw
            pygame.draw.rect(s, GREEN if r >= 0 else RED, (x + 1, min(ypx(0), ypx(r)), max(bw - 2, 1), abs(ypx(r) - ypx(0)) + 1))
        avg = [np.mean(hist[max(0, k - 9):k + 1]) for k in range(len(hist))]
        if len(avg) > 1:
            pygame.draw.lines(s, CYAN, False, [(L + (k + .5) * bw, ypx(a)) for k, a in enumerate(avg)], 2)
        pygame.draw.rect(s, EDGE, (L, T, pw, ph), 1)
        self.text(s, f'■ return   — mean(10) {avg[-1]:+.1f}   last {hist[-1]:+.1f}', (L + 4, 3), CYAN, self.small)
        return f'{len(self.ep_hist)} ep'

    def p_chart(self, name):
        specs = {'ch_speed': (['speed', 'target'], [CYAN, DIM], ['v', 'target'], ' km/h', 0, None),
                 'ch_ctrl': (['steer', 'throttle', 'brake'], [ORANGE, GREEN, RED], ['steer', 'thr', 'brk'], '', -1, 1),
                 'ch_reward': (['reward'], [GREEN], ['r'], '', None, None),
                 'ch_safety': (['front', 'ttc'], [YELLOW, RED], ['gap', 'ttc'], '', 0, COOP_CORRIDOR),
                 'ch_v2x': (['msg', 'lat', 'loss'], [CYAN, MAG, ORANGE], ['msg/s', 'ms', 'loss%'], '', 0, None),
                 'ch_value': (['value', 'pi_pedal', 'pi_steer'], [CYAN, GREEN, ORANGE], ['V', 'π_ped', 'π_str'], '', None, None),
                 'ch_perf': (['step_ms', 'vae_ms', 'pi_ms', 'draw_ms'], [CYAN, MAG, GREEN, ORANGE], ['sim', 'vae', 'π', 'draw'], '', 0, None)}
        keys, colors, names, unit, lo, hi = specs[name]

        def draw(s, env, o):
            self.chart(s, keys, colors, names, unit, lo, hi)
            return '30 s window'
        return draw

    def p_terminal(self, s, env, o):
        w, h = s.get_size()
        pygame.draw.rect(s, (6, 8, 11), (0, 0, w, h))
        rows = self.term_rows()
        for k, (stamp, simt, level, msg) in enumerate(self.term.view(rows)):
            y = 4 + k * 15
            self.text(s, stamp, (6, y), DIM, self.small)
            self.text(s, f'{level:4}', (92, y), LEVEL_COLOR[level], self.small)
            self.text(s, ('$ ' + msg) if level == 'CMD' else msg, (130, y - 1), LEVEL_COLOR[level] if level != 'INFO' else FG, self.mono)
        y = h - 20
        pygame.draw.line(s, EDGE, (0, y - 4), (w, y - 4))
        pw = self.text(s, f'coop@carla:{self.config.split()[0]}$ ', (6, y), GREEN, self.mono)
        cur = '█' if time.time() % 1 < .5 else ' '
        lw = self.text(s, self.line, (6 + pw, y), FG, self.mono)
        cw = self.text(s, self.composing, (6 + pw + lw, y), YELLOW, self.mono) if self.composing else 0
        if self.composing:
            pygame.draw.line(s, YELLOW, (6 + pw + lw, y + 15), (6 + pw + lw + cw, y + 15))  # IME composition, underlined
        self.text(s, cur, (6 + pw + lw + cw, y), FG, self.mono)
        if self.term.scroll:
            self.text(s, f'↑ {self.term.scroll} lines · PgDn', (w - 8, y), ORANGE, self.small, right=True)
        return f'● REC {os.path.basename(self.term.path)}'

    def term_rows(self):
        return (BOTTOM - 20 - 30) // 15

    def p_system(self, s, env, o):
        sim_ok = self.ms['step'] < 300
        cams_ok = all(c.frame is not None for c in self.cams.values())
        lid = sum(l.latest is not None for l in env.lidar_objs)
        denm = env.infra.warning_now() if env.infra else None
        v2x = env.infra.channel.summary(env.infra.t) if env.infra else None
        ctrl = env.vehicle.get_control()
        rows = [('SIM', GREEN if sim_ok else ORANGE, f'sync 20 Hz · step {self.ms["step"]:5.1f} ms', f'RTF {0.05 / max(1e-3, self.step_wall):.2f}×'),
                ('SENSORS', GREEN if cams_ok and lid == len(env.lidar_objs) else ORANGE, f'{len(self.cams) + bool(self.rsu_cam)} cam · {lid}/{len(env.lidar_objs)} lidar',
                 f'{len(env.infra.rsus) if env.infra else 0} RSU'),
                ('PERCEPT', GREEN, f'VAE z{LATENT_DIM} · {self.ms["vae"]:4.1f} ms', 'semseg→z'),
                ('FUSION', GREEN if lid else RED, f'{COOP_SECTORS} sect · obs {100 + COOP_OBS}', f'gap {min(env.front_obstacle, 99):.0f} m'),
                ('V2X', (GREEN if env.infra.enabled else RED) if env.infra else (50, 60, 70),
                 f"{v2x['msg_per_s']:.0f} msg/s · {v2x['latency_ms']:.0f} ms" if v2x else 'disabled', f"loss {100 * v2x['loss']:.0f}%" if v2x else ''),
                ('EDGE', RED if denm is not None else GREEN if env.infra else (50, 60, 70),
                 f"DENM {env.infra.warning['kind']} {denm:.1f} s" if denm is not None else 'conflict monitor idle' if env.infra else 'disabled', ''),
                ('POLICY', GREEN if self.policy_ok else ORANGE, f'{self.driver} · {self.ms["policy"]:4.2f} ms', f'V {self.value:+.2f}'),
                ('CONTROL', RED if ctrl.brake > 0 else GREEN, f'str {ctrl.steer:+.2f} thr {ctrl.throttle:.2f} brk {ctrl.brake:.2f}',
                 ('HORN ' if getattr(env, 'horn', False) else '') + {-1: '<< blink L', 1: 'blink R >>'}.get(getattr(env, 'blinker', 0), '')),
                ('RENDER', GREEN if self.fps > 10 else ORANGE, f'page {self.page} · {self.ms["draw"]:4.1f} ms', f'{self.fps:4.1f} fps'),
                ('LOGGER', GREEN, f'{len(self.term.lines)} lines', 'REC')]
        for k, (name, col, a, b) in enumerate(rows):
            y = 6 + k * 20
            self.led(s, (12, y + 7), col)
            self.text(s, name, (24, y), FG, self.bold)
            self.text(s, a, (104, y + 1), FG, self.small)
            self.text(s, b, (s.get_width() - 8, y + 1), DIM, self.small, right=True)
        return 'nominal' if sim_ok and cams_ok else 'degraded'

    def panel_fn(self, name):
        if name in CAMS or name == 'rsu':
            return self.p_camera(name)
        if name.startswith('ch_') and name != 'ch_coop':
            return self.p_episodes if name == 'ch_episodes' else self.p_chart(name)
        return getattr(self, 'p_' + name)

    # ------------------------------------------------------------------ frame
    def draw(self, env, o):
        t0 = time.time()
        self.loop_s, self.last_frame = t0 - self.last_frame, t0
        self.fps = 0.9 * self.fps + 0.1 / max(1e-3, self.loop_s)  # UI frames per second
        self.sim_now = env.world.get_snapshot().timestamp.elapsed_seconds
        if env.timesteps % 5 == 0:  # keep the CARLA server window on the ego
            t = env.vehicle.get_transform()
            f = t.get_forward_vector()
            env.world.get_spectator().set_transform(carla.Transform(
                t.location + carla.Location(x=-12 * f.x, y=-12 * f.y, z=8), carla.Rotation(pitch=-25, yaw=t.rotation.yaw)))
        s = self.screen
        s.fill(BG)
        # header: identity, run id, configuration, clocks
        pygame.draw.rect(s, HEADC, (0, 0, W, HEAD))
        pygame.draw.line(s, CYAN, (0, HEAD - 1), (W, HEAD - 1))
        x = 10 + self.text(s, 'COOP-DRIVE', (10, 6), CYAN, self.large)
        x += 12 + self.text(s, 'research console', (x + 10, 11), DIM, self.small)
        self.text(s, f'RUN {self.run_id}  ·  {self.config}  ·  ep {self.episode}  step {env.timesteps}  ·  R {self.ep_reward:+.1f}',
                  (x + 16, 8), FG, self.mono)
        right = f"sim {self.sim_t:8.2f} s   {time.strftime('%Y-%m-%d %H:%M:%S')}"
        self.text(s, right, (W - 10, 8), FG, self.mono, right=True)
        q, n = self.cfg['queue'], self.cfg['queue_total'] or len(appcfg.pipeline())
        job = f' {n - len(q) + 1}/{n}' if q and q[0] == self.cfg['job'] else ''
        state, scol = ('PAUSED', ORANGE) if self.paused else (self.mode.upper() + job, GREEN if self.mode == 'drive' else MAG)
        sw = self.text(s, state, (W - 330, 7), scol, self.bold, right=True)
        self.led(s, (W - 342 - sw, 15), scol)
        # page tabs
        for k, name in enumerate(PAGES):
            rx = 6 + k * 130
            on = name == self.page and not self.focus
            pygame.draw.rect(s, (28, 40, 56) if on else PANEL, (rx, HEAD + 3, 126, TABS - 5))
            pygame.draw.rect(s, CYAN if on else EDGE, (rx, HEAD + 3, 126, TABS - 5), 1)
            self.text(s, f'F{k + 1}  {name.upper()}', (rx + 10, HEAD + 6), CYAN if on else DIM, self.small)
        if self.focus:
            self.text(s, f'FOCUS {self.focus} · focus off', (6 + len(PAGES) * 130 + 10, HEAD + 6), ORANGE, self.small)
        self.text(s, f'obs {100 + COOP_OBS} · sectors {COOP_SECTORS} · coop {COOP_VEHICLES} · torch {torch.__version__.split("+")[0]} · '
                     f'{"cuda" if torch.cuda.is_available() else "cpu"}', (W - 10, HEAD + 6), DIM, self.small, right=True)
        layout = [(self.focus, PAGE_AREA)] if self.focus else PAGES[self.page]
        for name, rect in layout + STRIP:
            content = self.frame(name, rect)
            try:
                status = self.panel_fn(name)(content, env, o)
            except Exception as e:  # one broken panel must not take the console down
                status = 'error'
                self.text(content, f'{type(e).__name__}: {e}'[:90], (8, 8), RED, self.small)
            if status:
                self.text(s, status, (rect[0] + rect[2] - 6, rect[1] + 3), GREEN if status in ('live', 'nominal') else DIM, self.small, right=True)
        if self.need_capture:  # one replay frame per simulation step, however often the UI redraws
            self.frames.append(pygame.transform.smoothscale(s, (W // 2, H // 2)))
            self.need_capture = False
        self.present(s)
        self.ms['draw'] = (time.time() - t0) * 1000

    def draw_replay(self):
        '''Half speed playback of the stored frames; returns False when it is over.'''
        f = self.replay[min(self.replay_i // 2, len(self.replay) - 1)]
        frame = pygame.transform.smoothscale(f, (W, H))
        pygame.draw.rect(frame, RED, (0, 0, W, HEAD))
        frame.blit(self.bold.render(f'  REPLAY {self.replay_i // 2 + 1}/{len(self.replay)}  ·  half speed  ·  Esc to stop  ·  gif to save',
                                    True, FG), (10, 8))
        self.present(frame)
        self.replay_i += 1
        if self.replay_i // 2 >= len(self.replay):
            self.replay = None
        return self.replay is not None


def latest_meta(chkpt_dir):
    '''Training counters of the newest checkpoint_ppo_<n>.pickle (same numbering as continuous_driver.py).'''
    n = len(next(os.walk(chkpt_dir))[2]) - 1 if os.path.isdir(chkpt_dir) else -1
    path = f'{chkpt_dir}/checkpoint_ppo_{n}.pickle'
    if n < 0 or not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_meta(chkpt_dir, data, new_file):
    n = len(next(os.walk(chkpt_dir))[2])
    with open(f'{chkpt_dir}/checkpoint_ppo_{n if new_file else max(n - 1, 0)}.pickle', 'wb') as f:
        pickle.dump(data, f)


def cli_config(argv):
    '''Settings for a session started with command-line flags (old shortcuts, scripts): not saved, no restarts.'''
    p = argparse.ArgumentParser()
    p.add_argument('--app', action='store_true', help='settings from coopdrive.json (what CoopDrive.bat uses)')
    p.add_argument('--town', default='Town07')
    p.add_argument('--model', default='', help='policy .pth (default: latest in the ppo_coop folder)')
    p.add_argument('--snapshot', default=None, help='save a PNG of the window every 200 steps')
    p.add_argument('--expert', action='store_true', help='drive with the waypoint expert from bc.py')
    p.add_argument('--vn', action='store_true', help='chaotic Vietnamese-style urban traffic (use with --town Town02)')
    p.add_argument('--infra', action='store_true', help='roadside units + V2X (CAM / CPM / SPaT / DENM) feeding the ego')
    p.add_argument('--fullscreen', action='store_true', help='start fullscreen (F11 toggles)')
    p.add_argument('--collect', type=int, default=0, help='record N expert steps for bc.py train, then exit')
    p.add_argument('--noise', type=float, default=0.2, help='collect: steer noise std (executed, not labelled)')
    p.add_argument('--no-scenarios', dest='scenarios', action='store_false', help='collect: no pedestrians / slow cars')
    p.add_argument('--train', action='store_true', help='fine-tune PPO from the latest checkpoint, then exit')
    p.add_argument('--total-timesteps', type=int, default=100000)
    p.add_argument('--episode-length', type=int, default=3000)
    a = p.parse_args(argv)
    if a.app:
        return appcfg.load(), True, None
    assert not (a.collect and a.train), 'pick one of --collect / --train'
    cfg = dict(appcfg.defaults(), town=a.town, vn=a.vn, infra=a.infra, model=a.model, fullscreen=a.fullscreen,
               mode='collect' if a.collect else 'train' if a.train else 'drive', driver='expert' if a.expert else 'auto',
               collect_steps=a.collect, noise=a.noise, scenarios=a.scenarios, total_timesteps=a.total_timesteps,
               episode_length=a.episode_length)
    return cfg, False, a.snapshot


def main():
    cfg, app, snapshot = cli_config(sys.argv[1:])
    assert COOP_SECTORS, 'dashboard needs COOP_SECTORS > 0 (simulation/settings.py)'
    mode, town = cfg['mode'], cfg['town']
    seed = cfg['eval_seed']
    if mode == 'eval':
        random.seed(seed); np.random.seed(seed)
    client, world = ClientConnection(town).setup()
    if mode == 'eval':
        client.get_trafficmanager(TM_PORT).set_random_device_seed(seed)
    env = CarlaEnvironment(client, world, town, checkpoint_frequency=None, traffic=cfg['vn'], share=cfg['share'], infra=cfg['infra'])
    env.display_on = False  # the dashboard owns the pygame window
    encode = EncodeState(LATENT_DIM)
    chkpt_dir = f'checkpoints/PPO_coop/{town}'
    meta = latest_meta(chkpt_dir) if mode == 'train' else None
    agent = PPOAgent(town, meta['action_std_init'] if meta else 0.05)
    policy_ok, why = False, ''
    try:
        if cfg['model']:
            agent.old_policy.load_state_dict(torch.load(cfg['model']))
            agent.policy.load_state_dict(torch.load(cfg['model']))
        else:
            agent.load()
        policy_ok = True
    except Exception as e:  # no model yet, or trained with another observation size
        why = f'{type(e).__name__}: {e}'
    config = f"{town} {'VN' if cfg['vn'] else 'TM'}{' +INFRA' if cfg['infra'] else ''}{'' if cfg['share'] else ' noshare'} obs {100 + COOP_OBS}"
    dash = Dashboard(agent, policy_ok, mode, config, cfg, app)
    sys.stdout = Tee(sys.stdout, dash.log)
    if not policy_ok:
        dash.log(f'no matching policy ({why[:120]}) · the expert drives', 'WARN')
    expert_job = mode in ('collect', 'bc') or (mode == 'eval' and cfg['eval_expert'])
    dash.driver = 'expert' if expert_job or cfg['driver'] == 'expert' or not policy_ok else 'ppo'
    if cfg['fullscreen']:
        dash.set_fullscreen(True)
    tr, job = dash.train, cfg['job']
    data = proc = None
    eps = []
    if mode in ('train', 'eval') and not expert_job and not policy_ok:
        dash.fail_job('no policy to start from (run collect + bc first)')
        mode = dash.mode = 'drive'
    if mode == 'train':
        os.makedirs(chkpt_dir, exist_ok=True)
        meta = meta or {'cumulative_score': 0, 'episode': 0, 'timestep': 0, 'action_std_init': agent.action_std}
        if 'target' not in job:  # fixed when the job starts, so a restart continues to the same total
            appcfg.update_job(cfg, target=meta['timestep'] + cfg['total_timesteps'])
            dash.save_cfg()
        tr.update(timestep=meta['timestep'], steps0=meta['timestep'], episode=meta['episode'], avg=meta['cumulative_score'],
                  target=job['target'], std=agent.action_std)
        dash.status = f"fine-tune → {job['target']:,} steps"
        bc_actor, bc_path = None, f'data/bc_policy_{town}.pth'
        if cfg['bc_reg'] > 0 and os.path.exists(bc_path):  # anchor: keeps the fine-tune near the BC policy
            net = ActorCritic(100 + COOP_OBS, agent.action_dim, agent.action_std)
            net.load_state_dict(torch.load(bc_path))
            bc_actor = net.actor.eval()
            dash.log(f"BC anchor {bc_path} · coefficient {cfg['bc_reg']}", 'INFO')
        curve = os.path.join('logs', f'train_{dash.run_id}.csv')  # learning curve: one row per PPO update
        with open(curve, 'w') as f:
            f.write('update,episode,timestep,avg_return,last_return,samples,sgd_steps,policy_loss,value_loss,entropy,approx_kl,clip_frac,bc_loss,grad_norm,mean_reward,seconds\n')
        dash.log(f"PPO fine-tune at timestep {meta['timestep']:,} · episode {meta['episode']} · σ {agent.action_std} · "
                 f"target {job['target']:,} · checkpoints {chkpt_dir}", 'INFO')
    if mode == 'collect':
        import bc
        path = bc.DATA.format(town)
        data = ([], [], [], [], [])  # obs, obs_alt, act, rew, done
        if job.get('fresh'):
            if os.path.exists(path):
                os.replace(path, path + '.bak')
                dash.log(f'previous data kept as {path}.bak', 'INFO')
            appcfg.update_job(cfg, fresh=False)  # from now on a restart resumes this collection
            dash.save_cfg()
        elif os.path.exists(path):
            z = np.load(path)
            if z['obs'].shape[1:] == (100 + COOP_OBS,):
                data = tuple(list(z[k]) for k in ('obs', 'obs_alt', 'act', 'rew', 'done'))
                dash.log(f'resuming collection at {len(data[0]):,} steps', 'INFO')
            else:
                dash.log(f"{path} has {z['obs'].shape[1]} inputs, not {100 + COOP_OBS}: starting over", 'WARN')
        tr.update(target=cfg['collect_steps'], timestep=len(data[0]), steps0=len(data[0]))
        dash.status = f"expert data → {path}"
        dash.log(f"expert data collection · {cfg['collect_steps']:,} steps → {path}", 'INFO')
    if mode == 'eval':
        tr.update(target=cfg['eval_episodes'])
        who = 'expert' if expert_job else cfg['model'] or 'latest PPO'
        dash.status = f"{cfg['eval_name']} · {who} · seed {seed}"
        dash.log(f"evaluation {cfg['eval_name']}: {cfg['eval_episodes']} episodes × {cfg['eval_steps']} steps · {who} · "
                 f"share {'on' if cfg['share'] else 'off'}", 'INFO')
    if mode == 'bc':
        cmd = [sys.executable, '-u', 'bc.py', 'train', '--town', town] + (['--noshare'] if job.get('noshare') else [])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        lines = []
        threading.Thread(target=lambda: [lines.append(l.rstrip()) for l in proc.stdout], daemon=True).start()
        dash.status = 'behavior cloning ' + ('(ego + V2I only)' if job.get('noshare') else '(shared LiDAR)')
        dash.log(f"$ {' '.join(cmd[1:])}", 'INFO')
    clock = pygame.time.Clock()

    def step_async(action):
        '''env.step in a worker thread while the UI keeps drawing and reading keys at the configured fps.'''
        box = []

        def work():
            try:
                box.append(env.step(action))
            except Exception as e:  # handed back to the main thread
                box.append(e)
        th = threading.Thread(target=work, daemon=True)
        th.start()
        first = True
        while True:
            # full fps only while the user types / scrolls: drawing holds the GIL and slows the simulation thread
            busy = dash.handle_events(env) or time.time() - dash.last_input < 1.0
            if first or busy or time.time() - dash.last_frame > 0.5:
                dash.draw(env, o)
                first = False
            if not th.is_alive():
                break
            clock.tick(max(5, cfg['fps']))
        th.join()
        if isinstance(box[0], Exception):
            raise box[0]
        return box[0]

    def end_episode_early():
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list], True)
        env.sensor_list.clear(); env.actor_list.clear(); env.riders.clear(); env.remove_sensors()

    retries = 0
    try:
        while not dash.quit:
            if mode == 'eval':
                if len(eps) >= cfg['eval_episodes']:
                    out = evaluate.write(cfg['eval_name'], {k: cfg[k] for k in appcfg.DEFAULTS if k not in ('job', 'queue')}, eps)
                    dash.log('SUMMARY ' + ' · '.join(f'{k} {v}' for k, v in out['summary'].items()), 'INFO')
                    dash.finish_job(f"results/{cfg['eval_name']}.json")
                    break
                k = seed * 1000 + len(eps) + 100003 * retries  # same route + events per episode; a retry moves to the next seed
                random.seed(k); np.random.seed(k % (2 ** 32))
            observation = env.reset()
            if observation is None:
                retries += 1  # the same seed would pick the same occupied spawn point forever
                dash.log(f'reset failed (no free spawn point?), retry {retries}', 'WARN')
                if retries % 3 == 0:
                    for _ in range(20):  # let the traffic move off the spawn points
                        world.tick()
                continue
            retries = 0
            dash.new_episode(env)
            o = encode.process(observation)
            alt = env.coop_obs_other() if data else None
            stats = evaluate.EpisodeStats() if mode == 'eval' else None
            steps = stuck = 0
            done = crashed = False
            reward = 0.0
            while not dash.quit:
                if proc is not None and proc.poll() is not None and not lines:  # behavior cloning finished
                    if proc.returncode == 0:
                        dash.finish_job()
                    else:
                        dash.fail_job(f'bc.py exited with code {proc.returncode}')
                        proc = None
                    break
                while proc is not None and lines:
                    dash.log('bc │ ' + lines.pop(0), 'OUT')
                dash.handle_events(env)
                dash.run_pending(env)
                if dash.quit:
                    break
                if dash.replay:
                    dash.draw_replay()
                    clock.tick(20)
                    continue
                if dash.want_reset:
                    dash.want_reset = False
                    if mode == 'train' and agent.memory.dones:
                        agent.memory.dones[-1] = True
                    if data and data[-1]:
                        data[-1][-1] = True
                    end_episode_early()
                    break
                if dash.paused:
                    dash.draw(env, o)
                    clock.tick(max(5, cfg['fps']))
                    continue
                t0 = time.time()
                if dash.driver == 'expert':
                    action = expert_action(env)
                    executed = action + np.r_[np.random.normal(0, cfg['noise']), np.zeros(len(action) - 1)] if mode == 'collect' else action
                elif mode == 'eval':
                    with torch.no_grad():
                        action = executed = agent.old_policy.actor(o.cpu()).numpy()  # mean action: deterministic
                else:
                    action = executed = agent.get_action(o, train=mode == 'train')
                dash.ms['policy'] = (time.time() - t0) * 1000
                t0 = time.time()
                result = step_async(executed)
                dash.ms['step'] = (time.time() - t0) * 1000
                if result is None:
                    if mode == 'train':  # keep the rollout aligned: drop the action that got no reward
                        for buf in (agent.memory.observation, agent.memory.actions, agent.memory.log_probs):
                            buf.pop()
                        if agent.memory.dones:
                            agent.memory.dones[-1] = True
                    if stats:
                        stats.m['end'] = 'sim_error'
                        eps.append(stats.record(len(eps), dash.hit_ids))
                    dash.log('environment step failed, new episode', 'WARN')
                    break
                observation, reward, done, info = result
                steps += 1
                stuck = stuck + 1 if env.velocity < 1 else 0
                if mode == 'collect':
                    if cfg['scenarios'] and env.timesteps % 250 == 0 and env.front_obstacle == np.inf and env.velocity > 10:
                        spawn_scenario(env, random.choice([1, 2]))
                    done = done or stuck > 400  # 20 s still (e.g. stuck traffic): give up this episode
                    o_np = o.cpu().numpy()
                    for buf, v in zip(data, (o_np, np.concatenate([o_np[:100], alt]), action, reward, done)):
                        buf.append(v)
                    tr['timestep'] = len(data[0])
                    if not done:
                        alt = env.coop_obs_other()
                    if len(data[0]) >= cfg['collect_steps']:
                        done = True
                elif mode == 'train':
                    done = done or steps >= cfg['episode_length']
                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    tr['timestep'] += 1
                    if tr['timestep'] == tr['target'] - 1:
                        agent.chkpt_save()
                elif mode == 'eval':
                    stats.step(env, reward, done, dash.hit_ids)
                    tr['timestep'] = len(eps)
                    if steps >= cfg['eval_steps'] and not done:
                        eps.append(stats.record(len(eps), dash.hit_ids))  # still driving: 'max_steps'
                        dash.log(f"eval ep {len(eps) - 1}: max steps · {stats.m['km'] * 1000:.0f} m", 'INFO')
                        end_episode_early()
                        break
                dash.ep_reward += reward
                crashed = dash.after_step(env, o, reward) or crashed
                if done:
                    break
                t0 = time.time()
                o = encode.process(observation)
                dash.ms['vae'] = (time.time() - t0) * 1000
                if snapshot and env.timesteps % 200 == 0:
                    pygame.image.save(dash.screen, snapshot)
            if not done:
                continue
            # ---- episode finished
            dash.ep_hist.append(dash.ep_reward)
            dash.log(f'episode {dash.episode} end · R {dash.ep_reward:+.1f} · {steps} steps · ' + ('route done' if reward > -10 else 'penalty −10'),
                     'INFO' if reward > -10 else 'WARN')
            dash.mark('episode', 'end')
            env.riders.clear()
            if mode == 'collect':
                data[-1][-1] = True  # cut an unfinished episode for the return computation
                bc.save(town, *data)
                dash.log(f"collected {len(data[0]):,} / {cfg['collect_steps']:,} steps (saved)", 'INFO')
                if len(data[0]) >= cfg['collect_steps']:
                    dash.finish_job(f'{len(data[0]):,} steps')
            elif mode == 'eval':
                e = stats.record(len(eps), dash.hit_ids)
                if e['collisions'] and e['steps'] < 20:  # hit within 1 s of spawning: a spawn overlap, not the driver's doing
                    retries += 1  # same episode again on the next seed
                    dash.log(f"eval ep {e['episode']}: collision at spawn ({e['steps']} steps) · discarded, rerun", 'WARN')
                else:
                    eps.append(e)
                    dash.log(f"eval ep {e['episode']}: {e['end']} · {e['steps']} steps · {e['km'] * 1000:.0f} m · collisions {e['collisions']}", 'INFO')
            elif mode == 'train':
                tr['episode'] += 1
                tr['avg'] = (tr['avg'] * (tr['episode'] - 1) + dash.ep_reward) / tr['episode']
                state = {'cumulative_score': tr['avg'], 'episode': tr['episode'], 'timestep': tr['timestep'], 'action_std_init': agent.action_std}
                if tr['episode'] % 10 == 0:
                    dash.log(f"PPO update on {len(agent.memory.rewards):,} steps…", 'INFO')
                    dash.draw(env, o)
                    t0 = time.time()
                    st = agent.learn(bc_actor=bc_actor, bc_coef=cfg['bc_reg'] if bc_actor is not None else 0.0)
                    tr['ppo'] = st
                    with open(curve, 'a') as f:
                        f.write(','.join(str(round(v, 6)) for v in (tr['updates'] + 1, tr['episode'], tr['timestep'], tr['avg'], dash.ep_reward,
                                st['samples'], st['sgd_steps'], st['policy_loss'], st['value_loss'], st['entropy'], st['approx_kl'],
                                st['clip_frac'], st['bc_loss'], st['grad_norm'], st['mean_reward'], time.time() - t0)) + '\n')
                    agent.chkpt_save()
                    save_meta(chkpt_dir, state, new_file=False)
                    tr['updates'] += 1
                    tr['update_s'] = time.time() - t0
                    tr['saves'] += 1
                    dash.log(f"update {tr['updates']} done in {tr['update_s']:.1f} s · {st['sgd_steps']} SGD steps · KL {st['approx_kl']:.4f} · "
                             f"clip {100 * st['clip_frac']:.0f}% · V loss {st['value_loss']:.3f} · BC {st['bc_loss']:.4f} · avg return {tr['avg']:+.2f}", 'INFO')
                if tr['episode'] % 100 == 0:
                    agent.save()
                    save_meta(chkpt_dir, state, new_file=True)
                    tr['saves'] += 1
                    dash.log(f"new policy file at episode {tr['episode']}", 'INFO')
                if tr['timestep'] >= tr['target']:
                    agent.chkpt_save()
                    save_meta(chkpt_dir, state, new_file=False)
                    dash.finish_job(f"{tr['timestep']:,} steps")
            if (crashed or reward <= -10) and mode == 'drive':
                dash.start_replay()  # instant replay of the last seconds before the episode ended
    except Exception:
        import traceback
        dash.log(traceback.format_exc(), 'ERR')
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        sys.stdout = sys.stdout.stream if isinstance(sys.stdout, Tee) else sys.stdout
        dash.log('session closed', 'INFO')
        dash.term.close()
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list + env.walker_list], True)
        if env.traffic:
            env.traffic.destroy()
        if env.infra:
            env.infra.destroy(client)
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None  # leave the server usable for others
        world.apply_settings(settings)
        pygame.quit()
    sys.exit(dash.exit_code)


if __name__ == '__main__':
    main()
