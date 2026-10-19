'''
Live dashboard for the coop PPO agent (needs COOP_SECTORS > 0 and a running CARLA server).

  python dashboard.py [--town Town02 --vn] [--model path.pth] [--expert]

Press / or Enter to type a command, /help lists them.
'''
import argparse
import collections
import math
import os
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
from simulation.settings import COOP_SECTORS, COOP_CORRIDOR
from encoder_init import EncodeState
from autoencoder.decoder import Decoder
from networks.on_policy.ppo.agent import PPOAgent
from parameters import LATENT_DIM
from bc import expert_action
from insight import closest_approach, input_sensitivity, group_sensitivity, SessionMetrics

W, H, HEAD, CONSOLE = 1280, 840, 28, 118  # canvas; the window is scaled to fit the screen
BIG, CELL = (640, 340), (320, 352)
BG, PANEL, FG, DIM = (12, 12, 16), (22, 22, 30), (230, 230, 230), (130, 130, 140)
RED, ORANGE, YELLOW, GREEN, CYAN = (255, 70, 50), (255, 150, 30), (240, 210, 60), (80, 220, 120), (70, 200, 255)
SCENARIOS = {'ped': 1, 'slow': 2, 'parked': 3}
WEATHER = {'clear': 'ClearNoon', 'cloudy': 'CloudyNoon', 'rain': 'HardRainNoon', 'sunset': 'ClearSunset'}
HELP = [
    ('/spawn <event>', 'ped slow parked ' + ' '.join(EVENTS)),
    ('/chaos on|off|<sec>', 'random surprise events near the ego'),
    ('/density <n>', 'number of motorbikes (kept around the ego)'),
    ('/driver expert|ppo', 'who drives'),
    ('/weather <name>', ' '.join(WEATHER)),
    ('/v2x <key> <value>', 'latency <ms> · loss <%> · range <m> · gps <m> · cars|bikes <% fitted> · on · off'),
    ('/pause', 'freeze / resume the simulation'),
    ('/reset', 'new episode'),
    ('/replay', 'play back the last ~10 s (also automatic after a collision)'),
    ('/gif [file]', 'save the last ~10 s as an animated GIF in replays/'),
    ('/focus <panel>|off', 'enlarge a panel: ' + ' '.join(['camera', 'lidar', 'sensors', 'policy', 'reward', 'traffic'])),
    ('/snapshot [file]', 'save the window as PNG'),
    ('/clear', 'clear the log'),
    ('/quit', 'exit'),
    ('↑ ↓  Tab', 'command history, completion'),
]
PANELS = {'camera': (0, HEAD, *BIG), 'lidar': (BIG[0], HEAD, *BIG), 'sensors': (0, HEAD + BIG[1] + 2, *CELL),
          'policy': (CELL[0], HEAD + BIG[1] + 2, *CELL), 'reward': (2 * CELL[0], HEAD + BIG[1] + 2, *CELL),
          'traffic': (3 * CELL[0], HEAD + BIG[1] + 2, *CELL)}
COMPLETE = {'spawn': sorted(SCENARIOS) + sorted(EVENTS), 'weather': sorted(WEATHER), 'driver': ['expert', 'ppo'],
            'focus': sorted(PANELS) + ['off'], 'chaos': ['on', 'off'], 'v2x': ['latency', 'loss', 'range', 'gps', 'cars', 'bikes', 'on', 'off']}
BEV = (300, 180)  # top-down camera: 40 m up, 70 deg fov -> 56 m wide
BEV_PX_PER_M = BEV[0] / (2 * 40 * math.tan(math.radians(35)))


class ChaseCamera:
    '''RGB camera on the ego (third person by default, or top-down for the BEV); the callback only stores the frame.'''

    def __init__(self, vehicle, size=BIG, transform=None, fov=None):
        bp = vehicle.get_world().get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(size[0]))
        bp.set_attribute('image_size_y', str(size[1]))
        if fov:
            bp.set_attribute('fov', str(fov))
        self.frame = None
        transform = transform or carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15.0))
        self.sensor = vehicle.get_world().spawn_actor(bp, transform, attach_to=vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: ChaseCamera._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if self:
            a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            self.frame = a[:, :, 2::-1].swapaxes(0, 1).copy()  # BGRA -> RGB, (w, h, 3) for pygame


def lidar_render(clouds, poses, ego_pose, size, cam_back=28.0, cam_up=38.0, tilt=55.0, focal=330.0):
    '''CODD-style fused point cloud: oblique perspective from behind/above the ego, red near any sensor -> yellow far.'''
    w, h = size
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


class Dashboard:
    def __init__(self, agent, policy_ok):
        try:  # without this Windows magnifies the window by the display scale (200 %) and it no longer fits
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        pygame.init()
        pygame.key.set_repeat(400, 30)
        info = pygame.display.Info()
        k = min(info.current_w * 0.95 / W, (info.current_h - 120) * 0.95 / H)
        self.window = pygame.display.set_mode((int(W * k), int(H * k)), pygame.RESIZABLE)
        self.screen = pygame.Surface((W, H))  # everything is drawn here at a fixed size
        pygame.display.set_caption('Coop-Drive')
        self.font = pygame.font.SysFont('segoeui,dejavusans', 14)
        self.small = pygame.font.SysFont('segoeui,dejavusans', 12)
        self.big = pygame.font.SysFont('segoeui,dejavusans', 15, bold=True)
        self.mono = pygame.font.SysFont('consolas,dejavusansmono', 14)
        self.agent, self.policy_ok = agent, policy_ok
        self.decoder = Decoder(LATENT_DIM).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.decoder.load()
        self.decoder.eval()
        self.events, self.chase, self.episode, self.reward_hist = [], None, 0, []
        self.driver = 'ppo' if policy_ok else 'expert'
        self.typing, self.line, self.show_help = False, '', False
        self.paused = self.quit = self.want_reset = False
        self.ep_reward = 0.0
        self.history, self.hist_i = [], 0
        self.focus = None
        self.metrics, self.timeline, self.sim_t = SessionMetrics(), [], 0.0  # timeline: (sim time s, kind, label)
        self.frames = collections.deque(maxlen=120)  # ~10+ s of half-size window frames for the instant replay
        self.replay, self.replay_i = None, 0
        self.bev = self.collision = None
        self.hits = []

    def new_episode(self, env):
        self.chase = ChaseCamera(env.vehicle)
        self.bev = ChaseCamera(env.vehicle, BEV, carla.Transform(carla.Location(z=40.0), carla.Rotation(pitch=-90.0)), fov=70)
        bp = env.world.get_blueprint_library().find('sensor.other.collision')
        self.collision = env.world.spawn_actor(bp, carla.Transform(), attach_to=env.vehicle)
        self.hits = []
        self.collision.listen(lambda e: self.hits.append(e.other_actor.type_id))  # read on the main thread in mark()
        env.sensor_list += [self.chase.sensor, self.bev.sensor, self.collision]
        self.episode += 1
        self.ep_reward = 0.0
        self.metrics.episodes += 1
        self.mark('episode', f'ep {self.episode}')
        self.log(f'episode {self.episode}')

    def mark(self, kind, label=''):
        self.timeline = [e for e in self.timeline if e[0] > self.sim_t - 60] + [(self.sim_t, kind, label)]

    def after_step(self, env):
        '''Session metrics, timeline events and collisions for one simulation step.'''
        self.sim_t += 0.05
        w = env.infra.warning if env.infra else None
        if w and w['t'] - getattr(self, 'last_denm', -9) > 1.0:  # one mark per warning burst
            self.last_denm = w['t']
            self.mark('denm', w['kind'])
        for kind in self.metrics.step(env.velocity, env.front_obstacle, env.red_moving):
            self.mark(kind)
        if self.hits:
            what = self.hits[0].split('.')[-1]
            self.hits = []
            self.metrics.collisions += 1
            self.mark('collision', what)
            self.log(f'collision with {what}')
            return True
        return False

    def start_replay(self):
        if len(self.frames) < 5:
            return self.log('nothing to replay yet')
        self.replay, self.replay_i = list(self.frames), 0
        self.log(f'replay {len(self.replay)} frames · Esc to stop')

    def save_gif(self, path=None):
        from PIL import Image
        if len(self.frames) < 5:
            return self.log('nothing to save yet')
        os.makedirs('replays', exist_ok=True)
        path = path or os.path.join('replays', time.strftime('replay_%H%M%S.gif'))
        imgs = [Image.frombytes('RGB', f.get_size(), pygame.image.tostring(f, 'RGB')).resize((480, 315))
                for f in list(self.frames)[::2]]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        self.log(f'saved {path}')

    def log(self, text):
        self.events = ([time.strftime('%H:%M:%S  ') + text] + self.events)[:5]

    def text(self, s, pos, color=FG, font=None, bg=None):
        self.screen.blit((font or self.font).render(s, True, color, bg), pos)

    def panel(self, rect, title):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=6)
        self.text(title, (rect[0] + 8, rect[1] + 4), CYAN, self.big)
        return rect[0] + 8, rect[1] + 26

    def bar(self, x, y, w, value, label, color):
        pygame.draw.rect(self.screen, (50, 50, 60), (x, y, w, 10))
        zero, end = x + w / 2, x + w * (np.clip(value, -1, 1) + 1) / 2
        pygame.draw.rect(self.screen, color, (min(zero, end), y, max(abs(end - zero), 2), 10))
        self.text(f'{label} {value:+.2f}', (x + w + 6, y - 3), FG, self.small)

    # ------------------------------------------------------------------ commands
    def run(self, env, line):
        parts = line.strip().lstrip('/').split()
        if not parts:
            return
        cmd, args = parts[0].lower(), parts[1:]
        self.show_help = cmd == 'help'
        if cmd == 'help':
            return
        if cmd == 'spawn' and args:
            name = args[0].lower()
            if name in SCENARIOS:
                ok = spawn_scenario(env, SCENARIOS[name])
            elif name in EVENTS:
                ok = bool(EVENTS[name](env))
            else:
                return self.log(f'unknown event {name}')
            if ok:
                self.mark('chaos', name)
            return self.log(f'{name}' if ok else f'{name}: no free spot, try again')
        if cmd == 'chaos':
            if not env.traffic:
                return self.log('chaos needs --vn')
            a = args[0].lower() if args else ''
            env.traffic.chaos = 0.0 if a == 'off' else 5.0 if a in ('on', '') else max(1.0, float(a)) if a.replace('.', '', 1).isdigit() else env.traffic.chaos
            env.traffic.next_event = env.traffic.ticks + int(env.traffic.chaos * 20)
            return self.log(f'chaos {"off" if not env.traffic.chaos else f"~every {env.traffic.chaos:.0f} s"}')
        if cmd == 'density' and args and args[0].isdigit():
            if not env.traffic:
                return self.log('density needs --vn')
            return self.log(f'motorbikes {env.traffic.set_motorbikes(env, int(args[0]))}')
        if cmd == 'driver' and args and args[0] in ('expert', 'ppo'):
            if args[0] == 'ppo' and not self.policy_ok:
                return self.log('no matching PPO model')
            self.driver = args[0]
            return self.log(f'driver {self.driver}')
        if cmd == 'weather' and args and args[0] in WEATHER:
            env.world.set_weather(getattr(carla.WeatherParameters, WEATHER[args[0]]))
            return self.log(f'weather {args[0]}')
        if cmd == 'pause':
            self.paused = not self.paused
            return self.log('paused' if self.paused else 'resumed')
        if cmd == 'reset':
            self.want_reset = True
            return
        if cmd == 'v2x':
            if not env.infra:
                return self.log('v2x needs --infra')
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
            return self.log(f"v2x {'on' if env.infra.enabled else 'off'} · {ch.latency * 1000:.0f} ms · loss {ch.loss * 100:.0f}% · "
                            f"range {ch.range:.0f} m · gps {env.infra.gps_sigma:.1f} m · fitted cars {env.infra.cam_cars * 100:.0f}% "
                            f"bikes {env.infra.cam_bikes * 100:.0f}%")
        if cmd == 'replay':
            return self.start_replay()
        if cmd == 'gif':
            return self.save_gif(args[0] if args else None)
        if cmd == 'focus':
            name = args[0].lower() if args else 'off'
            if name != 'off' and name not in PANELS:
                return self.log(f'panels: {" ".join(PANELS)}')
            self.focus = None if name == 'off' else name
            return self.log(f'focus {name}')
        if cmd == 'snapshot':
            path = args[0] if args else time.strftime('snapshot_%H%M%S.png')
            pygame.image.save(self.screen, path)
            return self.log(f'saved {path}')
        if cmd == 'clear':
            self.events = []
            return
        if cmd in ('quit', 'exit'):
            self.quit = True
            return
        self.log(f'unknown: /{" ".join(parts)}  (/help)')

    def complete(self, line):
        '''Tab: complete the command, then its argument; several matches -> longest common prefix.'''
        parts = line.lstrip('/').split(' ')
        if len(parts) == 1:
            options, word = [c.split()[0].lstrip('/') for c, _ in HELP if c.startswith('/')], parts[0]
        else:
            options, word = COMPLETE.get(parts[0], []), parts[-1]
        hits = [o for o in options if o.startswith(word)]
        if not hits:
            return line
        done = os.path.commonprefix(hits) + (' ' if len(hits) == 1 else '')
        if len(hits) > 1:
            self.log('  '.join(hits))
        return '/' + ' '.join(parts[:-1] + [done])

    def handle_events(self, env):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit = True
            elif e.type == pygame.VIDEORESIZE:
                self.window = pygame.display.set_mode(e.size, pygame.RESIZABLE)
            elif e.type == pygame.KEYDOWN and not self.typing:
                if e.key in (pygame.K_SLASH, pygame.K_RETURN, pygame.K_KP_DIVIDE):
                    self.typing, self.line, self.hist_i = True, '/', len(self.history)
                elif e.key == pygame.K_ESCAPE:
                    self.show_help, self.replay = False, None
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.typing = False
                    if self.line.strip('/ '):
                        self.history = [h for h in self.history if h != self.line] + [self.line]
                    self.run(env, self.line)
                elif e.key == pygame.K_ESCAPE:
                    self.typing, self.line = False, ''
                elif e.key in (pygame.K_UP, pygame.K_DOWN) and self.history:
                    self.hist_i = max(0, min(len(self.history), self.hist_i + (-1 if e.key == pygame.K_UP else 1)))
                    self.line = self.history[self.hist_i] if self.hist_i < len(self.history) else '/'
                elif e.key == pygame.K_TAB:
                    self.line = self.complete(self.line)
                elif e.key == pygame.K_BACKSPACE:
                    self.line = self.line[:-1]
                elif e.unicode and e.unicode.isprintable():
                    self.line += e.unicode
        if env.traffic and env.traffic.log:  # surprise events fired by the chaos timer
            for name in env.traffic.log:
                self.log(f'chaos: {name}')
                self.mark('chaos', name)
            env.traffic.log.clear()

    # ------------------------------------------------------------------ drawing
    def draw(self, env, obs, action, reward):
        if env.timesteps % 5 == 0:  # keep the CARLA server window on the ego
            t = env.vehicle.get_transform()
            f = t.get_forward_vector()
            env.world.get_spectator().set_transform(carla.Transform(
                t.location + carla.Location(x=-12 * f.x, y=-12 * f.y, z=8), carla.Rotation(pitch=-25, yaw=t.rotation.yaw)))
        s = self.screen
        s.fill(BG)
        chaos = 'off' if not env.traffic else ('off' if not env.traffic.chaos else f'{env.traffic.chaos:.0f}s')
        self.text(f'COOP-DRIVE   {env.town}   driver {self.driver}   ep {self.episode}   step {env.timesteps}   '
                  f'R {self.ep_reward:.0f}   chaos {chaos}' + ('   PAUSED' if self.paused else ''), (10, 6), FG)

        # 1 camera
        self.panel((0, HEAD, *BIG), '')
        if self.chase and self.chase.frame is not None:
            s.blit(pygame.surfarray.make_surface(self.chase.frame), (0, HEAD))
        self.text(' Camera ', (8, HEAD + 4), CYAN, self.big, BG)

        # 2 cooperative LiDAR (CODD-style)
        scans = [l.latest for l in env.lidar_objs if l.latest is not None]
        n_cars = len(scans)
        if env.infra:
            scans += [r['lidar'].latest for r in env.infra.rsus if r['lidar'].latest is not None]
        img, marks = lidar_render([sc[0] for sc in scans], [sc[1] for sc in scans], env.lidar_objs[0].latest[1], BIG)
        s.blit(pygame.surfarray.make_surface(img), (BIG[0], HEAD))
        for u, v, i in marks:
            if i >= n_cars:
                pygame.draw.rect(s, YELLOW, (BIG[0] + u - 5, HEAD + v - 5, 10, 10), 2)
            else:
                pygame.draw.circle(s, (255, 255, 255) if i == 0 else CYAN, (BIG[0] + u, HEAD + v), 6 if i == 0 else 4, 2)
        rsu = f' + {len(scans) - n_cars} RSU' if env.infra else ''
        self.text(f' Coop LiDAR · {n_cars} cars{rsu} ', (BIG[0] + 8, HEAD + 4), CYAN, self.big, BG)

        row = HEAD + BIG[1] + 2
        sectors = obs[100:100 + COOP_SECTORS].cpu().numpy()
        # 3 sensors
        x, y = self.panel((0, row, *CELL), 'Sensors')
        sem = env.image_obs.reshape(80, 160, 3)[:, :, ::-1]  # buffer order is H x W, BGR
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(sem.swapaxes(0, 1).copy()), (300, 150)), (x, y))
        cx, cy, r = x + 150, y + 236, 70
        n = len(sectors)
        for k in range(n):
            a0, a1 = -math.pi + k * 2 * math.pi / n, -math.pi + (k + 1) * 2 * math.pi / n
            rr = 12 + (r - 12) * sectors[k]
            col = RED if sectors[k] < .15 else ORANGE if sectors[k] < .4 else GREEN
            pygame.draw.polygon(s, col, [(cx, cy)] + [(cx + rr * math.sin(a), cy - rr * math.cos(a)) for a in np.linspace(a0, a1, 4)])
        pygame.draw.circle(s, DIM, (cx, cy), r, 1)
        pygame.draw.polygon(s, FG, [(cx, cy - 9), (cx - 5, cy + 6), (cx + 5, cy + 6)])
        # outer ring: how much each direction drives the pedal right now (now + motion features of that sector)
        steer_s, pedal_s = input_sensitivity(self.agent.old_policy.actor, obs)
        c0 = LATENT_DIM + 5
        sect = pedal_s[c0:c0 + n] + pedal_s[c0 + n + 1:c0 + 2 * n + 1]
        top = sect.max() or 1.0
        for k in range(n):
            a0, a1 = -math.pi + k * 2 * math.pi / n, -math.pi + (k + 1) * 2 * math.pi / n
            w = sect[k] / top
            if w > 0.5:  # only the directions that clearly dominate the decision
                pts = [(cx + (r + 6) * math.sin(a), cy - (r + 6) * math.cos(a)) for a in np.linspace(a0, a1, 5)]
                pygame.draw.lines(s, (int(70 + 185 * w), int(200 * w), 255), False, pts, int(2 + 6 * w))

        # 4 perception + policy
        x, y = self.panel((CELL[0], row, *CELL), 'VAE · PPO')
        with torch.no_grad():
            rec = self.decoder(obs[:LATENT_DIM].unsqueeze(0).to(next(self.decoder.parameters()).device))[0]
            o = obs.cpu().unsqueeze(0)
            mean = self.agent.old_policy.actor(o)[0].numpy()
            value = float(self.agent.old_policy.critic(o)[0])
        recimg = (rec.permute(2, 1, 0).cpu().numpy() * 255).astype(np.uint8)
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(recimg), (300, 150)), (x, y))
        for k, val in enumerate(obs[:LATENT_DIM].cpu().numpy()):
            c = int(np.clip(128 + 40 * val, 0, 255))
            pygame.draw.rect(s, (c, 60, 255 - c), (x + k * 3, y + 160, 3, 16))
        self.bar(x, y + 196, 150, mean[0], 'π steer', ORANGE)
        self.bar(x, y + 216, 150, mean[1], 'π pedal', GREEN)
        groups = group_sensitivity(pedal_s, LATENT_DIM, COOP_SECTORS)
        total = sum(groups.values()) or 1.0
        why = sorted(groups.items(), key=lambda kv: -kv[1])[:3]
        verb = 'brakes' if mean[1] < -0.05 else 'accelerates' if mean[1] > 0.05 else 'holds'
        self.text(f'policy {verb} because of:' if self.policy_ok else 'no policy loaded (expert drives)', (x, y + 236), CYAN, self.small)
        for k, (name, val) in enumerate(why):
            pygame.draw.rect(s, (50, 50, 60), (x, y + 256 + k * 17, 120, 10))
            pygame.draw.rect(s, GREEN, (x, y + 256 + k * 17, int(120 * val / total), 10))
            self.text(f'{name} {100 * val / total:.0f}%', (x + 128, y + 253 + k * 17), FG, self.small)
        self.text(f'V(s) {value:+.2f}', (x, y + 310), FG, self.small)

        # 5 reward logic
        x, y = self.panel((2 * CELL[0], row, *CELL), 'Reward')
        centering = max(1 - env.distance_from_center / env.max_distance_from_center, 0)
        angle_f = max(1 - abs(env.angle / np.deg2rad(20)), 0)
        ctrl = env.vehicle.get_control()
        flags = [('collision', bool(env.collision_history)), (f'off-lane >{env.max_distance_from_center:.0f} m', env.distance_from_center > env.max_distance_from_center),
                 (f'>{env.max_speed:.0f} km/h', env.velocity > env.max_speed), ('stuck 10 s', False)]
        lines = [
            (f'speed {env.velocity:5.1f} km/h   [{env.min_speed:.0f}–{env.target_speed:.0f}]', FG),
            (f'center {env.distance_from_center:4.2f} m   ×{centering:.2f}', FG),
            (f'heading {math.degrees(env.angle):+5.1f}°   ×{angle_f:.2f}', FG),
            (f'reward {reward:+.2f}', GREEN if reward > 0 else RED),
            (f'steer {ctrl.steer:+.2f}  thr {ctrl.throttle:.2f}  brk {ctrl.brake:.2f}', RED if ctrl.brake > 0 else FG),
            (f'wp {env.current_waypoint_index}/{len(env.route_waypoints)}', FG),
        ]
        for k, (line, col) in enumerate(lines):
            self.text(line, (x, y + k * 20), col, self.small)
        for k, (name, on) in enumerate(flags):
            self.text(('● ' if on else '○ ') + name, (x + (k % 2) * 150, y + 128 + (k // 2) * 18), RED if on else DIM, self.small)
        if env.infra:
            v2x = env.infra.channel.summary(env.infra.t)
            ego_l = env.vehicle.get_location()
            near = sum(math.hypot(r['xy'][0] - ego_l.x, r['xy'][1] - ego_l.y) < 50 for r in env.infra.rsus)
            la = env.infra.light_ahead
            infra_lines = [(f"V2X {'on' if env.infra.enabled else 'OFF'}  {v2x['msg_per_s']:.0f} msg/s  {v2x['latency_ms']:.0f} ms  "
                            f"lost {100 * v2x['loss']:.0f}%", CYAN),
                           (f'RSU near {near}/{len(env.infra.rsus)}   objects known {len(env.infra.known)}', CYAN),
                           (f'light ahead {la[1].upper()} {la[2]:.0f} s · {la[0]:.0f} m' if la else 'light ahead: none', RED if la and la[1] == 'Red' else YELLOW if la else DIM)]
            for k, (line, col) in enumerate(infra_lines):
                self.text(line, (x, y + 172 + k * 18), col, self.small)
        self.reward_hist = (self.reward_hist + [reward])[-150:]
        pts = [(x + k * 2, y + 290 - 40 * max(rw, -1)) for k, rw in enumerate(self.reward_hist)]
        if len(pts) > 1:
            pygame.draw.lines(s, GREEN, False, pts, 1)

        # 6 traffic
        x, y = self.panel((3 * CELL[0], row, *CELL), 'Traffic')
        t = env.vehicle.get_transform()
        yaw = math.radians(t.rotation.yaw)
        mx, my, scale = x + BEV[0] // 2, y + BEV[1] // 2, BEV_PX_PER_M
        if self.bev and self.bev.frame is not None:
            s.blit(pygame.surfarray.make_surface(self.bev.frame), (x, y))

        def to_map(l):
            dx, dy = l.x - t.location.x, l.y - t.location.y
            f, rgt = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
            return mx + rgt * scale, my - f * scale

        s.set_clip(pygame.Rect(x, y, *BEV))
        route = [to_map(w.transform.location) for w in env.route_waypoints[env.current_waypoint_index::2][:40]]
        if len(route) > 1:
            pygame.draw.lines(s, (90, 90, 120), False, route, 2)
        # every road user within 30 m: where it will be in 2 s (arrow) and whether it meets the ego within 3 s
        ev = env.vehicle.get_velocity()
        c, sn = math.cos(yaw), math.sin(yaw)
        to_ego = lambda dx, dy: (dx * c + dy * sn, -dx * sn + dy * c)  # world xy -> (forward, right)
        worst = None
        for a in list(env.world.get_actors().filter('vehicle.*')) + list(env.world.get_actors().filter('walker.pedestrian.*')):
            if a.id == env.vehicle.id:
                continue
            l, v = a.get_location(), a.get_velocity()
            p = to_ego(l.x - t.location.x, l.y - t.location.y)
            if math.hypot(*p) > 30:
                continue
            va, vr = to_ego(v.x, v.y), to_ego(v.x - ev.x, v.y - ev.y)
            ttc, dist = closest_approach([p], [vr])
            danger = dist[0] < 2.5 and 0 < ttc[0] < 3
            u = (mx + p[1] * scale, my - p[0] * scale)
            tip = (mx + (p[1] + 2 * va[1]) * scale, my - (p[0] + 2 * va[0]) * scale)
            col = RED if danger else YELLOW if a.type_id.startswith('walker') else ORANGE if a.type_id in MOTORBIKES + BICYCLES else CYAN
            if math.hypot(va[0], va[1]) > 0.5:
                pygame.draw.line(s, col, u, tip, 2)
            pygame.draw.circle(s, col, (int(u[0]), int(u[1])), 6 if danger else 3, 2 if danger else 0)
            if danger:
                meet = (mx + (p[1] + vr[1] * ttc[0]) * scale, my - (p[0] + vr[0] * ttc[0]) * scale)
                pygame.draw.circle(s, RED, (int(meet[0]), int(meet[1])), 4)
                if worst is None or ttc[0] < worst[0]:
                    worst = (ttc[0], a.type_id.split('.')[-1])
        if env.infra:
            for r in env.infra.rsus:
                p = to_ego(r['xy'][0] - t.location.x, r['xy'][1] - t.location.y)
                pygame.draw.rect(s, YELLOW, (mx + p[1] * scale - 5, my - p[0] * scale - 5, 10, 10), 2)
            if env.infra.warning_now() is not None:
                w = env.infra.warning
                p = to_ego(w['x'] - t.location.x, w['y'] - t.location.y)
                cx_, cy_ = mx + p[1] * scale, my - p[0] * scale
                pygame.draw.line(s, RED, (cx_ - 7, cy_ - 7), (cx_ + 7, cy_ + 7), 3)
                pygame.draw.line(s, RED, (cx_ - 7, cy_ + 7), (cx_ + 7, cy_ - 7), 3)
            for o in env.infra.known.values():
                age = env.infra.t - o['t']
                p = to_ego(o['x'] + o['vx'] * age - t.location.x, o['y'] + o['vy'] * age - t.location.y)
                pygame.draw.rect(s, (120, 255, 255), (mx + p[1] * scale - 4, my - p[0] * scale - 4, 8, 8), 1)
        pygame.draw.polygon(s, FG, [(mx, my - 8), (mx - 5, my + 6), (mx + 5, my + 6)])
        s.set_clip(None)
        denm = env.infra.warning_now() if env.infra else None
        labels = ([(f"DENM: {env.infra.warning['kind']} cuts in {denm:.1f} s ({env.infra.warning['from']})", RED)] if denm is not None else [])             + ([(f'Conflict in {worst[0]:.1f} s: {worst[1]}', RED)] if worst else []) + detect_situations(env)
        for k, (label, col) in enumerate(labels[:5]):
            self.text('● ' + label, (x, y + 186 + k * 17), col, self.small)

        # console: log + command line, help overlay
        cy0 = H - CONSOLE
        pygame.draw.rect(s, PANEL, (0, cy0, W, CONSOLE), border_radius=6)
        self.text('Console', (8, cy0 + 4), CYAN, self.big)
        for k, e in enumerate(reversed(self.events[:5])):
            self.text(e[:58], (100, cy0 + 6 + k * 16), FG if k == min(4, len(self.events) - 1) else DIM, self.mono)
        m = self.metrics.summary()
        self.text(f"session {m['km'] * 1000:.0f} m   collisions/km {m['collisions_per_km']:.1f}   near-miss s/km "
                  f"{m['near_miss_s_per_km']:.1f}   red/km {m['red_runs_per_km']:.1f}   v {m['mean_speed']:.1f} km/h",
                  (650, cy0 + 4), FG, self.small)
        tx0, tx1, ty = 650, W - 16, cy0 + 52  # last 60 s, now on the right
        pygame.draw.line(s, DIM, (tx0, ty), (tx1, ty), 1)
        for sec in range(0, 61, 10):
            px = tx1 - (tx1 - tx0) * sec / 60
            pygame.draw.line(s, DIM, (px, ty - 3), (px, ty + 3), 1)
            self.text(f'-{sec}s' if sec else 'now', (px - 10, ty + 6), DIM, self.small)
        styles = {'denm': (RED, 10), 'chaos': (CYAN, 9), 'near_miss': (ORANGE, 5), 'red_light': (YELLOW, 7), 'collision': (RED, 12), 'episode': (FG, 14)}
        for k, (ts, kind, label) in enumerate(self.timeline):
            px = tx1 - (tx1 - tx0) * (self.sim_t - ts) / 60
            col, hgt = styles.get(kind, (DIM, 5))
            pygame.draw.line(s, col, (px, ty - hgt), (px, ty), 3 if kind == 'collision' else 1)
            if label and kind != 'near_miss':
                self.text(label, (px - 4, ty - hgt - 14 - 12 * (k % 2)), col, self.small)
        pygame.draw.rect(s, (8, 8, 12), (8, H - 28, W - 16, 22), border_radius=4)
        prompt = '> ' + self.line + ('_' if time.time() % 1 < .5 else '') if self.typing else '> press / or Enter · /help'
        self.text(prompt, (14, H - 26), FG if self.typing else DIM, self.mono)
        if self.focus:  # enlarge one panel over the dashboard, aspect kept
            rx, ry, rw, rh = PANELS[self.focus]
            area = (W, H - CONSOLE - HEAD - 4)
            k = min(area[0] / rw, area[1] / rh)
            big = pygame.transform.smoothscale(s.subsurface((rx, ry, rw, rh)).copy(), (int(rw * k), int(rh * k)))
            pygame.draw.rect(s, BG, (0, HEAD, W, area[1] + 2))
            s.blit(big, ((W - big.get_width()) // 2, HEAD))
            self.text(f' {self.focus} · /focus off ', (12, HEAD + 4), CYAN, self.big, BG)
        if self.show_help:
            bw, bh = 960, 28 + 22 * len(HELP)
            bx, by = (W - bw) // 2, (H - bh) // 2
            pygame.draw.rect(s, (8, 8, 12), (bx, by, bw, bh), border_radius=8)
            pygame.draw.rect(s, CYAN, (bx, by, bw, bh), 1, border_radius=8)
            for k, (c, d) in enumerate(HELP):
                self.text(c, (bx + 16, by + 14 + 22 * k), CYAN, self.mono)
                self.text(d, (bx + 240, by + 14 + 22 * k), FG, self.mono)
        self.frames.append(pygame.transform.smoothscale(s, (W // 2, H // 2)))
        pygame.transform.smoothscale(s, self.window.get_size(), self.window)
        pygame.display.flip()

    def draw_replay(self):
        '''Half speed playback of the stored frames; returns False when it is over.'''
        f = self.replay[min(self.replay_i // 2, len(self.replay) - 1)]
        frame = pygame.transform.smoothscale(f, (W, H))
        pygame.draw.rect(frame, RED, (0, 0, W, 30))
        frame.blit(self.big.render(f'  REPLAY {self.replay_i // 2 + 1}/{len(self.replay)}  ·  half speed  ·  Esc to stop  ·  /gif to save',
                                   True, FG), (10, 6))
        pygame.transform.smoothscale(frame, self.window.get_size(), self.window)
        pygame.display.flip()
        self.replay_i += 1
        if self.replay_i // 2 >= len(self.replay):
            self.replay = None
        return self.replay is not None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--town', default='Town07')
    p.add_argument('--model', default=None, help='policy .pth (default: latest in the ppo_coop folder)')
    p.add_argument('--snapshot', default=None, help='save a PNG of the window every 200 steps')
    p.add_argument('--expert', action='store_true', help='drive with the waypoint expert from bc.py')
    p.add_argument('--vn', action='store_true', help='chaotic Vietnamese-style urban traffic (use with --town Town02)')
    p.add_argument('--infra', action='store_true', help='roadside units + V2X (CAM / CPM / SPaT) feeding the ego')
    args = p.parse_args()
    assert COOP_SECTORS, 'dashboard needs COOP_SECTORS > 0 (simulation/settings.py)'

    client, world = ClientConnection(args.town).setup()
    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None, traffic=args.vn, infra=args.infra)
    env.display_on = False  # the dashboard owns the pygame window
    encode = EncodeState(LATENT_DIM)
    agent = PPOAgent(args.town, 0.05)
    policy_ok = False
    try:
        agent.old_policy.load_state_dict(torch.load(args.model)) if args.model else agent.load()
        policy_ok = True
    except Exception as e:  # no model yet, or trained with another observation size
        print('No matching policy, the expert drives:', e)
    dash = Dashboard(agent, policy_ok)
    if args.expert:
        dash.driver = 'expert'
    clock = pygame.time.Clock()
    try:
        while not dash.quit:
            observation = env.reset()
            if observation is None:
                continue
            dash.new_episode(env)
            action, reward = np.zeros(2), 0.0
            o = encode.process(observation)
            while not dash.quit:
                dash.handle_events(env)
                if dash.replay:
                    dash.draw_replay()
                    clock.tick(20)
                    continue
                if dash.want_reset:
                    dash.want_reset = False
                    client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list], True)
                    env.sensor_list.clear(); env.actor_list.clear(); env.riders.clear(); env.remove_sensors()
                    break
                if dash.paused:
                    dash.draw(env, o, action, 0.0)
                    clock.tick(20)
                    continue
                action = expert_action(env) if dash.driver == 'expert' else agent.get_action(o, train=False)
                result = env.step(action)
                if result is None:
                    break
                observation, reward, done, info = result
                dash.ep_reward += reward
                crashed = dash.after_step(env)
                if done:
                    dash.log(f'episode {dash.episode} end · R {dash.ep_reward:.0f} · ' + ('route done' if reward > -10 else 'penalty −10'))
                    dash.mark('episode', 'end')
                    env.riders.clear()
                    if crashed or reward <= -10:
                        dash.start_replay()  # instant replay of the last seconds before the episode ended
                    break
                o = encode.process(observation)
                dash.draw(env, o, action, reward)
                if args.snapshot and env.timesteps % 200 == 0:
                    pygame.image.save(dash.screen, args.snapshot)
    finally:
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list + env.walker_list], True)
        if env.traffic:
            env.traffic.destroy()
        if env.infra:
            env.infra.destroy(client)
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None  # leave the server usable for others
        world.apply_settings(settings)
        pygame.quit()


if __name__ == '__main__':
    main()
