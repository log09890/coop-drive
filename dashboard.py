'''
Live dashboard for the coop PPO agent (needs COOP_SECTORS > 0 and a running CARLA server).

  python dashboard.py [--town Town02 --vn] [--model path.pth] [--expert]

Press / or Enter to type a command, /help lists them.
'''
import argparse
import math
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
    ('/pause', 'freeze / resume the simulation'),
    ('/reset', 'new episode'),
    ('/snapshot [file]', 'save the window as PNG'),
    ('/clear', 'clear the log'),
    ('/quit', 'exit'),
]


class ChaseCamera:
    '''Third person RGB camera; the callback only stores the frame (pygame stays on the main thread).'''

    def __init__(self, vehicle):
        bp = vehicle.get_world().get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(BIG[0]))
        bp.set_attribute('image_size_y', str(BIG[1]))
        self.frame = None
        self.sensor = vehicle.get_world().spawn_actor(
            bp, carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15.0)), attach_to=vehicle)
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
    return img, [(int(a), int(b), i == 0) for i, (a, b, o) in enumerate(zip(su, sv, sok)) if o]


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

    def new_episode(self, env):
        self.chase = ChaseCamera(env.vehicle)
        env.sensor_list.append(self.chase.sensor)
        self.episode += 1
        self.ep_reward = 0.0
        self.log(f'episode {self.episode}')

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

    def handle_events(self, env):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit = True
            elif e.type == pygame.VIDEORESIZE:
                self.window = pygame.display.set_mode(e.size, pygame.RESIZABLE)
            elif e.type == pygame.KEYDOWN and not self.typing:
                if e.key in (pygame.K_SLASH, pygame.K_RETURN, pygame.K_KP_DIVIDE):
                    self.typing, self.line = True, '/'
                elif e.key == pygame.K_ESCAPE:
                    self.show_help = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.typing = False
                    self.run(env, self.line)
                elif e.key == pygame.K_ESCAPE:
                    self.typing, self.line = False, ''
                elif e.key == pygame.K_BACKSPACE:
                    self.line = self.line[:-1]
                elif e.unicode and e.unicode.isprintable():
                    self.line += e.unicode
        if env.traffic and env.traffic.log:  # surprise events fired by the chaos timer
            for name in env.traffic.log:
                self.log(f'chaos: {name}')
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
        img, marks = lidar_render([sc[0] for sc in scans], [sc[1] for sc in scans], env.lidar_objs[0].latest[1], BIG)
        s.blit(pygame.surfarray.make_surface(img), (BIG[0], HEAD))
        for u, v, ego in marks:
            pygame.draw.circle(s, (255, 255, 255) if ego else CYAN, (BIG[0] + u, HEAD + v), 6 if ego else 4, 2)
        self.text(f' Coop LiDAR · {len(scans)} sensors ', (BIG[0] + 8, HEAD + 4), CYAN, self.big, BG)

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
        self.bar(x, y + 244, 150, float(action[0]), 'a steer', ORANGE)
        self.bar(x, y + 264, 150, float(action[1]), 'a pedal', GREEN)
        self.text(f'V(s) {value:+.2f}', (x, y + 292), FG)

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
        self.reward_hist = (self.reward_hist + [reward])[-150:]
        pts = [(x + k * 2, y + 290 - 40 * max(rw, -1)) for k, rw in enumerate(self.reward_hist)]
        if len(pts) > 1:
            pygame.draw.lines(s, GREEN, False, pts, 1)

        # 6 traffic
        x, y = self.panel((3 * CELL[0], row, *CELL), 'Traffic')
        t = env.vehicle.get_transform()
        yaw = math.radians(t.rotation.yaw)
        mx, my, scale = x + 150, y + 95, 1.6

        def to_map(l):
            dx, dy = l.x - t.location.x, l.y - t.location.y
            f, rgt = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
            return mx + rgt * scale, my - f * scale

        s.set_clip(pygame.Rect(x - 4, y, 312, 180))
        route = [to_map(w.transform.location) for w in env.route_waypoints[::4]]
        if len(route) > 1:
            pygame.draw.lines(s, (70, 70, 90), False, route, 3)
        riders = {r.v.id for r in env.riders if r.v}
        for a in env.world.get_actors().filter('vehicle.*'):
            if a.id != env.vehicle.id:
                bike = a.type_id in MOTORBIKES + BICYCLES
                pygame.draw.circle(s, RED if a.id in riders else ORANGE if bike else CYAN, to_map(a.get_location()), 3 if bike else 4)
        for a in env.world.get_actors().filter('walker.pedestrian.*'):
            pygame.draw.circle(s, YELLOW, to_map(a.get_location()), 2)
        pygame.draw.polygon(s, FG, [(mx, my - 7), (mx - 4, my + 5), (mx + 4, my + 5)])
        s.set_clip(None)
        for k, (label, col) in enumerate(detect_situations(env)[:5]):
            self.text('● ' + label, (x, y + 186 + k * 17), col, self.small)

        # console: log + command line, help overlay
        cy0 = H - CONSOLE
        pygame.draw.rect(s, PANEL, (0, cy0, W, CONSOLE), border_radius=6)
        self.text('Console', (8, cy0 + 4), CYAN, self.big)
        for k, e in enumerate(reversed(self.events[:5])):
            self.text(e, (100, cy0 + 6 + k * 16), FG if k == min(4, len(self.events) - 1) else DIM, self.mono)
        pygame.draw.rect(s, (8, 8, 12), (8, H - 28, W - 16, 22), border_radius=4)
        prompt = '> ' + self.line + ('_' if time.time() % 1 < .5 else '') if self.typing else '> press / or Enter · /help'
        self.text(prompt, (14, H - 26), FG if self.typing else DIM, self.mono)
        if self.show_help:
            bw, bh = 960, 28 + 22 * len(HELP)
            bx, by = (W - bw) // 2, (H - bh) // 2
            pygame.draw.rect(s, (8, 8, 12), (bx, by, bw, bh), border_radius=8)
            pygame.draw.rect(s, CYAN, (bx, by, bw, bh), 1, border_radius=8)
            for k, (c, d) in enumerate(HELP):
                self.text(c, (bx + 16, by + 14 + 22 * k), CYAN, self.mono)
                self.text(d, (bx + 240, by + 14 + 22 * k), FG, self.mono)
        pygame.transform.smoothscale(s, self.window.get_size(), self.window)
        pygame.display.flip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--town', default='Town07')
    p.add_argument('--model', default=None, help='policy .pth (default: latest in the ppo_coop folder)')
    p.add_argument('--snapshot', default=None, help='save a PNG of the window every 200 steps')
    p.add_argument('--expert', action='store_true', help='drive with the waypoint expert from bc.py')
    p.add_argument('--vn', action='store_true', help='chaotic Vietnamese-style urban traffic (use with --town Town02)')
    args = p.parse_args()
    assert COOP_SECTORS, 'dashboard needs COOP_SECTORS > 0 (simulation/settings.py)'

    client, world = ClientConnection(args.town).setup()
    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None, traffic=args.vn)
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
                if done:
                    dash.log(f'episode {dash.episode} end · R {dash.ep_reward:.0f} · ' + ('route done' if reward > -10 else 'penalty −10'))
                    env.riders.clear()
                    break
                o = encode.process(observation)
                dash.draw(env, o, action, reward)
                if args.snapshot and env.timesteps % 200 == 0:
                    pygame.image.save(dash.screen, args.snapshot)
    finally:
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list + env.walker_list], True)
        if env.traffic:
            env.traffic.destroy()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None  # leave the server usable for others
        world.apply_settings(settings)
        pygame.quit()


if __name__ == '__main__':
    main()
