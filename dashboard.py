'''
Live dashboard for the coop PPO agent (needs COOP_SECTORS > 0 and a running CARLA server).

  python dashboard.py                       # latest model in preTrained_models/ppo_coop/<town>
  python dashboard.py --model path/to.pth   # a specific policy

Keys: 1 người băng qua đường, 2 xe chạy chậm phía trước, 3 xe đỗ chắn đường, R episode mới, Esc thoát.
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
from simulation.scenarios import spawn_scenario, NAMES
from simulation.settings import COOP_SECTORS, COOP_RANGE, COOP_CORRIDOR
from encoder_init import EncodeState
from autoencoder.decoder import Decoder
from networks.on_policy.ppo.agent import PPOAgent
from parameters import LATENT_DIM
from bc import expert_action

W, H, HEAD = 1280, 720, 28
BIG, CELL = (640, 340), (320, 352)
BG, PANEL, FG, DIM = (12, 12, 16), (22, 22, 30), (230, 230, 230), (130, 130, 140)
RED, ORANGE, YELLOW, GREEN, CYAN = (255, 70, 50), (255, 150, 30), (240, 210, 60), (80, 220, 120), (70, 200, 255)


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


def detect_situations(env, sectors):
    '''Rule-based labels for what is happening around the ego right now.'''
    out = []
    t = env.vehicle.get_transform()
    loc, fwd = t.location, t.get_forward_vector()
    for a in env.world.get_actors().filter('vehicle.*'):
        if a.id == env.vehicle.id:
            continue
        d = a.get_location() - loc
        ahead = d.x * fwd.x + d.y * fwd.y
        side = abs(-d.x * fwd.y + d.y * fwd.x)
        if 0 < ahead < 25 and side < 2.5:
            speed = math.hypot(a.get_velocity().x, a.get_velocity().y) * 3.6
            out.append((f'Xe phía trước {ahead:.0f} m ({speed:.0f} km/h)', RED if ahead < 10 else ORANGE))
            break
    for a in env.world.get_actors().filter('walker.pedestrian.*'):
        dist = a.get_location().distance(loc)
        if dist < 15:
            out.append((f'Người đi bộ cách {dist:.0f} m', RED if dist < 6 else ORANGE))
            break
    if env.vehicle.is_at_traffic_light():
        out.append((f'Đèn giao thông: {env.vehicle.get_traffic_light_state()}', YELLOW))
    if env.map.get_waypoint(loc).is_junction:
        out.append(('Đang ở giao lộ', YELLOW))
    if env.distance_from_center > 1.5:
        out.append((f'Lệch làn {env.distance_from_center:.1f} m', RED if env.distance_from_center > 2.5 else ORANGE))
    if env.front_obstacle < COOP_CORRIDOR:  # LiDAR, car-wide lane corridor (computed by the env for the obs)
        out.append((f'LiDAR: vật cản trong làn phía trước {env.front_obstacle:.1f} m', RED if env.front_obstacle < 10 else ORANGE))
    return out or [('Đường thông thoáng', GREEN)]


class Dashboard:
    def __init__(self, agent):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption('Coop-Drive dashboard')
        self.font = pygame.font.SysFont('segoeui,dejavusans', 14)
        self.small = pygame.font.SysFont('segoeui,dejavusans', 12)
        self.big = pygame.font.SysFont('segoeui,dejavusans', 16, bold=True)
        self.agent = agent
        self.decoder = Decoder(LATENT_DIM).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.decoder.load()
        self.decoder.eval()
        self.events, self.chase, self.episode = [], None, 0
        self.reward_hist = []
        self.driver = 'PPO'

    def new_episode(self, env):
        self.chase = ChaseCamera(env.vehicle)
        env.sensor_list.append(self.chase.sensor)
        self.episode += 1
        self.ep_reward = 0.0
        self.log(f'Episode {self.episode} bắt đầu')

    def log(self, text):
        self.events = ([time.strftime('%H:%M:%S ') + text] + self.events)[:6]

    def text(self, s, pos, color=FG, font=None, bg=None):
        self.screen.blit((font or self.font).render(s, True, color, bg), pos)

    def panel(self, rect, title):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=6)
        self.text(title, (rect[0] + 8, rect[1] + 4), CYAN, self.big)
        return rect[0] + 8, rect[1] + 26

    def bar(self, x, y, w, value, lo, hi, label, color):
        pygame.draw.rect(self.screen, (50, 50, 60), (x, y, w, 10))
        zero = x + w * (0 - lo) / (hi - lo)
        end = x + w * (np.clip(value, lo, hi) - lo) / (hi - lo)
        pygame.draw.rect(self.screen, color, (min(zero, end), y, max(abs(end - zero), 2), 10))
        self.text(f'{label} {value:+.2f}', (x + w + 6, y - 3), FG, self.small)

    def draw(self, env, obs, action, reward):
        self.ep_reward += reward
        s = self.screen
        s.fill(BG)
        self.text(f'COOP-DRIVE  |  {env.town}  |  lái bởi: {self.driver}  |  ep {self.episode}  |  bước {env.timesteps}  |  '
                  f'reward tích luỹ {self.ep_reward:.1f}  |  phím 1/2/3 kịch bản, R episode mới, Esc thoát', (10, 6), FG)
        sectors = obs[100:100 + COOP_SECTORS].cpu().numpy()

        # 1. Xe chạy
        x, y = self.panel((0, HEAD, *BIG), '1. Xe chạy')
        if self.chase and self.chase.frame is not None:
            s.blit(pygame.surfarray.make_surface(self.chase.frame), (0, HEAD))
            self.text(' 1. Xe chạy ', (8, HEAD + 4), CYAN, self.big, BG)

        # 2. LiDAR hợp tác (CODD-style)
        scans = [l.latest for l in env.lidar_objs if l.latest is not None]
        img, marks = lidar_render([sc[0] for sc in scans], [sc[1] for sc in scans], env.lidar_objs[0].latest[1], BIG)
        s.blit(pygame.surfarray.make_surface(img), (BIG[0], HEAD))
        for u, v, ego in marks:
            pygame.draw.circle(s, (255, 255, 255) if ego else CYAN, (BIG[0] + u, HEAD + v), 6 if ego else 4, 2)
        self.text(f' 2. LiDAR hợp tác: ego + {len(scans) - 1} xe chia sẻ ', (BIG[0] + 8, HEAD + 4), CYAN, self.big, BG)
        self.text('đỏ = gần cảm biến, vàng = xa  |  ○ trắng = ego, ○ xanh = xe hợp tác', (BIG[0] + 8, HEAD + 26), DIM, self.small, BG)

        row = HEAD + BIG[1] + 2
        # 3. Sensor: what the agent sees
        x, y = self.panel((0, row, *CELL), '3. Cảm biến')
        sem = env.image_obs.reshape(80, 160, 3)[:, :, ::-1]  # buffer order is H x W, BGR
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(sem.swapaxes(0, 1).copy()), (300, 150)), (x, y))
        self.text('camera semantic (đầu vào VAE)', (x, y + 152), DIM, self.small)
        cx, cy, r = x + 150, y + 236, 64
        n = len(sectors)
        for k in range(n):
            a0, a1 = -math.pi + k * 2 * math.pi / n, -math.pi + (k + 1) * 2 * math.pi / n
            rr = 12 + (r - 12) * sectors[k]
            col = RED if sectors[k] < .15 else ORANGE if sectors[k] < .4 else GREEN
            pygame.draw.polygon(s, col, [(cx, cy)] + [(cx + rr * math.sin(a), cy - rr * math.cos(a)) for a in np.linspace(a0, a1, 4)])
        pygame.draw.circle(s, DIM, (cx, cy), r, 1)
        pygame.draw.polygon(s, FG, [(cx, cy - 9), (cx - 5, cy + 6), (cx + 5, cy + 6)])
        self.text(f'{n} hướng, bán kính = khoảng trống (tối đa {COOP_RANGE:.0f} m)', (x, y + 306), DIM, self.small)

        # 4. Xử lý: VAE + policy
        x, y = self.panel((CELL[0], row, *CELL), '4. Xử lý (VAE + PPO)')
        with torch.no_grad():
            rec = self.decoder(obs[:LATENT_DIM].unsqueeze(0).to(next(self.decoder.parameters()).device))[0]
            o = obs.cpu().unsqueeze(0)
            mean = self.agent.old_policy.actor(o)[0].numpy()
            value = float(self.agent.old_policy.critic(o)[0])
        recimg = (rec.permute(2, 1, 0).cpu().numpy() * 255).astype(np.uint8)
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(recimg), (300, 150)), (x, y))
        self.text('VAE tái tạo từ latent 95 chiều', (x, y + 152), DIM, self.small)
        z = obs[:LATENT_DIM].cpu().numpy()
        for k, val in enumerate(z):
            c = int(np.clip(128 + 40 * val, 0, 255))
            pygame.draw.rect(s, (c, 60, 255 - c), (x + k * 3, y + 172, 3, 16))
        self.text('latent z (đỏ +, xanh −)', (x, y + 190), DIM, self.small)
        self.bar(x, y + 216, 150, mean[0], -1, 1, 'actor: lái', ORANGE)
        self.bar(x, y + 236, 150, mean[1], -1, 1, 'actor: ga(+)/phanh(−)', GREEN)
        self.bar(x, y + 256, 150, float(action[0]), -1, 1, 'lấy mẫu: lái', ORANGE)
        self.bar(x, y + 276, 150, float(action[1]), -1, 1, 'lấy mẫu: ga/phanh', GREEN)
        self.text(f'critic V(s) = {value:+.2f}', (x, y + 298), FG)

        # 5. Logic: reward + done rules
        x, y = self.panel((2 * CELL[0], row, *CELL), '5. Logic (reward & luật dừng)')
        centering = max(1 - env.distance_from_center / env.max_distance_from_center, 0)
        angle_f = max(1 - abs(env.angle / np.deg2rad(20)), 0)
        ctrl = env.vehicle.get_control()
        lines = [
            (f'Tốc độ {env.velocity:5.1f} km/h  (mục tiêu {env.min_speed:.0f}–{env.target_speed:.0f}, dừng nếu > {env.max_speed:.0f})', FG),
            (f'Lệch tâm làn {env.distance_from_center:4.2f} m  → hệ số {centering:.2f}', FG),
            (f'Lệch hướng {math.degrees(env.angle):+5.1f}°  → hệ số {angle_f:.2f}', FG),
            (f'Reward bước này {reward:+.2f}', GREEN if reward > 0 else RED),
            (f'Điều khiển thực: lái {ctrl.steer:+.2f}, ga {ctrl.throttle:.2f}, phanh {ctrl.brake:.2f}', RED if ctrl.brake > 0 else FG),
            (f'Waypoint {env.current_waypoint_index}/{len(env.route_waypoints)}', FG),
            ('Luật dừng episode (reward −10):', CYAN),
            (f'  va chạm: {"CÓ" if env.collision_history else "không"}', RED if env.collision_history else DIM),
            (f'  lệch làn > {env.max_distance_from_center} m: {"CÓ" if env.distance_from_center > env.max_distance_from_center else "không"}', DIM),
            (f'  quá tốc > {env.max_speed:.0f} km/h: {"CÓ" if env.velocity > env.max_speed else "không"}', DIM),
            ('  đứng yên > 10 s (trừ khi có vật cản < 10 m), hết 7500 bước', DIM),
        ]
        for k, (line, col) in enumerate(lines):
            self.text(line, (x, y + k * 19), col, self.small)
        self.reward_hist = (self.reward_hist + [reward])[-150:]
        pts = [(x + k * 2, y + 300 - 25 * max(rw, -1)) for k, rw in enumerate(self.reward_hist)]
        if len(pts) > 1:
            pygame.draw.lines(s, GREEN, False, pts, 1)

        # 6. Tình huống giao thông: map + labels + scenarios
        x, y = self.panel((3 * CELL[0], row, *CELL), '6. Tình huống giao thông')
        t = env.vehicle.get_transform()
        yaw = math.radians(t.rotation.yaw)
        mx, my, scale = x + 150, y + 105, 1.6

        def to_map(l):
            dx, dy = l.x - t.location.x, l.y - t.location.y
            f, rgt = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
            return mx + rgt * scale, my - f * scale

        clip = pygame.Rect(x - 4, y, 312, 200)
        s.set_clip(clip)
        pygame.draw.lines(s, (70, 70, 90), False, [to_map(w.transform.location) for w in env.route_waypoints[::4]], 3)
        for a in env.world.get_actors().filter('vehicle.*'):
            if a.id != env.vehicle.id:
                pygame.draw.circle(s, CYAN, to_map(a.get_location()), 4)
        for a in env.world.get_actors().filter('walker.pedestrian.*'):
            pygame.draw.circle(s, YELLOW, to_map(a.get_location()), 3)
        pygame.draw.polygon(s, FG, [(mx, my - 7), (mx - 4, my + 5), (mx + 4, my + 5)])
        s.set_clip(None)
        for k, (label, col) in enumerate(detect_situations(env, sectors)[:4]):
            self.text('● ' + label, (x, y + 206 + k * 17), col, self.small)
        for k, e in enumerate(self.events[:4]):
            self.text(e, (x, y + 278 + k * 15), DIM, self.small)
        pygame.display.flip()

    def keys(self, env):
        '''Returns False to quit, 'reset' for a new episode, else None.'''
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                return 'reset'
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                kind = e.key - pygame.K_0
                name = NAMES[kind]
                self.log(('Kịch bản: ' if spawn_scenario(env, kind) else 'Không spawn được: ') + name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--town', default='Town07')
    p.add_argument('--model', default=None, help='policy .pth (default: latest in the ppo_coop folder)')
    p.add_argument('--snapshot', default=None, help='save a PNG of the window every 200 steps')
    p.add_argument('--expert', action='store_true', help='drive with the waypoint expert from bc.py instead of the policy')
    args = p.parse_args()
    assert COOP_SECTORS, 'dashboard needs COOP_SECTORS > 0 (simulation/settings.py)'

    client, world = ClientConnection(args.town).setup()
    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None)
    env.display_on = False  # the dashboard owns the pygame window
    encode = EncodeState(LATENT_DIM)
    agent = PPOAgent(args.town, 0.05)
    driver = 'tài xế mẫu (bám waypoint)' if args.expert else 'PPO'
    if not args.expert:
        try:
            agent.old_policy.load_state_dict(torch.load(args.model)) if args.model else agent.load()
        except Exception as e:  # no model yet, or trained with another observation size
            print('Could not load the policy, driving with the expert instead:', e)
            driver = 'tài xế mẫu (chưa có model khớp)'
    dash = Dashboard(agent)
    dash.driver = driver
    try:
        while True:
            observation = env.reset()
            if observation is None:
                continue
            dash.new_episode(env)
            done = False
            while not done:
                o = encode.process(observation)
                action = expert_action(env) if driver != 'PPO' else agent.get_action(o, train=False)
                result = env.step(action)
                if result is None:
                    break
                observation, reward, done, info = result
                if done:
                    dash.log(f'Episode {dash.episode} kết thúc: {dash.ep_reward + reward:.0f} điểm, '
                             f'{"hoàn thành tuyến" if reward > -10 else "phạt −10 (va chạm / lệch làn / quá tốc / đứng yên)"}')
                    break
                dash.draw(env, encode.process(observation), action, reward)
                if args.snapshot and env.timesteps % 200 == 0:
                    pygame.image.save(dash.screen, args.snapshot)
                k = dash.keys(env)
                if k is False:
                    return
                if k == 'reset':
                    client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list], True)
                    env.sensor_list.clear(); env.actor_list.clear(); env.remove_sensors()
                    break
    finally:
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list + env.walker_list], True)
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None  # leave the server usable for others
        world.apply_settings(settings)
        pygame.quit()


if __name__ == '__main__':
    main()
