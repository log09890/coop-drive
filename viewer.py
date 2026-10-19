'''
Passive viewer: watch whatever run is using the server (training, evaluation, benchmark) without disturbing it.
It never ticks and spawns no vehicle; it follows the 'hero' (the ego of the running script) with a chase camera and
a top-down camera, and re-attaches when the ego is respawned after an episode.

  python viewer.py
'''
import math
import time
import numpy as np
import pygame
from simulation.connection import carla
from simulation.settings import HOST, PORT
from dashboard import ChaseCamera, BEV, BIG

W, H = BIG[0] + 2 * BEV[0] + 20, BIG[1] + 40


def find_hero(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.attributes.get('role_name') == 'hero':
            return a
    return None


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Coop-Drive viewer (read-only)')
    font = pygame.font.SysFont('segoeui,dejavusans', 15)
    hero, cams = None, []
    try:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
            try:
                world.wait_for_tick(5.0)  # the running script ticks; we only follow
            except RuntimeError:
                pass
            if hero is None or not hero.is_alive:
                for c in cams:
                    if c.sensor.is_alive:
                        c.sensor.destroy()
                cams, hero = [], find_hero(world)
                if hero:
                    cams = [ChaseCamera(hero),
                            ChaseCamera(hero, (2 * BEV[0], 2 * BEV[1]), carla.Transform(carla.Location(z=40.0), carla.Rotation(pitch=-90.0)), fov=70)]
            screen.fill((12, 12, 16))
            if hero and cams:
                for c, pos in zip(cams, ((0, 30), (BIG[0] + 20, 30))):
                    if c.frame is not None:
                        screen.blit(pygame.surfarray.make_surface(c.frame), pos)
                v = hero.get_velocity()
                info = f'watching hero {hero.id} · {math.hypot(v.x, v.y) * 3.6:4.1f} km/h · {len(world.get_actors().filter("vehicle.*"))} vehicles · Esc to close'
            else:
                info = 'waiting for a running script (no hero vehicle yet)…'
            screen.blit(font.render(info, True, (230, 230, 230)), (10, 6))
            pygame.display.flip()
    finally:
        for c in cams:
            if c.sensor.is_alive:
                c.sensor.destroy()
        pygame.quit()


if __name__ == '__main__':
    main()
