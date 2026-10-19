'''Scripted traffic situations on the route ahead of the ego (dashboard keys 1/2/3, BC data collection).'''
from simulation.connection import carla
from simulation.settings import TM_PORT

NAMES = {1: 'Người băng qua đường', 2: 'Xe chạy chậm phía trước', 3: 'Xe đỗ chắn đường'}


def spawn_scenario(env, kind):
    '''Scripted traffic situations placed on the route ahead of the ego. Actors die with the episode.'''
    bl, wps, i = env.blueprint_library, env.route_waypoints, env.current_waypoint_index
    wp = wps[min(i + {1: 35, 2: 25, 3: 30}[kind], len(wps) - 1)].transform  # far enough to see it in the 20 m corridor and stop
    if kind == 1:
        right = wp.get_right_vector()
        tr = carla.Transform(wp.location + carla.Location(x=right.x * 4, y=right.y * 4, z=1.0))
        actor = env.world.try_spawn_actor(bl.filter('walker.pedestrian.*')[0], tr)
        if actor:
            actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(-right.x, -right.y, 0), speed=1.4))
    else:
        tr = carla.Transform(wp.location + carla.Location(z=0.5), wp.rotation)
        actor = env.world.try_spawn_actor(bl.filter('vehicle.audi.a2')[0], tr)
        if actor and kind == 2:
            actor.set_autopilot(True, TM_PORT)
            env.client.get_trafficmanager(TM_PORT).vehicle_percentage_speed_difference(actor, 70)  # 70 % slower
            env.client.get_trafficmanager(TM_PORT).ignore_signs_percentage(actor, 100)  # slow, but never parked at a stop sign
        elif actor:
            actor.apply_control(carla.VehicleControl(hand_brake=True))
    if actor:
        env.actor_list.append(actor)
    return actor is not None
