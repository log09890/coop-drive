'''Destroy leftover vehicles/sensors/walkers (e.g. after a crashed run). Optional arg: map to load, e.g. Town07.'''
import sys; sys.path.insert(0, '.')
from simulation.connection import carla
c = carla.Client('localhost', 2000); c.set_timeout(60)
if len(sys.argv) > 1: c.load_world(sys.argv[1])
w = c.get_world(); a = w.get_actors()
junk = [x for x in a if x.type_id.split('.')[0] in ('vehicle', 'sensor', 'walker', 'controller')]
print(w.get_map().name, 'removing', len(junk))
c.apply_batch_sync([carla.command.DestroyActor(x) for x in junk])
