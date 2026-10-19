import os
import sys
import glob

try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')

import carla
from simulation.settings import PORT, TIMEOUT, HOST, COOP_SECTORS

class ClientConnection:
    def __init__(self, town):
        self.client = None
        self.town = town

    def setup(self):
        try:

            # Connecting to the  Server
            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            # maps/<town>.xodr (e.g. NgaTuSo from osm_map.py): build a world from OpenDRIVE, else a stock CARLA town
            xodr = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maps', self.town + '.xodr')
            if os.path.exists(xodr):
                with open(xodr, encoding='utf-8') as f:
                    text = f.read()
                current = self.client.get_world()
                if current.get_map().name == 'OpenDriveMap' and current.get_map().to_opendrive() == text:
                    self.world = current  # already loaded: generating it a second time crashes the 0.9.10 server
                else:
                    if current.get_map().name == 'OpenDriveMap':  # a different generated map: go through a stock town,
                        self.client.set_timeout(120.0)            # generating over a generated world crashes the server
                        self.client.load_world('Town01')
                    params = carla.OpendriveGenerationParameters()
                    # walls at the road edges: a generated map has no ground, anything leaving the road falls and is destroyed
                    # 3 m / 100 m mesh: the 0.9.10 generator fails less on big maps; walls 3.5 m: a 1-storey "frontage" along the streets
                    params.vertex_distance, params.max_road_length, params.wall_height = 3.0, 100.0, 3.5
                    params.additional_width, params.smooth_junctions, params.enable_mesh_visibility = 0.6, True, True
                    params.enable_pedestrian_navigation = False  # True crashes the 0.9.10 server (tested): scripted crossers instead
                    self.client.set_timeout(180.0)  # building the road mesh takes a while
                    for attempt in range(3):  # the 0.9.10 generator fails at random ("failed to connect to newly created map")
                        try:
                            self.world = self.client.generate_opendrive_world(text, params)
                            break
                        except RuntimeError as e:
                            print(f'map generation failed ({e}), retry {attempt + 1}/3')
                            self.client.get_server_version()  # raises if the server itself died: nothing to retry then
                    else:
                        raise RuntimeError('could not generate ' + xodr)
                    self.client.set_timeout(TIMEOUT)
            else:
                self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            # Coop cars use the Traffic Manager, which is only stable in synchronous mode (as in CODD)
            settings = self.world.get_settings()
            settings.synchronous_mode = bool(COOP_SECTORS)
            settings.fixed_delta_seconds = 0.05 if COOP_SECTORS else None
            self.world.apply_settings(settings)
            self.client.get_trafficmanager(8000).set_synchronous_mode(bool(COOP_SECTORS))
            return self.client, self.world

        except Exception as e:
            print(
                'Failed to make a connection with the server: {}'.format(e))
            self.error()

    # An error method: prints out the details if the client failed to make a connection
    def error(self):

        print("\nClient version: {}".format(
            self.client.get_client_version()))
        print("Server version: {}\n".format(
            self.client.get_server_version()))

        if self.client.get_client_version != self.client.get_server_version:
            print(
                "There is a Client and Server version mismatch! Please install or download the right versions.")
