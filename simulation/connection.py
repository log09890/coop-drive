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
from simulation.settings import PORT, TIMEOUT, HOST, COOP_SECTORS, TM_PORT

class ClientConnection:
    def __init__(self, town):
        self.client = None
        self.town = town

    def setup(self):
        try:

            # Connecting to the  Server
            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            # Coop cars use the Traffic Manager, which is only stable in synchronous mode (as in CODD)
            settings = self.world.get_settings()
            settings.synchronous_mode = bool(COOP_SECTORS)
            settings.fixed_delta_seconds = 0.05 if COOP_SECTORS else None
            self.world.apply_settings(settings)
            self.client.get_trafficmanager(TM_PORT).set_synchronous_mode(bool(COOP_SECTORS))
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
