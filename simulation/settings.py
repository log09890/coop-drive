'''
All the parameters used in the Simulation has been documented here.

Easily modifiable paramters with the quick access in this settings.py file \
    to achieve quick modifications especially during the training sessions.

Names of the parameters are self-explanatory therefore elimating the use of further comments.
'''


HOST = "localhost"
PORT = 2000
TIMEOUT = 20.0

CAR_NAME = 'model3'
EPISODE_LENGTH = 120
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = False  # upstream pygame chase view; dashboard.py replaces it


RGB_CAMERA = 'sensor.camera.rgb'
SSC_CAMERA = 'sensor.camera.semantic_segmentation'



# Cooperative lidar (CODD-style). COOP_SECTORS = 0 turns it off and keeps the original 100-dim obs (pretrained PPO).
COOP_SECTORS = 16
COOP_VEHICLES = 5
COOP_RANGE = 50.0
COOP_CORRIDOR = 20.0  # lane-ahead obstacle check length (m), one extra obs value
COOP_OBS = 2 * (COOP_SECTORS + 1) if COOP_SECTORS else 0  # (sectors + corridor) now, and their change since the last step
LIDAR_CHANNELS = 32
LIDAR_LOWER_FOV = -25.0
LIDAR_POINTS_PER_SECOND = 100000
