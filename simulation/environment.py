import time
import traceback
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor, LidarSensor
from simulation.coop import sector_distances, corridor_distance, safety_penalty
from simulation.settings import *


class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True, traffic=False, share=True, infra=False) -> None:
        self.share = share  # False: the coop features use the ego LiDAR only (ablation)
        self.prev_coop, self.last_sectors, self.prev_pedal = {}, np.ones(max(COOP_SECTORS, 1)), 0.0


        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.lidar_objs = list()
        self.front_obstacle = np.inf
        self.last_blocked = 0.0
        self.d_steer = self.d_pedal = 0.0
        self.blinker, self.horn, self.horns = 0, False, 0  # -1 left, 1 right; horn this step; honks this episode
        self.vehicle_xy, self.vehicle_ids = np.zeros((0, 2)), np.zeros(0, int)
        self.red_moving = False

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()
        # traffic=True: chaotic Vietnamese-style background traffic (motorbikes, jaywalkers), kept across episodes
        self.traffic = None
        self.riders = []
        # infra=True: roadside units at the junctions + V2X (CAM / CPM / SPaT) feed the ego (simulation/infrastructure.py)
        self.infra = None
        if infra and COOP_SECTORS:
            from simulation.infrastructure import Infrastructure
            self.infra = Infrastructure(world)
        self.vehicle_locs = []
        self.vehicle_xy, self.vehicle_ids, self.vehicle_ext = np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2))
        if traffic:
            from simulation.vn_traffic import VNTraffic
            spawns = self.map.get_spawn_points()
            ego_spawn = spawns[38] if town == "Town07" else spawns[1] if town == "Town02" else None
            self.traffic = VNTraffic(client, world, avoid=ego_spawn.location if ego_spawn else None)



    # A reset function for reseting our environment.
    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()


            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)
            vehicle_bp.set_attribute('role_name', 'hero')  # centre of the TM hybrid-physics radius

            if self.town == "Town07":
                transform = self.map.get_spawn_points()[38] #Town7  is 38 
                self.total_distance = 750
            elif self.town == "Town02":
                transform = self.map.get_spawn_points()[1] #Town2 is 1
                self.total_distance = 780
                if self.traffic:  # VN mode (always a fresh start): a new random route every time
                    transform = random.choice(self.map.get_spawn_points())
            else:
                self.total_distance = 250
                transform = self.long_route_spawn(150)  # not right before a dead end

            # spawn just above the asphalt: a higher drop reads as a collision + overspeed (3D velocity) at the start
            transform.location.z = self.map.get_waypoint(transform.location).transform.location.z + 0.5
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            for attempt in range(100):  # background traffic may be passing over the spawn point: let it clear
                if self.vehicle is not None or not COOP_SECTORS:
                    break
                self.world.tick()
                if self.traffic and attempt % 10 == 9:  # VN mode: that spot stays busy, try another one
                    transform = random.choice(self.map.get_spawn_points())
                    transform.location.z = self.map.get_waypoint(transform.location).transform.location.z + 0.5
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if self.vehicle is None:
                raise RuntimeError('could not spawn the ego vehicle')  # never put None in actor_list: DestroyActor(None) segfaults
            self.actor_list.append(self.vehicle)


            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            self.wait_for(lambda: len(self.camera_obj.front_camera))
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22 #km/h
            self.max_speed = 25.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.red_wait = 0.0
            self.blinker, self.horn, self.horns = 0, False, 0
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0


            if self.checkpoint_frequency is None or self.traffic:
                self.fresh_start = True  # no checkpoints (eval, BC data, dense VN traffic): never teleport into traffic
            if self.fresh_start:
                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.traffic:  # random turn at every junction (VN mode)
                        options = current_waypoint.next(1.0)
                        if not options:
                            break
                        next_waypoint = random.choice(options) if len(options) > 1 else options[0]
                    elif self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                transform.location.z += 0.3  # a waypoint sits on the asphalt: teleporting there makes the car pop up
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            # Cooperative cars (CODD): autopilot NPCs around the ego, each sharing its lidar scan
            if COOP_SECTORS:
                self.lidar_objs.append(LidarSensor(self.vehicle))
                spawn_points = [sp for sp in self.map.get_spawn_points() if sp.location.distance(transform.location) < COOP_RANGE]
                random.shuffle(spawn_points)
                for sp in spawn_points[:COOP_VEHICLES]:
                    car = self.world.try_spawn_actor(random.choice(self.blueprint_library.filter('vehicle.*')), sp)
                    if car is None:
                        continue
                    car.set_autopilot(True, TM_PORT)
                    tm = self.client.get_trafficmanager(TM_PORT)
                    tm.ignore_signs_percentage(car, 100)  # 0.9.10 TM can wait forever at Town07 stop signs and block the route
                    tm.ignore_lights_percentage(car, 100)
                    self.actor_list.append(car)
                    self.lidar_objs.append(LidarSensor(car))
                self.sensor_list += [l.sensor for l in self.lidar_objs]
                self.wait_for(lambda: self.lidar_objs[0].latest is not None)

            self.prev_coop = {}  # motion features start at zero
            if self.infra:
                self.infra.reset()
            self.navigation_obs = np.concatenate([np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle]), self.coop_obs()])

                        
            time.sleep(0.5)
            if COOP_SECTORS:
                for _ in range(10):  # let the spawned cars settle
                    self.world.tick()
            self.collision_history.clear()

            self.episode_start_time = self.now()
            return [self.image_obs, self.navigation_obs]

        except:
            traceback.print_exc()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.fresh_start = False

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                if COOP_SECTORS:  # second action: > 0 throttle, < 0 brake (upstream mapped [-1, 1] to throttle only)
                    a = max(min(float(action_idx[1]), 1.0), -1.0)
                    throttle, brake = max(a, 0.0), max(-a, 0.0)
                    self.d_steer, self.d_pedal, self.prev_pedal = steer - self.previous_steer, a - self.prev_pedal, a
                else:
                    throttle = float((action_idx[1] + 1.0)/2)
                    throttle = max(min(throttle, 1.0), 0.0)
                    brake = 0.0
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1, brake=brake))
                if COOP_SECTORS:
                    self.horn = len(action_idx) > 2 and float(action_idx[2]) > 0.5  # third action: the horn (> 0.5 = sounding)
                    self.horns += self.horn
                    self._update_blinker()
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                self.throttle = 1.0
            if COOP_SECTORS:
                self.world.tick()
                vs = self.world.get_actors().filter('vehicle.*')
                self.vehicle_locs = [(a.id, a.get_location()) for a in vs]  # for rider clearance
                self.vehicle_xy = np.array([(l.x, l.y) for _, l in self.vehicle_locs]).reshape(-1, 2)
                self.vehicle_ids = np.array([i for i, _ in self.vehicle_locs])
                self.vehicle_ext = np.array([(a.bounding_box.extent.x, a.bounding_box.extent.y) for a in vs]).reshape(-1, 2)
                self.riders = [r for r in self.riders if r.step()]  # scripted reckless motorbikes
                if self.infra:
                    self.infra.update(self)
                if self.traffic:
                    self.traffic.update(self)
            
            # Traffic Light state. VN mode: real lights (the swarm waits at red), but the ego waits at most VN_RED_WAIT s,
            # then its light turns green and the rest of its junction red: long reds only slowed training down
            red = self.vehicle.is_at_traffic_light() and self.vehicle.get_traffic_light_state() == carla.TrafficLightState.Red
            self.red_wait = self.red_wait + 0.05 if red else 0.0
            if red and (not self.traffic or self.red_wait > VN_RED_WAIT):
                traffic_light = self.vehicle.get_traffic_light()
                if self.traffic:
                    for other in traffic_light.get_group_traffic_lights():
                        if other.id != traffic_light.id:
                            other.set_state(carla.TrafficLightState.Red)
                traffic_light.set_state(carla.TrafficLightState.Green)
                self.red_wait = 0.0

            self.collision_history = self.collision_obj.collision_data            

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()


            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

             # Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            # Rewards are given below!
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                done = True
                reward = -10
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
            elif max(self.episode_start_time, self.last_blocked) + 10 < self.now() and self.velocity < 1.0:  # stopping for an obstacle (and 10 s after it clears) is fine
                reward = -10
                done = True
            elif self.velocity > self.max_speed:
                reward = -10
                done = True

            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.continous_action_space:
                    if self.velocity < self.min_speed:
                        reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                    elif self.velocity > self.target_speed:               
                        reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    else:                                         
                        reward = 1.0 * centering_factor * angle_factor 
                else:
                    reward = 1.0 * centering_factor * angle_factor
                if COOP_SECTORS and self.continous_action_space:
                    red = self.traffic is not None and self.vehicle.is_at_traffic_light() and \
                        self.vehicle.get_traffic_light_state() == carla.TrafficLightState.Red and self.velocity > 5
                    self.red_moving = red
                    reward -= 0.02 * self.horn  # honking costs a little: it must buy something (bikes moving aside)
                    reward -= safety_penalty(self.front_obstacle, self.last_sectors.min() * COOP_RANGE, self.velocity, red,
                                             self.d_steer, self.d_pedal)

            if self.timesteps >= 7500:
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            self.wait_for(lambda: len(self.camera_obj.front_camera))

            self.image_obs = self.camera_obj.front_camera.pop(-1)
            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.concatenate([np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle]), self.coop_obs()])
            
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                # batch destroy: destroying Traffic Manager cars one by one aborts the 0.9.10 client
                self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.sensor_list + self.actor_list], bool(COOP_SECTORS))
                self.sensor_list.clear()
                self.actor_list.clear()
                self.remove_sensors()
            
            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        except:
            traceback.print_exc()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()



# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            if COOP_SECTORS:
                self.world.tick()  # walkers only exist (and their controllers can start) after a tick
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True, TM_PORT)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # A random spawn point from which the road goes on for at least `metres`
    def long_route_spawn(self, metres, tries=30):
        points = self.map.get_spawn_points()
        for _ in range(tries):
            sp = random.choice(points)
            wp, n = self.map.get_waypoint(sp.location), 0
            while n < metres:
                nxt = wp.next(1.0)
                if not nxt:
                    break
                wp, n = random.choice(nxt), n + 1
            if n >= metres:
                return sp
        return random.choice(points)


    # Seconds for the episode rules: simulated time in sync mode (the sim may run slower than real time)
    def now(self):
        return self.timesteps * 0.05 if COOP_SECTORS else time.time()


    # Wait for sensor data; in sync mode (coop) the sim only moves when we tick
    def wait_for(self, ready):
        t = time.time()
        while not ready():
            time.sleep(0.0001)
            if COOP_SECTORS and time.time() - t > 0.05:
                self.world.tick()
                t = time.time()


    # Coop observation: (sectors, corridor) now and their change since the last step, so the agent sees motion
    def coop_obs(self):
        if not COOP_SECTORS:
            return np.zeros(0)
        return np.concatenate([self._coop_vec(self.share, primary=True), self.signal_obs(), self.infra_obs()])

    # The same features for the other sharing mode (no side effects on the env), recorded for the ablation
    def coop_obs_other(self):
        return np.concatenate([self._coop_vec(not self.share, primary=False), self.signal_obs(), self.infra_obs()])

    def _update_blinker(self):
        '''Turn signal from the route: the road 3 m ahead vs 25 m ahead turns by more than 30 deg -> blink that way.'''
        i, wps = self.current_waypoint_index, self.route_waypoints
        blink = 0
        if i + 25 < len(wps):
            turn = (wps[i + 25].transform.rotation.yaw - wps[i + 3].transform.rotation.yaw + 180) % 360 - 180
            blink = 1 if turn > 30 else -1 if turn < -30 else 0  # UE4 yaw grows clockwise: positive = right
        if blink != self.blinker:
            self.blinker = blink
            L = carla.VehicleLightState
            self.vehicle.set_light_state(L.RightBlinker if blink > 0 else L.LeftBlinker if blink < 0 else L.NONE)

    # Turn signals and horn as the ego perceives them (a camera reads blinkers of the vehicles ahead)
    def signal_obs(self):
        out = [float(self.blinker < 0), float(self.blinker > 0), float(self.horn)]
        t = self.vehicle.get_transform()
        f = t.get_forward_vector()
        near = []
        if len(self.vehicle_ids):
            d = self.vehicle_xy - (t.location.x, t.location.y)
            along, lat = d @ (f.x, f.y), d @ (-f.y, f.x)
            for k in np.argsort(np.hypot(along, lat)):
                if self.vehicle_ids[k] != self.vehicle.id and -5 < along[k] < 30 and abs(lat[k]) < 8:
                    near.append(k)
                    if len(near) == 4:
                        break
        lights = {}
        for a in (self.world.get_actors([int(self.vehicle_ids[k]) for k in near]) if near else []):
            try:
                lights[a.id] = int(a.get_light_state())
            except RuntimeError:  # bike recycled/destroyed since vehicle_ids was cached
                pass
        L = carla.VehicleLightState
        for k in near:
            ls = lights.get(int(self.vehicle_ids[k]), 0)
            out += [along[k] / 30, lat[k] / 30, float(bool(ls & int(L.LeftBlinker))), float(bool(ls & int(L.RightBlinker)))]
        out += [0.0] * 4 * (4 - len(near))
        return np.array(out, np.float32)

    # What the ego knows through V2X (objects from RSUs / coop cars, the light ahead); zeros without infrastructure
    def infra_obs(self):
        return self.infra.obs(self) if self.infra else np.zeros(INFRA_DIM, np.float32)

    def _coop_vec(self, share, primary):
        sectors, front = self.coop_features(share)
        if primary:
            self.front_obstacle, self.last_sectors = front, sectors
            if front < 10 or (self.infra and self.infra.warning_now() is not None):  # waiting out a DENM is not idling
                self.last_blocked = self.now()
        cur = np.append(sectors, min(front, COOP_CORRIDOR) / COOP_CORRIDOR)
        prev = self.prev_coop.get(share, cur)
        self.prev_coop[share] = cur
        return np.concatenate([cur, cur - prev])

    # Fuse the latest scans (all coop cars, or only the ego with share=False) into sector distances + lane obstacle
    def coop_features(self, share):
        ego = self.lidar_objs[0]
        scans = [l.latest for l in (self.lidar_objs if share else self.lidar_objs[:1]) if l.latest is not None]
        ext = self.vehicle.bounding_box.extent
        clouds, poses, z = [s[0] for s in scans], [s[1] for s in scans], dict(z_min=-ego.height + 0.3, z_max=-ego.height + 2.0)
        sectors = sector_distances(clouds, poses, ego.latest[1], (ext.x + 0.3, ext.y + 0.3), COOP_SECTORS, COOP_RANGE, **z)
        i = self.current_waypoint_index + 3  # ~front bumper
        path = [(w.transform.location.x, w.transform.location.y, w.transform.location.z)
                for w in self.route_waypoints[i:i + int(COOP_CORRIDOR) + 1]]
        front = corridor_distance(clouds, poses, path, ext.y + 0.3, z_min=0.3, z_max=1.5,
                                  ego_pose=ego.latest[1], ego_half_extent=(ext.x + 0.3, ext.y + 0.3)) if len(path) > 1 else np.inf
        return sectors, front


    # Clean up method
    def remove_sensors(self):
        self.lidar_objs = list()
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

