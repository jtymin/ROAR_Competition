"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )


    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )



        lookahead_distance = int(np.clip(vehicle_velocity_norm / 3.0, 8, 20))  # Conservative lookahead

        waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + lookahead_distance) % len(self.maneuverable_waypoints)]

        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Determine turn sharpness and adjust speed accordingly
        turn_sharpness = abs(delta_heading)
        if turn_sharpness > np.pi / 18:  # Conservative threshold for sharp turns
            steering_smooth_factor = 6.0
            target_speed = 25  # More conservative speed for sharp turns
        else:
            steering_smooth_factor = 4.0
            target_speed = 75  # Increased speed for straighter sections

        steer_control = (
            -steering_smooth_factor * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Smooth throttle control for better acceleration management
        throttle_control = 0.02 * (target_speed - vehicle_velocity_norm)

        # Implement predictive braking with earlier and more aggressive deceleration
        brake_control = 0.0
        if turn_sharpness > np.pi / 18:
            if vehicle_velocity_norm > target_speed:
                brake_control = np.clip(0.8 * (vehicle_velocity_norm - target_speed), 0.0, 1.0)
        else:
            if vehicle_velocity_norm > target_speed + 10:
                brake_control = np.clip(0.5 * (vehicle_velocity_norm - target_speed), 0.0, 1.0)

        # Additional safety measures
        max_safe_speed = 90  # Safety speed limit
        if vehicle_velocity_norm > max_safe_speed:
            throttle_control = min(throttle_control - 0.3, 0.0)  # Reduce throttle if over speed limit
            brake_control = min(brake_control + 0.3, 1.0)  # Increase braking if needed

        # Ensure controls are within safe limits
        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": brake_control,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control