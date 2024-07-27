        lookahead_distance = int(np.clip(vehicle_velocity_norm / 4.0, 8, 20))  # Adjusted lookahead for faster response

        waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + lookahead_distance) % len(self.maneuverable_waypoints)]
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        turn_sharpness = abs(delta_heading)
        if turn_sharpness > np.pi / 45:  # More aggressive threshold for sharp turns
            steering_smooth_factor = 40.0
            target_speed = 85  # Lower speed for sharp turns
        else:
            steering_smooth_factor = 3.0
            target_speed = 120  # Higher speed for straights

        steer_control = (-steering_smooth_factor * delta_heading / np.pi) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        throttle_control = 0.0095 * (target_speed - vehicle_velocity_norm)  # Slightly increased throttle control

        brake_control = 0.0
        if turn_sharpness > np.pi / 20:
            if vehicle_velocity_norm > target_speed:
                brake_control = np.clip(0.9 * (vehicle_velocity_norm - target_speed), 0.0, 1.0)
        else:
            if vehicle_velocity_norm > target_speed + 15:
                brake_control = np.clip(0.6 * (vehicle_velocity_norm - target_speed), 0.0, 1.0)

        max_safe_speed = 100  # Adjusted safety speed limit for Monza
        if vehicle_velocity_norm > max_safe_speed:
            throttle_control = min(throttle_control - 0.4, 0.0)
            brake_control = min(brake_control + 0.4, 1.0)

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": brake_control,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }