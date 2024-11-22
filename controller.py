import numpy as np
import math
from sac_controller import SACController

class Controller:
    def __init__(self, state_size=10, action_size=1, a_max=60, training=True, device='cpu'):
        """
        Controller using SAC for swing-up task.
        """
        self.device = device
        self.training = training
        self.a_max = a_max

        # SAC controller
        self.sac_controller = SACController(state_size, action_size, a_max, training, device)
        
    def wrap_angle(self, angle):
        """
        Wrap angle to the range [-π, π].
        """
        return math.atan2(math.sin(angle), math.cos(angle))



    def compute_reward(self, state, action):
        """
        Compute the immediate reward based on the current state and action.

        Rewards:
        - Penalize large cart position to encourage staying near the center.
        - Penalize high angular velocities to reduce oscillations.
        - Encourage pendulum angles to be closer to the upright position (angle = 0 radians).
        - Penalize extreme actions to avoid large forces.
        """
        # Extract state components
        cart_pos = state[0]  # Cart position
        cart_vel = state[1]  # Cart velocity
        angles = [self.wrap_angle(state[i]) for i in [2, 4, 6, 8]]  # Pendulum angles, wrapped to [-π, π]
        angular_vels = [state[i] for i in [3, 5, 7, 9]]  # Angular velocities for pendulum links

        # Reward parameters
        cart_pos_penalty_weight = 1.0
        angular_vel_penalty_weight = 0.5
        upright_bonus_weight = 1.0
        action_penalty_weight = 0.1

        # Penalize cart position deviation from the center
        cart_position_penalty = -cart_pos_penalty_weight * (cart_pos ** 2)

        # Penalize high angular velocities
        angular_velocity_penalty = -angular_vel_penalty_weight * sum(vel ** 2 for vel in angular_vels)

        # Encourage pendulum angles closer to the upright position (angle = 0 radians)
        # Since cos(0) = 1 and cos(±π) = -1, (1 + cos(angle)) / 2 ranges from 0 to 1
        upright_bonus = upright_bonus_weight * sum((1 + math.cos(angle)) / 2 for angle in angles)

        # Penalize large actions to avoid extreme forces
        #action_penalty = -action_penalty_weight * (action / self.a_max) ** 2

        # Combine all components
        reward = cart_position_penalty + angular_velocity_penalty + upright_bonus 



        return reward


    def compute_force(self, state):
        """
        Compute force using SAC controller.
        """
        action = self.sac_controller.compute_force(state)
        return action
