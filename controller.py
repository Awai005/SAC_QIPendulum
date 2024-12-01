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
        Compute the immediate reward based on the current state, action, and physical parameters.

        Rewards:
        - Incorporates physical parameters of links into penalties and bonuses.
        - Encourages link coordination and efficient movement.
        """
        # Extract state components
        cart_pos = state[0]  # Cart position
        cart_vel = state[1]  # Cart velocity
        angles = [self.wrap_angle(state[i]) for i in [2, 4, 6, 8]]  # Pendulum angles, wrapped to [-π, π]
        angular_vels = [state[i] for i in [3, 5, 7, 9]]  # Angular velocities for pendulum links

        # Physical parameters of the system
        masses = [0.2, 0.25, 0.43, 0.81]  # Mass of links
        lengths = [0.15, 0.20, 0.40, 0.80]  # Lengths of links
        cog_distances = [0.075, 0.1, 0.2, 0.4]  # Distance from hinge to CoG
        moments_of_inertia = [0.0015, 0.0033, 0.0229, 0.1728]  # MoI of links about their CoG

        # Reward parameters
        cart_pos_penalty_weight = 0.5
        angular_vel_penalty_weight = 0.1
        swing_up_reward_weight = 250 # 250
        link_interaction_penalty_weight = 1.5

        # Dynamically scale weights based on physical parameters
        scaled_weights = [
        m * l * cog * moi for m, l, cog, moi in zip(masses, lengths, cog_distances, moments_of_inertia)]

        # 1. Penalize cart position deviation from the center
        cart_position_penalty = -cart_pos_penalty_weight * (cart_pos ** 2)

        # 2. Penalize high angular velocities
        angular_velocity_penalty = -angular_vel_penalty_weight * sum(
            scaled_weights[i] * angular_vels[i] ** 2 for i in range(len(angular_vels))
        )


        # 5. Reward progress toward the upright position (swing-up incentive)
        swing_up_reward = swing_up_reward_weight * sum(
            scaled_weights[i] * (1 - abs(angles[i] / math.pi)) for i in range(len(angles))
        )


        # 7. Penalize large relative angle differences between consecutive links
        link_interaction_penalty = -link_interaction_penalty_weight * sum(
            (angles[i] - angles[i + 1]) ** 2 / (masses[i] * lengths[i]) for i in range(len(angles) - 1)
        )

        # Combine all components into the final reward
        reward = (
              swing_up_reward
            + cart_position_penalty
            + angular_velocity_penalty
        )

        return reward


    def compute_force(self, state):
        """
        Compute force using SAC controller.
        """
        action = self.sac_controller.compute_force(state)
        return action
