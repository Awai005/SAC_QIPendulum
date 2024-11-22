import os
import numpy as np
import math
import socket
import time
import torch
from controller import Controller  # Ensure 'controller.py' is in the same directory

def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 65432))
            s.sendall(command.encode())
            data = s.recv(1024).decode()
        return data.strip()
    except ConnectionRefusedError:
        print("Error: Unable to connect to the QIP system. Ensure it is running and listening on port 65432.")
        exit(1)
    except Exception as e:
        print(f"Socket error: {e}")
        exit(1)

def wrap_angle(angle):
    return math.remainder(angle, 2 * np.pi)

def check_done_condition(state):
    cart_position = state[0]
    if cart_position < -6 or cart_position > 6:
        return True
    angles = [abs(wrap_angle(state[i])) for i in [2, 4, 6, 8]]
    threshold = np.pi 
    if any(angle >= threshold for angle in angles):
        return True
    return False

def main():
    NUM_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 10000
    ENV_RESET_COMMAND = 'reset'
    ENV_STEP_COMMAND = 'step'
    BEST_MODEL_PATH = 'sac_model.pth'  # Path to save the SAC model
    REWARD_HISTORY_PATH = 'sac_reward_history.npy'
    BATCH_SIZE = 256  # Batch size for SAC updates

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize your controller with training=True for SAC
    your_controller = Controller(
        state_size=10,
        action_size=1,
        a_max=60,
        training=True,    # Training SAC controller
        device=device
    )

    # Load the SAC model if previously saved
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading SAC model from '{BEST_MODEL_PATH}'...")
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        your_controller.sac_controller.policy_net.load_state_dict(checkpoint['sac_policy_net'])
        your_controller.sac_controller.critic_net.load_state_dict(checkpoint['sac_critic_net'])
        your_controller.sac_controller.critic_net_target.load_state_dict(checkpoint['sac_critic_net_target'])
        your_controller.sac_controller.policy_optimizer.load_state_dict(checkpoint['sac_policy_optimizer'])
        your_controller.sac_controller.critic_optimizer.load_state_dict(checkpoint['sac_critic_optimizer'])
        your_controller.sac_controller.log_alpha = checkpoint['sac_log_alpha']
        your_controller.sac_controller.alpha_optimizer.load_state_dict(checkpoint['sac_alpha_optimizer'])
        print("SAC model loaded successfully.")
    else:
        print("Starting SAC training from scratch.")

    best_cumulative_reward = -np.inf
    reward_history = []

    start_time = time.time()
    for episode in range(1, NUM_EPISODES + 1):
        state_str = send_command(ENV_RESET_COMMAND)
        state = list(map(float, state_str.split()))
        cumulative_reward = 0
        done = False

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # Select action using SAC controller
            action = your_controller.compute_force(state)
            action_value = float(action[0])  # Extract the scalar value for the environment

            # Step in environment
            next_state_str = send_command(f"{ENV_STEP_COMMAND} {action_value}")
            next_state_values = next_state_str.strip().split()
            if len(next_state_values) != len(state):
                print(f"Unexpected number of next_state values received: {len(next_state_values)}")
                done = True
                break
            next_state = list(map(float, next_state_values))

            # Check for termination condition
            done = check_done_condition(next_state)

            # Compute reward
            reward = your_controller.compute_reward(state, action_value)  # Use action_value for reward computation
            cumulative_reward += reward

            # Store transition
            your_controller.sac_controller.store_transition(state, action, reward, next_state, done)

            # Update SAC controller
            if len(your_controller.sac_controller.replay_buffer) > BATCH_SIZE:
                your_controller.sac_controller.update_parameters(BATCH_SIZE)

            # Update state
            state = next_state
            if done:
                break

        reward_history.append(cumulative_reward)

        # Save the best SAC model
        if cumulative_reward > best_cumulative_reward:
            best_cumulative_reward = cumulative_reward
            torch.save({
                'sac_policy_net': your_controller.sac_controller.policy_net.state_dict(),
                'sac_critic_net': your_controller.sac_controller.critic_net.state_dict(),
                'sac_critic_net_target': your_controller.sac_controller.critic_net_target.state_dict(),
                'sac_policy_optimizer': your_controller.sac_controller.policy_optimizer.state_dict(),
                'sac_critic_optimizer': your_controller.sac_controller.critic_optimizer.state_dict(),
                'sac_log_alpha': your_controller.sac_controller.log_alpha,
                'sac_alpha_optimizer': your_controller.sac_controller.alpha_optimizer.state_dict()
            }, BEST_MODEL_PATH)
            print(f"Episode {episode}: New best cumulative reward {best_cumulative_reward:.2f}. SAC model saved.")

        # Print progress
        if episode % 50 == 0 or episode == 1:
            avg_reward = np.mean(reward_history[-50:]) if episode >= 50 else np.mean(reward_history)
            print(f"Episode {episode}/{NUM_EPISODES} - "
                  f"Average Reward (last 50): {avg_reward:.2f} - "
                  f"Best Reward: {best_cumulative_reward:.2f}")

    # Training completed
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes.")
    print(f"Best cumulative reward achieved: {best_cumulative_reward:.2f}")
    print(f"SAC model saved to '{BEST_MODEL_PATH}'.")

    # Save reward history
    np.save(REWARD_HISTORY_PATH, np.array(reward_history))
    print(f"Reward history saved to '{REWARD_HISTORY_PATH}'.")

if __name__ == "__main__":
    main()
