import torch
from controller import Controller
from render.render import Renderer

def main():
    #####################################################################################
    # Instantiate your controller and load the trained SAC model.
    SEC = 10  # Rendering period in seconds.

    # Initialize the controller with training=False for evaluation
    your_controller = Controller(
        state_size=10,
        action_size=1,
        a_max=60,
        training=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load the trained SAC model parameters.
    SAC_MODEL_PATH = 'sac_model.pth'  # Path to the saved SAC model
    if torch.cuda.is_available():
        sac_checkpoint = torch.load(SAC_MODEL_PATH)
    else:
        sac_checkpoint = torch.load(SAC_MODEL_PATH, map_location=your_controller.device)

    # Load the SAC policy network
    your_controller.sac_controller.policy_net.load_state_dict(sac_checkpoint['sac_policy_net'])
    print("SAC model loaded successfully.")

    # Since we're evaluating, no need to load optimizers or target networks.
    print("Model successfully loaded for rendering.")
    #####################################################################################

    # Instantiate the Renderer and begin rendering.
    renderer = Renderer(controller=your_controller, sec=SEC)
    print("Starting the rendering process...")
    transitions = renderer.render()

    # Save the transitions to a txt file.
    with open('transitions.txt', 'w') as f:
        for transition in transitions:
            state, action, next_state = transition
            f.write(f"State: {state}\n")
            f.write(f"Action: {action}\n")
            f.write(f"Next State: {next_state}\n")
            f.write("\n")  # Add a blank line for separation between transitions

    print("Transitions saved to 'transitions.txt'.")

if __name__ == '__main__':
    main()
