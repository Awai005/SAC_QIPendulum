# Control of a Quadruple Inverted Pendulum using Soft Actor Critic Agent

This project implements a control system for a quadruple inverted pendulum on a cart. The system uses a Soft Actor-Critic (SAC) controller for swing up and stabilization of a Pendulum.

---

## Background

This project builds upon experience gained during the 2nd Young AI Competition, where I won a bronze medal with a Q-table and LQR implementation for controlling an inverted pendulum. The competition provided valuable insights into control systems and reinforcement learning, which have been instrumental in developing this advanced hierarchical control system.

[Link to the competition or project](https://github.com/Awai005/qip_system)

---

## Features

- **SAC Controller:** Training code available for SAC model for efficient control.
- **Modular Design:** Clear separation between different controllers and training scripts.
- **Visualization Tools:** Includes scripts for rendering and visualizing the pendulum's behavior.

---

## Installation

### Prerequisites
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib


### Training the SAC Controller
```bash
python trainer.py
```


### Visualizing the Trained Model
```bash
python main.py
```

## Experience from the 2nd Young AI Competition
Participating in the 2nd Young AI Competition was a pivotal experience that deepened my understanding of control systems and reinforcement learning. By developing a Q-table and LQR implementation for an inverted pendulum, I learned the importance of:

Modeling complex dynamics accurately.
Designing efficient control strategies.
Balancing exploration and exploitation in learning algorithms.
Winning the bronze medal motivated me to tackle more complex systems, leading to this project on controlling a quadruple inverted pendulum using advanced reinforcement learning techniques.
[Q_Learning Implementation](https://github.com/Awai005/qip_system)

Acknowledgments
2nd Young AI Competition: For providing the platform to explore and develop foundational skills in AI and control systems.
