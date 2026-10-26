# 🚀 Reinforcement Learning for 7-DoF Robotic Manipulation: TD3 Algorithm in MuJoCo (Pusher-v5)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-112C3E)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics_Engine-1A1A1A?logo=google&logoColor=white)](https://mujoco.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-Production_RL-4B8BBE)](https://stable-baselines3.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Mathematics-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model_Training-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

This repository provides a complete **Reinforcement Learning (RL)** pipeline for training a highly flexible **7-Degree-of-Freedom (7-DoF) robotic arm** to manipulate objects within the `Pusher-v5` environment. By leveraging the **Twin Delayed DDPG (TD3)** algorithm and the **MuJoCo physics engine**, this project tackles high-dimensional continuous control, dense reward engineering, and action noise injection for advanced robotic manipulation.

---
## 🛤️ Deep Reinforcement Learning Journey

This project elevates RL complexity from simple navigation tasks to advanced physical object manipulation using Stable Baselines3:

### Phase 1: Environment Anatomy & Version Control
- **Continuous Control Action Space:** 7 continuous dimensions representing the torque applied to 7 distinct joints (mimicking human arm kinematics), bounded between .
- **Observation Space:** 23-dimensional feature vector tracking joint angles, joint velocities, cylinder (X, Y) coordinates, and red target (X, Y) coordinates.
- **Gymnasium Version Conflict Resolution:** Addressed the classic `mujoco<3` version dependency conflict with `Pusher-v4`. Engineered a clean migration to `Pusher-v5` to maintain compatibility with modern physics simulators without degrading server-side environments.

### Phase 2: Reward Engineering & Cost Minimization
- **Dense Reward Shaping:** The environment relies on continuous negative rewards (cost minimization) to force the RL agent to solve the task efficiently.
- **The Mathematical Objective:** `Reward = - (Arm to Cylinder Distance) - (Cylinder to Target Distance) - (Control Energy Penalty)`.
- **Behavioral Result:** The negative distance penalty persists until the cylinder reaches the target, mathematically forcing the neural network to learn physical manipulation rather than settling for local optima.

### Phase 3: TD3 Production Training & Noise Injection
- **Action Noise for Exploration:** Because the TD3 algorithm employs a deterministic policy, we injected `NormalActionNoise` (`sigma=0.1`) across all 7 joints. This mathematical "jitter" forces the robotic arm to explore continuous spaces effectively.
- **Dense vs. Sparse Rewards:** Since the reward signal is dense (continuous distance feedback), standard TD3 proved highly capable without the computational overhead of Hindsight Experience Replay (HER).
- **Model Checkpointing:** Implemented `CheckpointCallback` in Stable Baselines3 to safely export model weights every 50,000 timesteps.

### Phase 4: Learning Curve & Cognitive Analysis
- **Model Evaluation:** Analyzed the agent's cognitive evolution over 300,000 steps. The `ep_len_mean` remained strictly at 100 steps (no termination state), while the `ep_rew_mean` successfully converged to `~-31.2`.
- **Training Graph Breakdown:**
  1. **Rapid Cognitive Jump (Episodes 0 - 250):** The reward spiked vertically from -150 to ~-40. The neural network rapidly grasped the core physics objective.
  2. **Micro-adjustments (Episodes 250 - 3000):** A steady, oscillating upward trend. The robotic arm spent thousands of episodes fine-tuning the 7-DoF joints down to the millimeter to optimize object manipulation and minimize energy penalties.

---

## 🧠 Core AI & Machine Learning Concepts Explored

- **7-DoF Robotic Manipulation:** Managing complex kinematics in high-dimensional continuous control spaces.
- **Deterministic Policy Optimization:** Engineering random exploration (`NormalActionNoise`) in TD3 to escape local minima in Reinforcement Learning.
- **Reward Function Design:** Implementing dense cost minimization techniques in MuJoCo environments.
- **Learning Curve Diagnostics:** Interpreting early cognitive breakthroughs versus late-stage micro-adjustments in Deep Learning models.

---

## 🛠️ Installation & Repository Structure

### Prerequisites for Machine Learning Environment

```bash
pip install torch numpy pandas matplotlib seaborn gymnasium[mujoco] stable-baselines3[extra] imageio
```

### Repository Structure

```text
📦 TD3-MuJoCo-Pusher-7DoF
 ┣ 📜 "Pusher with TD3.py"            # Main Python script: TD3 Training, Noise Injection, and Evaluation
 ┣ 📂 models                          
 ┃ ┗ 📜 td3_pusher_part1.zip          # Trained Neural Network Weights (~300k steps)            
 ┣ 📂 videos                          
 ┃ ┗ 🎬 mujoco_pusher_td3.mp4         # Rendered simulation of the robotic arm manipulating the object
 ┗ 📜 README.md
```

---

## 🏋️ Training and Evaluation

To execute the Reinforcement Learning training pipeline, inject action noise, and render the final MuJoCo evaluation:
```bash
python "Pusher with TD3.py"
```

## 👨‍💻 Author
**Mohammed Arif Mahyoub Haider**

*Electrical Engineer - Computer and Industrial Control*
