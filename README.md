# Guided Cost Learning (GCL) Implementation

## Project Description
This project provides an implementation of Guided Cost Learning (GCL), an algorithm focused on inverse reinforcement learning (IRL). The primary objective is to learn a cost function from expert demonstrations, which can then be used to train a policy that mimics the expert's behavior. The implementation aims to replicate the key findings of the original GCL paper.

## Papers Replicated
- Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization (Finn et al., 2016)

## Algorithms Overview

### GCL (Guided Cost Learning)
- **Type**: Inverse Reinforcement Learning, Adversarial Imitation Learning
- **Key Features**:
  - Learns a cost function from expert trajectories.
  - Utilizes an adversarial framework where a policy is optimized against the learned cost function.
  - Leverages importance sampling to handle high-dimensional state-action spaces.
  - Can be combined with standard RL algorithms (like TRPO or PPO) for policy optimization based on the learned cost.

## Contributions
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License
MIT License

## References
1. Finn, C., Levine, S., & Abbeel, P. (2016). Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization.
