# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
import matplotlib.pyplot as plt
from irlc import main_plot, savepdf
from irlc.ex01.agent import train
import numpy as np
import gymnasium as gym
from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
np.seterr(all='raise')

class LinearSemiGradSarsa(LinearSemiGradQAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, q_encoder=None):
        r"""Implement the Linear semi-gradient Sarsa method from (SB18, Section 10.1)"""
        super().__init__(env, gamma, epsilon=epsilon, alpha=alpha, q_encoder=q_encoder)

    def pi(self, s, k, info=None): 
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Random action
            if info is not None and 'mask' in info:
                from irlc.ex08.rl_agent import _masked_actions
                actions = _masked_actions(self.env.action_space, info['mask'])
                return np.random.choice(actions)
            else:
                return self.env.action_space.sample()
        else:
            # Greedy action
            return self.Q.get_optimal_action(s, info)
        return action

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        # Semi-gradient Sarsa update
        # Get the next action (for Sarsa)
        ap = self.pi(sp, 0, info_sp)
        
        # Compute the TD target
        q_target = r + (0 if done else self.gamma * self.Q(sp, ap))
        
        # Compute TD error and update weights
        delta = q_target - self.Q(s, a)
        self.Q.w += self.alpha * delta * self.Q.x(s, a)

        if sum(np.abs(self.Q.w)) > 1e5: raise Exception("Weights diverged. Decrease alpha")

    def __str__(self):
        return f"LinSemiGradSarsa{self.gamma}_{self.epsilon}_{self.alpha}"

experiment_sarsa = "experiments/mountaincar_Sarsa"

if __name__ == "__main__":
    from irlc.ex11.semi_grad_q import experiment_q, alpha, x
    from irlc.ex09 import envs

    env = gym.make("MountainCar500-v0")
    for _ in range(10):
        agent = LinearSemiGradSarsa(env, gamma=1, alpha=alpha, epsilon=0)
        train(env, agent, experiment_sarsa, num_episodes=300, max_runs=10)

    main_plot(experiments=[experiment_q, experiment_sarsa], x_key=x, y_key='Length', smoothing_window=30)
    savepdf("semigrad_q_sarsa")
    plt.show()

    # Turn off averaging
    main_plot(experiments=[experiment_q, experiment_sarsa], x_key=x, y_key='Length', smoothing_window=30, units="Unit", estimator=None)
    savepdf("semigrad_q_sarsa_individual")
    plt.show()
