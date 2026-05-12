# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc import train
from irlc.gridworld.gridworld_environments import GridworldEnvironment
from irlc.ex11.feature_encoder import FeatureEncoder
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa
from irlc.ex10.sarsa_agent import SarsaAgent
from irlc import interactive, savepdf

np.seterr(all='raise')

small_circle_grid = [[' ', +1,'#'],
                     [' ','#', -1],
                     ['S',' ',' ']]

class TinyGridworld(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(small_circle_grid, *args, **kwargs)


class LinearFeatureEncoder(FeatureEncoder):
    """Feature encoder using linear features: z_lin(s) = [1, u, v]"""
    def __init__(self, env):
        self.env = env
        self.H = env.mdp.height
        self.W = env.mdp.width
        self.l = 3  # dimension of z_lin(s) = [1, u, v]
        self.na = env.action_space.n  # number of actions
        super().__init__(env)

    @property
    def d(self):
        """Total feature dimension: 4*l (action-dependent construction)"""
        return 4 * self.l

    def x(self, s, a):
        """
        Compute action-dependent feature vector x(s, a).
        For action a, place z_lin(s) in the a-th block of 4 blocks.
        """
        i, j = s
        u = i / (self.H - 1) if self.H > 1 else 0
        v = j / (self.W - 1) if self.W > 1 else 0
        z_lin = np.array([1.0, u, v])
        
        # Create action-dependent feature vector
        x = np.zeros(self.d)
        start_idx = a * self.l
        x[start_idx:start_idx + self.l] = z_lin
        return x

class QuadraticFeatureEncoder(FeatureEncoder):
    def __init__(self, env):
        self.env = env
        self.H = env.mdp.height
        self.W = env.mdp.width
        self.l = 6
        self.na = env.action_space.n
        super().__init__(env)

    @property
    def d(self):
        return 4 * self.l

    def x(self, s, a):
        i, j = s
        u = i / (self.H - 1) if self.H > 1 else 0
        v = j / (self.W - 1) if self.W > 1 else 0
        z_quad = np.array([1.0, u, v, u**2, v**2, u*v])

        x = np.zeros(self.d)
        start_idx = a * self.l
        x[start_idx:start_idx + self.l] = z_quad
        return x

def linear_experiment(episodes=10000, alpha=0.02, states_and_actions=( ((0,0),0) )):
    env = TinyGridworld()
    
    # Create the linear feature encoder and agent
    q_encoder = LinearFeatureEncoder(env)
    agent = LinearSemiGradSarsa(env, gamma=1.0, epsilon=1.0, alpha=alpha, q_encoder=q_encoder)
    
    # Train the agent
    train(env, agent, num_episodes=episodes, verbose=False)
    
    # Extract Q-values for the specified states and actions
    q_values = {(s, a): agent.Q(s, a) for s, a in states_and_actions}
    return q_values

def quadratic_experiment(episodes=10000, alpha=0.02, states_and_actions=( ((0,0),0) )):
    env = TinyGridworld()
    q_encoder = QuadraticFeatureEncoder(env)
    agent = LinearSemiGradSarsa(env, gamma=1.0, epsilon=1.0, alpha=alpha, q_encoder=q_encoder)
    train(env, agent, num_episodes=episodes, verbose=False)
    q_values = {(s, a): agent.Q(s, a) for s, a in states_and_actions}
    return q_values


if __name__ == "__main__":
    env = TinyGridworld()
    agent = SarsaAgent(env, alpha=0.02, gamma=1., epsilon=1) 
    train(env, agent, num_episodes=10000, verbose=False) 
    # Example: This will extract the states/actions.
    agent.label = "Sarsa after 10000 episodes, alpha=0.02"
    env, agent = interactive(env, agent)
    savepdf('sarsa_eval10k', env=env)
    env.close()
    # Evaluate the Sarsa agent:
    states_and_actions = [((0, 0), 0),
                          ((0, 0), 1),
                          ((2, 0), 3),
                          ((0, 2), 1),
                          ]
    q_values = {(s, a): agent.Q[s, a] for s, a in states_and_actions}
    for (s, a), q in q_values.items(): 
        print(f"In state {s=} and action {a=} we have Q(s,a) = {q}.")
    # Question 1: Estimate using the linear function approximator
    qs = linear_experiment(episodes=10000, alpha=0.01, states_and_actions=states_and_actions)
    for (s,a), q in qs.items(): 
        print(f"In state {s=} and action {a=} we have Q(s,a)={q} (Linear function approximator)")
    # Question 2: Estimate using the quadratic function approximator
    qs = quadratic_experiment(episodes=10000, alpha=0.01, states_and_actions=states_and_actions)
    for (s, a), q in qs.items(): 
        print(f"In state {s=} and action {a=} we have Q(s,a)={q} (Quadratic function approximator)")
