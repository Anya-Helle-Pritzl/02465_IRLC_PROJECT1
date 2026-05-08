# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.ex10.q_agent import QAgent
from irlc.gridworld.gridworld_environments import GridworldEnvironment, grid_bridge_grid
from irlc import train
from irlc.ex08.rl_agent import TabularQ

# A simple UCB action-selection problem (basic problem)
very_basic_grid = [['#',1, '#'],
                    [1, 'S', 2],
                    ['#',1, '#']]


class UCBAgent(QAgent):
    """UCB agent that uses Upper Confidence Bound exploration instead of epsilon-greedy."""
    def __init__(self, env, gamma=1.0, alpha=0.5, c=1.0):
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=0)
        self.c = c
        # Track visit counts: N[s] = total visits to state s, N_sa[s,a] = total visits to action a in state s
        self.N = TabularQ(env)  # Total state visits
        self.N_sa = TabularQ(env)  # State-action visits

    def pi(self, s, k, info=None):
        """Return action using UCB exploration."""
        import numpy as np
        
        if info is not None and 'seed' in info:
            np.random.seed(info['seed'])
        
        # Get available actions
        if info is not None and 'mask' in info:
            available_actions = [a for a in range(len(info['mask'])) if info['mask'][a] == 1]
        else:
            available_actions = list(range(self.env.action_space.n))
        
        N_s = self.N[s, 0]  # Total visits to this state (stored in first action slot)
        if N_s == 0:
            N_s = 1  # Avoid division by zero on first visit
        
        # Compute UCB value for each action
        max_ucb = -float('inf')
        best_action = available_actions[0]
        
        for a in available_actions:
            q_value = self.Q[s, a]
            n_sa = self.N_sa[s, a]
            if n_sa == 0:
                # Unvisited actions have infinite UCB (high priority)
                ucb = float('inf')
            else:
                exploration_bonus = self.c * np.sqrt(np.log(N_s) / n_sa)
                ucb = q_value + exploration_bonus
            
            # Pick the action with highest UCB, breaking ties by lowest action index
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = a
        
        return best_action

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        """Update Q-values and visit counts."""
        # Update visit counts
        self.N[s, 0] += 1  # Increment total state visits
        self.N_sa[s, a] += 1  # Increment state-action visits
        
        # Standard Q-learning update
        a_star = self.Q.get_optimal_action(sp, info_sp)
        self.Q[s, a] = self.Q[s, a] + self.alpha * (r + self.gamma * self.Q[sp, a_star] - self.Q[s, a])

def get_ucb_actions(layout : list, alpha : float, c : float, episodes : int, plot=False) -> list: 
    """ Return the sequence of actions the agent tries in the environment with the given layout-string when trained over 'episodes' episodes.
    To create an environment, you can use the line:

    > env = GridworldEnvironment(layout)

    See also the demo-file.

    The 'plot'-parameter is optional; you can use it to add visualization using a line such as:

    if plot:
        env = GridworldEnvironment(layout, render_mode='human')

    Or you can just ignore it. Make sure to return the truncated action list (see the rebels_demo.py-file or project description).
    In other words, the return value should be a long list of integers corresponding to actions:
    actions = [0, 1, 2, ..., 1, 3, 2, 1, 0, ...]
    """
    render_mode = 'human' if plot else None
    env = GridworldEnvironment(layout, render_mode=render_mode)
    agent = UCBAgent(env, gamma=1.0, alpha=alpha, c=c)
    stats, trajectories = train(env, agent, num_episodes=episodes, return_trajectory=True)
    env.close()
    
    # Concatenate all action sequences, excluding the last dummy action from each trajectory
    actions = []
    for trajectory in trajectories:
        actions.extend(trajectory.action[:-1])
    
    return actions

if __name__ == "__main__":
    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=4, plot=False)
    print("Number of actions taken", len(actions))
    print("List of actions taken over 4 episodes", actions)

    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=8, plot=False)
    print("Number of actions taken", len(actions))
    print("Actions taken over 8 episodes", actions)

    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=9, plot=False)
    print("Number of actions taken", len(actions))
    print("Actions taken over 9 episodes", actions) # In this particular case, you can also predict the 9th action. Why?

    # Simulate 100 episodes. This should solve the problem.
    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=100, plot=False)
    print("Basic: Actions taken over 100 episodes", actions)

    # Simulate 100 episodes for the bridge-environment. The UCB-based method should solve the environment without being overly sensitive to c.
    # You can compare your result with the Q-learning agent in the demo, which performs horribly.
    actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, c=5, episodes=300, plot=False)
    print("Bridge: Actions taken over 300 episodes. The agent should solve the environment:", actions)
