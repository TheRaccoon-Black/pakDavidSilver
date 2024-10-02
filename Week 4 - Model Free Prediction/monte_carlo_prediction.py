import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class BlackjackEnv:
    def __init__(self):
        self.action_space = [0, 1]  # 0: Stick, 1: Hit
        self.state = None
        self.reset()

    def reset(self):
        """Reset the environment to an initial state."""
        player_card = random.randint(1, 10)
        dealer_card = random.randint(1, 10)
        usable_ace = random.choice([True, False])  # Randomly assign usable ace
        self.state = (player_card, dealer_card, usable_ace)
        return self.state

    def step(self, action):
        """Perform an action in the environment."""
        player_card, dealer_card, usable_ace = self.state
        
        if action == 1:  # Hit
            player_card += random.randint(1, 10)
            if player_card > 21:  # Player bust
                return None, -1, True, {}
        else:  # Stick
            while dealer_card < 17:  # Dealer hits until reaching 17 or higher
                dealer_card += random.randint(1, 10)
            if dealer_card > 21 or player_card > dealer_card:
                return self.state, 1, True, {}
            elif player_card < dealer_card:
                return self.state, -1, True, {}
            else:
                return self.state, 0, True, {}
        
        self.state = (player_card, dealer_card, usable_ace)
        return self.state, 0, False, {}

    def render(self):
        """Optional method to render the environment."""
        print(f"Player: {self.state[0]}, Dealer: {self.state[1]}, Usable Ace: {self.state[2]}")

def sample_policy(state):
    """Sample policy for blackjack (randomly choose an action)."""
    return np.random.choice([0, 1])  # Randomly choose to stick or hit

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    returns_num = defaultdict(float)
    V = defaultdict(float)

    for episode in range(num_episodes):
        episode_rewards = defaultdict(float)
        state = env.reset()

        if state is None:
            print(f"Episode {episode} reset returned None, skipping...")
            continue

        terminated = False
        while not terminated:
            action = policy(state)
            next_state, reward, terminated, _ = env.step(action)

            if next_state is None:
                print(f"Episode {episode} step returned None, terminating...")
                break

            episode_rewards[state] += reward
            state = next_state

        for state in episode_rewards:
            returns_num[state] += 1
            V[state] += (episode_rewards[state] - V[state]) / returns_num[state]

    return V

def plot_value_function(V, title="Value Function"):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.bar(range(len(V)), list(V.values()), align='center')
    plt.xticks(range(len(V)), [str(k) for k in V.keys()], rotation=45)
    plt.show()

if __name__ == "__main__":
    env = BlackjackEnv()
    num_episodes = 10000
    V_10k = mc_prediction(sample_policy, env, num_episodes=num_episodes)
    plot_value_function(V_10k, title="Value Function after 10,000 Episodes")
