import numpy as np

def one_step_lookahead(environment, state, V, discount_factor):
    """
    Helper function to calculate the value function.
    """
    # Number of actions in the environment
    nA = environment.action_space.n

    # Create a vector of dimensionality same as the number of actions
    action_values = np.zeros(nA)

    for action in range(nA):
        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])

    return action_values


def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    """
    Evaluate a policy given a deterministic environment.
    """
    nS = environment.observation_space.n  # Number of states
    V = np.zeros(nS)  # Initialize the value function

    for i in range(int(max_iterations)):
        delta = 0
        for state in range(nS):
            v = 0
            for action, action_probability in enumerate(policy[state]):
                for state_probability, next_state, reward, terminated in environment.P[state][action]:
                    v += action_probability * state_probability * (reward + discount_factor * V[next_state])
            delta = max(delta, abs(V[state] - v))
            V[state] = v

        if delta < theta:
            break

    return V


def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):
    """
    Policy Iteration algorithm to solve a Markov Decision Process (MDP).
    """
    nS = environment.observation_space.n  # Number of states
    nA = environment.action_space.n  # Number of actions
    policy = np.ones((nS, nA)) / nA  # Initialize a random policy

    for i in range(int(max_iterations)):
        V = policy_evaluation(policy, environment, discount_factor)
        stable_policy = True

        for state in range(nS):
            current_action = np.argmax(policy[state])
            action_values = one_step_lookahead(environment, state, V, discount_factor)
            best_action = np.argmax(action_values)

            if current_action != best_action:
                stable_policy = False

            policy[state] = np.eye(nA)[best_action]

        if stable_policy:
            break

    return policy, V


def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    """
    Value Iteration algorithm to solve a Markov Decision Process (MDP).
    """
    nS = environment.observation_space.n  # Number of states
    V = np.zeros(nS)  # Initialize value function

    for i in range(int(max_iterations)):
        delta = 0
        for state in range(nS):
            action_values = one_step_lookahead(environment, state, V, discount_factor)
            best_action_value = np.max(action_values)
            delta = max(delta, abs(V[state] - best_action_value))
            V[state] = best_action_value

        if delta < theta:
            break

    policy = np.zeros((nS, environment.action_space.n))
    for state in range(nS):
        action_values = one_step_lookahead(environment, state, V, discount_factor)
        best_action = np.argmax(action_values)
        policy[state][best_action] = 1.0

    return policy, V
