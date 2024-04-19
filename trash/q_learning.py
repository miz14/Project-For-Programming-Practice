import torch
import random

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
     def policy_function(state):
         probs = torch.ones(n_action) * epsilon / n_action
         q_values = estimator.predict(state)
         best_action = torch.argmax(q_values).item()
         probs[best_action] += 1.0 - epsilon
         action = torch.multinomial(probs, 1).item()
         return action
     return policy_function

def q_learning(env,encoder, estimator, n_episode, n_action, gamma = 1.0, epsilon = 0.1, epsilon_decay=0.99):
    total_reward_episode = [0] * n_episode
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** episode, n_action)
        state, info = env.reset()
        input_image = state 
        output_value = encoder(input_image)
        state = output_value
        is_done = False
        action_count = 0
        lives = 5
        while not is_done:
            is_dead = False
            action_count += 1
            if random.uniform(0, 1) < epsilon:
                action =  env.action_space.sample()
            else:
                action = policy(state)
            next_state, reward, is_done, truncation, info = env.step(action)
            input_image = next_state 
            output_value = encoder(input_image)
            next_state = output_value
            q_values_next = estimator.predict(next_state)
            td_target = reward + gamma * torch.max(q_values_next)
            if info["lives"] < lives:
                lives -= 1
                is_dead = True
            estimator.update(state, action, td_target, is_dead)
            total_reward_episode[episode] += reward
            if is_done:
                break
            if truncation:
                break
            state = next_state
        print(f"episode{episode}")
    return total_reward_episode