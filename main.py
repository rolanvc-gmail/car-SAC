import gym
import airgym.envs
import numpy as np
from sac_agent import Agent
from utils import plot_learning_curve

def main(env):
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_episodes = 10000000 # 10M
    best_score = env.reward_range[0]
    score_history = []
    figure_file = "plots/" + "car_gym.png"

    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)  # observation of ndarray(84,84,1)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, 'score %.1f'% score, 'avg_score %.1f'%avg_score)
    if not load_checkpoint:
        x=[ i+1 for i in range(n_episodes)]
        plot_learning_curve(x, score_history, figure_file)



if __name__ == '__main__':
    environment = gym.make("airgym:car-gym-v0")
    main(environment)




