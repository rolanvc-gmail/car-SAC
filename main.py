import gym
import airgym.envs
import numpy as np
from sac_agent import Agent
import torch
from datetime import datetime
import csv


if torch.cuda.is_available():
    print("Using CUDA device:{}".format(torch.cuda.get_device_name(0)))
else:
    print("No CUDA detected. Using CPU")

def main(env):
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_episodes = 10000  #10M
    best_score = env.reward_range[0]
    score_history = []
    scores_only = []
    start_dt = datetime.now()
    score_file = "data/scores-{}.csv".format(start_dt.strftime("%m-%d-%Y-%H:%M:%S"))

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
            action = action.flatten()
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append((i, score))
        scores_only.append(score)
        avg_score = np.mean(scores_only[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, 'score %.1f'% score, 'avg_score %.1f'%avg_score)

    end_dt = datetime.now()
    runtime_diff = end_dt - start_dt
    tot_sec = runtime_diff.total_seconds()
    per_episode = tot_sec / n_episodes
    print("Total Elapsed time is {} seconds or {} sec per episode".format(tot_sec, per_episode))

    with open(score_file, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(score_history)


if __name__ == '__main__':
    environment = gym.make("airgym:car-gym-v0")
    main(environment)




