from collections import deque
import dql_sample
import gym
import random

env = gym.make('CartPole-v0')

replay_memory = deque()


for i_episode in range(5000):
    observation = env.reset()
    for t in range(1000):
        env.render()
        s=observation
        action = dql_sample.getAction(observation)
        observation, reward, done, info = env.step(action)

        if len(replay_memory) == 2000 :
            replay_memory.popleft()
        replay_memory.append([s, action, reward, observation])

        if done:
            if len(replay_memory) < 100:
                dql_sample.learn([[s, action, reward, observation]], i_episode)
            else:
                sample = [ replay_memory[i] for i in random.sample(range(len(replay_memory)), 100) ]
                dql_sample.learn(sample, i_episode)
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break
