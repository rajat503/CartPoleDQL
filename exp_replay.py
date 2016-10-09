from collections import deque
import dql_sample
import gym
import random
from random import randint

env = gym.make('CartPole-v0')

replay_memory = deque()


for i_episode in range(50000):
    observation = env.reset()
    for t in range(1000):
        env.render()
        s=observation
        action = dql_sample.getAction(observation)
        if random.uniform(0,1)<0.2:
            action = int(random.getrandbits(1))
        observation, reward, done, info = env.step(action)

        if len(replay_memory) == 2000 :
            replay_memory.popleft()
        replay_memory.append([s, action, reward, observation])

        if t % 5 == 0:
            if len(replay_memory) < 100:
                dql_sample.learn([[s, action, reward, observation]], i_episode)
            else:
                sample = [ replay_memory[i] for i in random.sample(range(len(replay_memory)), 100) ]
                dql_sample.learn(sample, i_episode)

        if done:
            if len(replay_memory) < 100:
                dql_sample.learn([[s, action, reward, observation]], i_episode)
            else:
                sample = [ replay_memory[i] for i in random.sample(range(len(replay_memory)), 100) ]
                dql_sample.learn(sample, i_episode)
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break
