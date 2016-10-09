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
        if i_episode <50:
            epsilon = 0.8
            rate = 1e-6
        if i_episode >50:
            epsilon = 0.8
            rate = 1e-4
        if i_episode >= 200 and i_episode <=300:
            epsilon = 0.4
        if i_episode > 300:
            epsilon = 0.3
        if i_episode > 4000:
            epsilon = 0.02
            rate = 1e-5
        if random.uniform(0,1)<epsilon:
            action = int(random.getrandbits(1))
        observation, reward, done, info = env.step(action)

        if len(replay_memory) == 5000 :
            replay_memory.popleft()
        replay_memory.append([s, action, reward, observation])

        if t % 2 == 0:
            if len(replay_memory) < 100:
                dql_sample.learn([[s, action, reward, observation]], i_episode, rate)
            else:
                sample = [ replay_memory[i] for i in random.sample(range(len(replay_memory)), 100) ]
                dql_sample.learn(sample, i_episode, rate)
        if done:
            if len(replay_memory) < 100:
                dql_sample.learn([[s, action, reward, observation]], i_episode, rate)
            else:
                sample = [ replay_memory[i] for i in random.sample(range(len(replay_memory)), 100) ]
                dql_sample.learn(sample, i_episode, rate)
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break
