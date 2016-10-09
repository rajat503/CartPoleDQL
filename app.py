import random
from random import randint
import dql
import gym
env = gym.make('CartPole-v0')
for i_episode in range(2000):
    observation = env.reset()
    for t in range(1000):
        env.render()
        s=observation
        action = dql.getAction(observation)
        if i_episode < 200:
            epsilon = 0.8
        if i_episode >= 200 and i_episode <=500:
            epsilon = 0.4
        if i_episode > 500:
            epsilon = 0.2
        if random.uniform(0,1)<epsilon:
            action = int(random.getrandbits(1))
        observation, reward, done, info = env.step(action)
        dql.learn(s,reward, observation, i_episode)

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break

xx=raw_input()

done = False
observation = env.reset()
while not done:
    env.render()
    print(observation)
    action = dql.getAction(observation)
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
