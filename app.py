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
        if random.uniform(1,0)<0.2:
            action = randint(0,1)
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
