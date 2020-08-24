import gym
import minerl
import logging
import random
import matplotlib.pyplot as plt

import numpy as np
from numpy import savetxt
from numpy import loadtxt
from os import path

#space_compass = [0,1,-1,2,-2,3,-3,4,-4,6,-6,10,-10,20,-20,45,-45,90,-90,120,-120,170,-170,180,-180]
#space_turn    = [0,1,-1,2,-2,3,-3,4,-4,6,-6,10,-10,20,-20,45,-45,90,-90,120,-120,170,-170,180,-180]
#space_pixel   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 256]
space_turn = [0,1,-1,3,-3,6,-6,10,-10]
space_compass = [0,2,-2,5,-5,10,-10,20,-20,45,-45,90,-90,160,-160]
space_pixel = [0, 80, 160, 250]

# return [pos,val] for value in space
def discretize(value,space):
    #print ("type value",type(value))
    #print ("type space",type(space))
    diff = np.abs(space - value)
    #print ("diff",diff)
    pos = np.argmin(diff)
    #print ("pos",pos)
    #print ("return",[pos,space[pos]])
    return [pos,space[pos]]

def plot_stats(angles, rewards, net_rewards):
    plt.subplot(311)
    plt.plot(angles)
    plt.subplot(312)
    plt.plot(rewards)
    plt.subplot(313)
    plt.plot(net_rewards)
    plt.show()

#logging.basicConfig(level=logging.DEBUG)

def q_index(s, r, g, b):
    return s*len(space_pixel)*len(space_pixel)*len(space_pixel) + r*len(space_pixel)*len(space_pixel) + g*len(space_pixel) + b

# 1. Load Environment and Q-table structure
env = gym.make('MineRLNavigateDense-v0')

Q = np.zeros([len(space_compass)*len(space_pixel)*len(space_pixel)*len(space_pixel),len(space_turn)])

if path.exists('Qvalue.csv'):
    Q = loadtxt('Qvalue.csv', delimiter=',')
# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
print("--------------Q-value------------",Q)

# 2. Parameters of Q-learning
eta = .628
gma = 0.9
epis = 1000
MAX_STEPS = 1500
rev_list = [] # rewards per episode calculate

rewards = []
angles = []
net_rewards = []

# # Q-learn on examples first
# data = minerl.data.make('MineRLNavigateDense-v0')
# for current_state, action, reward, next_state, done \
#     in data.batch_iter(batch_size=1, num_epochs=1, seq_len=1):
#         s,compass = discretize(current_state['compassAngle'][0],space_compass)
#         r,_ = discretize(current_state["pov"][0][0, 32, 32, 0],space_pixel)
#         g,_ = discretize(current_state["pov"][0][0, 32, 32, 1],space_pixel)
#         b,_ = discretize(current_state["pov"][0][0, 32, 32, 2],space_pixel)
#
#         a,turn = discretize(turn,space_turn)
#
#         Q[q_index(s,r,g,b),action] = Q[q_index(s,r,g,b),a] + eta*(reward + gma*np.max(Q[q_index(s1,r1,g1,b1),:]) - Q[q_index(s,r,g,b),a])

# 3. Q-learning Algorithm
for i in range(0,epis):
    # Reset environment
    obs = env.reset()
    done = False
    net_reward = 0
    rewards = []
    angles = []

    # Reduce state space
    #print ("Compass",obs["compassAngle"])
    #print ("Space_compass",space_compass)
    #print ("----------------compass------------")
    s,compass = discretize(obs["compassAngle"],space_compass)
    r,_ = discretize(obs["pov"][32, 32, 0],space_pixel)
    g,_ = discretize(obs["pov"][32, 32, 1],space_pixel)
    b,_ = discretize(obs["pov"][32, 32, 2],space_pixel)

# The Q-Table learning algorithm
    j = 0
    while not done:
        j+=1

        # Choose action from Q table
        a = np.argmax(Q[q_index(s,r,g,b),:])   # index of the maximum deemed reward
        turn = space_turn[a]

        # Mostly Ignore Q for early action
        #   turn += (random.random()*2-1)*180 *(1./(i+1))

        # Modify action to get exploration
        noise = 0
        if random.random() < 0.05:
            noise = (random.random()*2-1)*90
        turn = turn+noise
        # Make sure a is in [-180,180] range
        turn = ((180+turn)%360)-180
        # Reduce action space
        #print("------------------------ call---------------------------")
        turn=np.array(turn)
        a,turn = discretize(turn,space_turn)


        # Expand action to full space
        action = env.action_space.noop()
        action['camera'] = [0, turn]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1

        #Get new state & reward from environment
        obs, reward, done, info = env.step(
            action)

        # Reduce state space
        s1,compass1 = discretize(obs["compassAngle"],space_compass)
        r1,_ = discretize(obs["pov"][32, 32, 0],space_pixel)
        g1,_ = discretize(obs["pov"][32, 32, 1],space_pixel)
        b1,_ = discretize(obs["pov"][32, 32, 2],space_pixel)

        #Update Q-Table with new knowledge
        Q[q_index(s,r,g,b),a] = Q[q_index(s,r,g,b),a] + eta*(reward + gma*np.max(Q[q_index(s1,r1,g1,b1),:]) - Q[q_index(s,r,g,b),a])
        s = s1
        r = r1
        g = g1
        b = b1
        compass = compass1

        net_reward += reward
        rewards.append(net_reward)
        angles.append(obs['compassAngle'])
        ### print("Total reward: ", net_reward)
        print("[{},{}] Reward: {:+09.4f} Total reward: {:+09.4f}".format(i,j, reward, net_reward))

        if j > MAX_STEPS:
            print ("I m done")
            break
    print("--------------Q-value------------",Q)
    net_rewards.append(net_reward)
    savetxt('Qvalue.csv',Q, delimiter=',')
plot_stats(angles, rewards, net_rewards)
