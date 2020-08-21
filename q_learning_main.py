import gym
import minerl
import logging
import random
import matplotlib.pyplot as plt

import numpy as np

space_compass = [0,1,-1,2,-2,3,-3,4,-4,6,-6,10,-10,20,-20,45,-45,90,-90,120,-120,170,-170,180,-180]
space_turn    = [0,1,-1,2,-2,3,-3,4,-4,6,-6,10,-10,20,-20,45,-45,90,-90,120,-120,170,-170,180,-180]
q_file = 'Qvalue.csv'

# Load experience (Q) from prepared experiments
def load_experience(file, Q = None):
    data = np.loadtxt(file, delimiter=',')
    # TODO: Assert that data matches Q size.
    # TODO: If file is missing, print error, return Q instead
    return data

# Store experience (Q) for later re-use
def save_experience(file, Q):
    np.savetxt(file, Q, delimiter=',')
    return file

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

def plot_stats(angles, rewards):
    plt.subplot(211)
    plt.plot(angles)
    plt.subplot(212)
    plt.plot(rewards)
    plt.show()

#logging.basicConfig(level=logging.DEBUG)


# 1. Load Environment and Q-table structure
env = gym.make('MineRLNavigateDense-v0')

Q = np.zeros([len(space_compass),len(space_turn)])
# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
print("--------------Q-value------------",Q)

# Loading previous experience
Q = load_experience(q_file, Q)
print(Q)

# 2. Parameters of Q-learning
eta = .628
gma = .9
epis = 10
MAX_STEPS = 1500
rev_list = [] # rewards per episode calculate

# 3. Q-learning Algorithm
rewards = []
angles = []
for i in range(0,epis):
    # Reset environment
    obs = env.reset()
    done = False
    net_reward = 0
    max_net_reward = float('-inf')
    rewards = []
    angles = []

    # Reduce state space
    #print ("Compass",obs["compassAngle"])
    #print ("Space_compass",space_compass)
    #print ("----------------compass------------")
    s,compass = discretize(obs["compassAngle"],space_compass)

# The Q-Table learning algorithm
    j = 0
    while not done:
        j+=1

        # Choose action from Q table
        a = np.argmax(Q[s,:])   # index of the maximum deemed reward
        turn = space_turn[a]

        # Mostly Ignore Q for early action
        #   turn += (random.random()*2-1)*180 *(1./(i+1))

        # Modify action to get exploration
        noise = 0
        if random.random() < 0.25:
            noise = (random.random()*2-1)*180
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

        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(reward + gma*np.max(Q[s1,:]) - Q[s,a])
        s = s1
        compass = compass1

        net_reward += reward
        if net_reward > max_net_reward:
            max_net_reward = net_reward
    
        rewards.append(net_reward)
        angles.append(obs['compassAngle'])
        ### print("Total reward: ", net_reward)

        print("[{},{}] Reward: {} Total: {} (of {})".format(i,j, reward, net_reward, max_net_reward))

        if j > MAX_STEPS:
            print ("Aborting at MAX_STEPS!")
            break
    #print("--------------Q-value------------",Q)
    q_file = save_experience(q_file,Q)

data = load_experience(q_file, Q)
print(data)

plot_stats(angles, rewards)
