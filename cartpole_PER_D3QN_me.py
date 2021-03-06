import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import random
import numpy as np
import pylab
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from PER import *


# neural network for DQN
def  ourModel(input_shape, action_space, dueling):
    x_input = Input(input_shape)
    x = x_input

    # input layer of state size 4 and hidden layer with 128 nodes
    x = Dense(512, input_shape=input_shape, activation="relu",kernel_initializer='he_uniform')(x)
    # hidden layer with 64 nodes
    x = Dense(256, activation="relu", kernel_initializer='he_uniform')(x)
    # hidden layer with 16 nodes
    x = Dense(64, activation="relu", kernel_initializer='he_uniform')(x)

    if dueling:
        # D3QN:
        # only investigate the outcome a certain action if it is relevant, which saves recources
        # D3QN separates the value and advantage stream: 
        # value stream has a single output V(s)
        state_value = Dense(1, kernel_initializer='he_uniform')(x)
        state_value = Lambda(lambda s: K.expand_dims(s[:,0], -1), output_shape=(action_space,))(state_value)
        # advantage stream has has two outputs for action a1 and a2
        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(x)
        # lambda layer: user defined input: A(s,a)=1/norm(a)*\sum_a' A(s,a')
        action_advantage = Lambda(lambda a: a[:,:] - K.mean(a[:,:],keepdims=True), output_shape=(action_space,))(action_advantage)
        # aggregate the value and advantage stream back together: the 
        # resulting stream has two outputs Q(s,a1) and Q(s,a2)
        x = Add()([state_value, action_advantage])
    else:
        # regular DDQN: 
        # only a forth Dense layer.
        # output layer with # of actions: 2 nodes (left right)
        x = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(x)

    model = Model(inputs = x_input, outputs=x, name='CartPole_PER_D3QN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model



class agent_DQN:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.seed(0)
        self.env._max_episode_steps = 1000  # default for cartpole: max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        memory_size = 10000
        self.MEMORY = Memory(memory_size)   # new PER version
        self.memory = deque(maxlen=2000)    # old version
        self.use_per = True                 # use new PER instead of old deque

        self.total_episodes = 1000    # number of episodes to train
        self.gamma = 0.95       # descount rate
        self.epsilon = 1.0      # exploration probability at the start
        self.epsilon_min = 0.01 # minimum exploration probability
        self.epsilon_decay = 0.0005 # exponential decay ratefor exploration probability
        self.epsilon_greedy = False # choose epsilon greedy acting strategy
        self.batch_size = 32
        self.train_start = 1000 
        self.tau = 0.1          # for soft updating weights of target model aka polyak averaging
        self.soft_updates = False 
        self.dueling = True     # use a dueling double DQN 

        self.scores = [] 
        self.episodes = []
        self.average = []

        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.model_name = os.path.join(self.Save_Path, "PER_egreedy_D3QN_"+self.env_name+".h5")

        # create the models for DOUBLE dqn
        self.model = ourModel(input_shape=(self.state_size,),action_space=self.action_size, dueling=self.dueling)
        self.target_model = ourModel(input_shape=(self.state_size,),action_space=self.action_size, dueling=self.dueling)

    
    def update_target_model(self):
        if not self.soft_updates:
            self.target_model.set_weights(self.model.get_weights())
            return
        else:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.tau) + q_weight * self.tau
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)


    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.use_per:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))


    def act(self, state, decay_step):
        if self.epsilon_greedy:
            # epsilon greedy strategy
            explore_prob = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)
        else:
            # old strategy
            if self.epsilon >self.epsilon_min:
                self.epsilon *= (1 - self.epsilon_decay)
            explore_prob = self.epsilon
        
        if explore_prob > np.random.rand():
            # choose exploration --> random action
            return random.randrange(self.action_size), explore_prob
        else:
            # choose exploitation --> maximize model
            return np.argmax(self.model.predict(state)), explore_prob

    
    def replay(self):
        if self.use_per:
            # sample with prioretized experience replay of size batch_size
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:         
            # randomly sample a minibatch from the memory of size batch_size
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # TODO: for speedup this could be done on the tensor level, but it's easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        
        # do batch prediction to save speed --> DDQN: predict with TWO networks!
        target = self.model.predict(state)
        target_old = np.array(target)
        next_target = self.model.predict(next_state)
        target_val =self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q-value for the action used
            if (done[i]):
                target[i][action[i]] = reward[i]
            else:
                # Standart DQN: 
                # The same Q-Network calculates the action and evaluates the action 
                # Correlation: Each update would consist of the single online network updating 
                # its weights to better predict what it itself outputs — the agent is trying to 
                # fit to a target value that it itself defines and this can result in the network 
                # quickly updating itself too drastically in an unproductive way --> Overestimation
                # target[i][action[i]] = reward[i] + self.gamma * (np.amax(next_target[i]))
                # DDQN: 
                # The current Q-Network selects the action: a'_max = argmax_a'{Q(s',a')}
                a_tilde_max = np.argmax(next_target[i])
                # the target Q-Network evaluates the action: Q_max = Q_target(s',a'_max)
                target[i][action[i]] = reward[i] + self.gamma * target_val[i][a_tilde_max]

        if self.use_per:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, np.array(action)] - target[indices, np.array(action)])
            # update the priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)

        # train the neural net with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
    

    def load(self, name):
        self.model = load_model(name)


    def save(self, name):
        self.model.save(name)
    

    pylab.figure(figsize=(18,9))
    def plot_model(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores)/len(self.scores))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        ddqn = 'D3QN_'
        softupdate = ''
        egreedy = ''
        per = ''
        if self.soft_updates:
            softupdate = 'soft_'
        if self.epsilon_greedy:
            egreedy = 'egreedy_'
        if self.use_per:
            per = 'PER_'
        try:
            pylab.savefig(ddqn+per+softupdate+egreedy+self.env_name+'.png')
        except OSError:
            pass

        return str(self.average[-1])[:5]

    
    def run(self):
        decay_step = 0      # for epsilon-greedy
        for e in range(self.total_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                #self.env.render()
                decay_step += 1     # for epsilon-greedy
                action, explore_prob = self.act(state, decay_step)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                if (not done or i == self.env._max_episode_steps-1):
                    reward = reward
                else:
                    reward = -100
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # in every step update the target model
                    self.update_target_model()
                    # plot the results
                    average = self.plot_model(i, e) 
                    print("episode: {}/{}, \t score: {}, \t exploration probability: {:.2}, \t average: {}".format(e, self.total_episodes, i, explore_prob, average))
                    if i == self.env._max_episode_steps:
                        print("Saving trained model as ", self.model_name)
                        self.save(self.model_name)
                        break
                
                self.replay()
        self.env.close


    def test(self):
        self.load(self.model_name)
        for e in range(self.total_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, _, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1

                if done:
                    print("episode: {}/{}, \t score: {}".format(e, self.total_episodes, i))
                    # break # use break to view all episodes
                    return # use return to stop after reaching the top score



if __name__ == "__main__":
    # create object
    agent = agent_DQN('CartPole-v1')

    # DQN learning phase
    agent.run()

    # test the learned policy
    # agent.test()