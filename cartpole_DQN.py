import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import random
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop


# neural network for DQN
def  ourModel(input_shape, action_space):
    x_input = Input(input_shape)

    # input layer of state size 4 and hidden layer with 128 nodes
    x = Dense(512, input_shape=input_shape, activation="relu",kernel_initializer='he_uniform')(x_input)
    # hidden layer with 64 nodes
    x = Dense(256, activation="relu", kernel_initializer='he_uniform')(x)
    # hidden layer with 16 nodes
    x = Dense(64, activation="relu", kernel_initializer='he_uniform')(x)
    # output layer with # of actions: 2 nodes (left right)
    x = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(x)

    model = Model(inputs = x_input, outputs=x, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model



class agent_DQN:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        # default for cartpole: max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.episodes = 1000    # number of episodes to train
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95       # descount rate
        self.epsilon = 1.0      # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000 

        # create the model
        self.model = ourModel(input_shape=(self.state_size,),action_space=self.action_size)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if (len(self.memory) > self.train_start):
            if (self.epsilon > self.epsilon_min):
                self.epsilon *= self.epsilon_decay


    def act(self, state):
        if (np.random.random_sample() <= self.epsilon):
            # act for exploration
            return random.randrange(self.action_size)
        else:
            # act for exploitation
            return np.argmax(self.model.predict(state))

    
    def replay(self):
        if (len(self.memory) < self.train_start):
            return
        
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
        
        # do batch prediction to save speed
        target = self.model.predict(state)
        next_target = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q-value for the action used
            if (done[i]):
                target[i][action[i]] = reward[i]
            else:
                # Standart DQN: choose the max Q-value along the next actions. 
                # Selection and evaluation of action is on the target Q Network
                # Q_max = max_a' {Q_target(s',a')}
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(next_target[i]))

        # train the neural net with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
    

    def load(self, name):
        self.model = load_model(name)


    def save(self, name):
        self.model.save(name)

    
    def run(self):
        for e in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.act(state)
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
                    print("episode: {}/{}, \t score: {}, \t e: {:.2}".format(e, self.episodes, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                        return
                
                self.replay()


    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.episodes):
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
                    print("episode: {}/{}, \t score: {}".format(e, self.episodes, i))
                    break



if __name__ == "__main__":
    # create object
    agent = agent_DQN()

    # DQN learning phase
    #agent.run()

    # test the learned policy
    agent.test()