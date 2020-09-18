import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'    # run on GPU
import random
import gym
import pylab
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import cv2



def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)

    # CNN structure --> works without CNN
    # X = Conv2D(64, 5, strides=(3,3), padding="valid", input_shape=input_shape, activation="elu", data_format="channels_first")(X_input)
    # X = Conv2D(64, 4, strides=(2,2), padding="valid", activation="elu", data_format="channels_first")(X)
    # X = Conv2D(64, 3, strides=(1,1), padding="valid", activation="elu", data_format="channels_first")(X)
    X = Flatten(input_shape=input_shape)(X_input) # directly flatten input image

    # ANN structure: 
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    # Hidden layer with 256 nodes
    # X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    # Hidden layer with 64 nodes
    # X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 6 nodes 
    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    # add "value" parameter to conver PG to an actor critic model
    value = Dense(1,kernel_initializer='he_uniform')(X)

    # Actor --> same as PG
    actor = Model(inputs = X_input, outputs = action, name='Pong_A2C_Actor_model')
    actor.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=lr))
    # Critic --> new
    critic = Model(inputs = X_input, outputs = value, name='Pong_A2C_Critic_model')
    critic.compile(loss="mse", optimizer=RMSprop(lr=lr))
    
    return actor, critic


class agent_A2C():
    def __init__(self, env_name):
        # initialize pong environment
        self.env_name = env_name       
        self.env = gym.make(env_name)
        # self.env.seed(0)  
        self.action_size = self.env.action_space.n

        # initialize PG parameters
        self.gamma = 0.99       # discount rate
        self.EPISODES = 1000
        self.max_average = -21  # pong specific
        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.lr = 0.000025

        # instantiate games and plot memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.scores = []
        self.episodes = []
        self.average = []
        
        self.save_path = 'ModelsDocker'
        self.image_memory = np.zeros(self.state_size)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.model_name = os.path.join(self.save_path, self.path)

        # create a actor network model
        self.Actor, self.Critic = OurModel(input_shape= self.state_size, action_space=self.action_size, lr=self.lr)


    # !!!!!!!!!!!!!!!!
    def remember(self, state, action, reward):
        # store episode actions to memory
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    
    # !!!!!!!!!!!!!!!!
    def act(self, state):
        prediction = self.Actor.predict(state)[0]
        # choose an action randomly according to the probability distribution calculated by the NN actor
        action = np.random.choice(self.action_size, p=prediction)
        return action


    def discount_rewards(self, reward):
        # compute the gamma discounted reward over an episode
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0:
                running_add = 0     # pong specific game boundary
            running_add = running_add * self.gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)   # normalize the result
        discounted_r /= np.std(discounted_r)    # divide by the standart deviation 
        return discounted_r


    # !!!!!!!!!!!!!!!!
    def replay(self):
        # reshape the memory to the appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        # compute the discounted reward
        discounted_r = self.discount_rewards(self.rewards)

        # get the critic network predictions
        value = self.Critic.predict(states)[:,0]
        # compute advantage
        advantage = discounted_r - value
        
        # training of actor and critic
        self.Actor.fit(states, actions, sample_weight=advantage, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)

        # reset the training memory
        self.states = []
        self.actions = []
        self.rewards = []
    

    def load(self, actor_name, critic_name):
        self.Actor = load_model(actor_name, compile=False)
        # the critic model is not required for the testing
        # self.Critic = load_model(critic_name, compile=False)


    def save(self):
        self.Actor.save(self.model_name + '_Actor.h5')
        self.Critic.save(self.model_name + '_Critic.h5')

    
    pylab.figure(figsize=(18,9))
    def plot_model(self, score, epsiode):
        self.scores.append(score)
        self.episodes.append(epsiode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        # if (str(epsiode)[-2:] == "00"):     # much faster than: episode % 100
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        try:
            pylab.savefig(self.path + ".png")
        except OSError:
            pass
        
        return self.average[-1]


    def imshow(self, image, rem_step=0):
        cv2.imshow(self.model_name + str(rem_step), image[rem_step, ...])
        if cv2.waitKey(25) & 0xFF ==ord("q"):
            cv2.destroyAllWindows()
            return
    

    def get_image(self, frame):
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255
        # converting to RGB (OpenCV way)
        #frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)     

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # push our data by 1 frame, similar as deq() function work
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)

        # inserting new frame to free space
        self.image_memory[0,:,:] = new_frame

        # show image frame   
        #self.imshow(self.image_memory,0)
        #self.imshow(self.image_memory,1)
        #self.imshow(self.image_memory,2)
        #self.imshow(self.image_memory,3)
        return np.expand_dims(self.image_memory, axis=0)


    def reset(self):
        frame = self.env.reset()
        for i in range(self.REM_STEP):
            state = self.get_image(frame)
        
        return state

    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.get_image(next_state)

        return next_state, reward, done, info


    def run(self):
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            score = 0
            saving = ''
            while not done:
                # self.env.render()

                # actor picks an action
                action = self.act(state)
                # new state
                next_state, reward, done, _ = self.step(action)
                # memorize state, action, reward for the training
                self.remember(state, action, reward)
                # update the current step
                state = next_state
                score += reward

                if done:
                    average = self.plot_model(score, e)
                    # save the best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        saving = "SAVING"
                    else:
                        saving = ""
                    
                    print("epsisode: {}/{}, \tscore: {}, \taverage: {:.2f} {}".format(e, self.EPISODES, score, average, saving))
                    self.replay()
        
        # close the environment when finished trainin 
        self.env.close()

    
    def test(self, actor_name, critic_name):
        self.load(actor_name, critic_name)
        for e in range(100):
            state = self.reset()
            done = False
            score = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        
        self.env.close()




if __name__ == "__main__":
    # create object
    environment = 'Pong-v0'                 # only get every 4th frame
    environment = 'PongDeterministic-v4'    # get every single frame
    agent = agent_A2C(environment)

    # PG learning phase
    agent.run()

    # test the learned policy
    # agent.test('ModelsDocker/PongDeterministic-v4_A2C_2.5e-05_Actor.h5', 'critic_model_isnt_required')
