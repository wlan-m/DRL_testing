import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'    # run on GPU
import random
import gym
import pylab
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
# from keras import backend as K
import cv2

# import needed for threading
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.backend import set_session
# from keras.backend.tensorflow_backend import set_session
import threading
from threading import Thread, Lock
import time

# configure Keras and TensorFlow sessions and graph
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)
K.set_session(sess)
graph = tf.compat.v1.get_default_graph()



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
    actor = Model(inputs = X_input, outputs = action)
    actor.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=lr))
    # Critic --> new
    critic = Model(inputs = X_input, outputs = value)
    critic.compile(loss="mse", optimizer=RMSprop(lr=lr))
    
    return actor, critic


class agent_A3C():
    def __init__(self, env_name):
        # initialize pong environment
        self.env_name = env_name       
        self.env = gym.make(env_name)
        # self.env.seed(0)  
        self.action_size = self.env.action_space.n

        # initialize PG parameters
        self.gamma = 0.99       # discount rate
        self.EPISODES = 2000
        self.episode = 0
        self.max_average = -21.0  # pong specific
        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.lock = Lock()
        self.lr = 0.000025

        # instantiate games and plot memory
        self.scores = []
        self.episodes = []
        self.average = []
        
        self.save_path = 'Models'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.path = '{}_A3C_{}'.format(self.env_name, self.lr)
        self.model_name = os.path.join(self.save_path, self.path)

        # create a actor-critic network model
        self.Actor, self.Critic = OurModel(input_shape= self.state_size, action_space=self.action_size, lr=self.lr)

        # make the predict functions to work while multithreading
        self.Actor._make_predict_function()
        self.Critic._make_predict_function()
        # self.Actor.predict(self.reset(self.env))
        # self.Critic.predict(self.reset(self.env))

        global graph
        graph = tf.get_default_graph()

    
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
    def replay(self, states, actions, rewards):
        # reshape the memory to the appropriate shape for training
        states = np.vstack(states)
        actions = np.vstack(actions)
        # compute the discounted reward
        discounted_r = self.discount_rewards(rewards)

        # get the critic network predictions
        value = self.Critic.predict(states)[:,0]
        # compute advantage
        advantage = discounted_r - value
        
        # training of actor and critic
        self.Actor.fit(states, actions, sample_weight=advantage, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
    

    def load(self, critic_name, actor_name):
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
    

    def get_image(self, frame, image_memory):
        # all these steps that were on a global level for a2c need to be converted
        # to a local level, so they can be done for all environments in parallel
        if image_memory.shape == (1,*self.state_size):
            image_memory = np.squeeze(image_memory)

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
        image_memory = np.roll(image_memory, 1, axis = 0)

        # inserting new frame to free space
        image_memory[0,:,:] = new_frame

        # show image frame   
        #self.imshow(image_memory,0)
        #self.imshow(image_memory,1)
        #self.imshow(image_memory,2)
        #self.imshow(image_memory,3)
        return np.expand_dims(image_memory, axis=0)


    def reset(self, env):
        image_memory = np.zeros(self.state_size)
        frame = env.reset()
        for i in range(self.REM_STEP):
            state = self.get_image(frame, image_memory)
        
        return state

    
    def step(self, action, env, image_memory):
        next_state, reward, done, info = env.step(action)
        next_state = self.get_image(next_state, image_memory)

        return next_state, reward, done, info


    def run(self):
        # the run function is ONLY used for A2C!
        for e in range(self.EPISODES):
            state = self.reset(self.env)
            done = False
            score = 0
            saving = ''

            # instantiate games memory 
            states = []
            actions = []
            rewards = []

            while not done:
                # self.env.render()

                # actor picks an action
                action = self.act(state)
                # new state
                next_state, reward, done, _ = self.step(action, self.env, state)
                # memorize state, action, reward for the training
                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action_onehot] = 1
                actions.append(action_onehot)
                rewards.append(reward)

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
                    self.replay(states, actions, rewards)
                    # reset training memory
                    states = []
                    actions = []
                    rewards = []
        
        # close the environment when finished trainin 
        self.env.close()

    
    # a set of worker agents, each with their own network and environment will be created. 
    # Each of these workers will run on a separate processor thread, so if there will be more 
    # workers than threads on cpu, simply worker will play game slower, this won't give more speed.
    def train(self, n_threads): 
        # we are closing self.env we created before, because we just used it to get environment parameters
        self.env.close()
        # instatiate one environment per thread
        envs = [gym.make(self.env_name) for i in range(n_threads)]

        # create the threads
        threads = [threading.Thread(target=self.train_threading, daemon=True, args=(self, envs[i], i)) for i in range(n_threads)]
        
        for t in threads:
            time.sleep(2)
            t.start()
    

    def train_threading(self, agent, env, thread):
        # we create defined number of environments which we target to train_threading function, 
        # this function will be used to train agents in parallel
        global graph
        with graph.as_default():
            while self.episode < self.EPISODES:
                # reset the episode
                score = 0
                done = False
                saving = ''
                state = self.reset(env)
                # instantiate games memory 
                states = []
                actions = []
                rewards = [] 

                while not done:
                    action = agent.act(state)
                    next_state, reward, done, _ = self.step(action, env, state)

                    states.append(state)
                    action_onehot = np.zeros([self.action_size])
                    action_onehot[action_onehot] = 1
                    actions.append(action_onehot)
                    rewards.append(reward)

                    score += reward
                    state = next_state

                self.lock.acquire()
                self.replay(states, actions, rewards)
                self.lock.release()

                # update episode count
                with self.lock:
                    average = self.plot_model(score, self.episode)
                    # saving the best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        saving = "SAVING"
                    else:
                        saving = ""
                    
                    print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, thread, score, average, saving))
                    if(self.episode < self.EPISODES):
                        self.episode += 1

            env.close()



    
    def test(self, actor_name, critic_name):
        self.load(actor_name, critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        
        self.env.close()




if __name__ == "__main__":
    # create object
    environment = 'Pong-v0'                 # only get every 4th frame
    environment = 'PongDeterministic-v4'    # get every single frame
    agent = agent_A3C(environment)

    # PG learning phase
    # agent.run()               # --> only used for A2C
    agent.train(n_threads=3)    # --> used for A3C

    # test the learned policy
    # agent.test('Models/PongDeterministic-v4_A3C_2.5e-05_Actor.h5', 'Models/PongDeterministic-v4_A3C_2.5e-05_Critic.h5')
