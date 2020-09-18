import gym
import numpy as np
import random
import cv2


class Agent_rand():

    def __init__(self, n_tries, random_action):
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = n_tries
        self.random_action = random_action

        # pictures
        self.ROWS = 160
        self.COLS = 240
        self.REM_STEP = 4
        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))

        self.run_env()
    

    def imshow(self, image, rem_step=0):
        cv2.imshow("cartpole"+str(rem_step), image[rem_step, ...])
        if cv2.waitKey(25) & 0xFF ==ord("q"):
            cv2.destroyAllWindows()
            return
            

    def getImage(self):
        # get simulation image 
        img = self.env.render(mode='rgb_array')
        # convert image to black and white and decrease the resolution
        img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_grey_resized = cv2.resize(img_grey, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_bw_resized = img_grey_resized[img_grey_resized < 255] = 0
        img_bw_resized = img_grey_resized / 255
        # shift the 4 saved images to the back and add the new one to the front
        self.image_memory = np.roll(self.image_memory, 1, axis=0) 
        self.image_memory[0,:,:] = img_bw_resized

        self.imshow(self.image_memory, 0)

        return np.expand_dims(self.image_memory, axis=0)


    def reset(self):
        self.env.reset()
        for i in range(self.REM_STEP):
            state = self.getImage()
        
        return state

    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.getImage()
        
        return next_state, reward, done, info


    def get_action(self, state):
        # sample a random action from the action space = {0,1}
        # action = random.choice(range(self.action_size))
        action = self.env.action_space.sample()
        return action
    

    # # cant be used, because the state is defiened otherwise (as image!)
    # def get_action_better(self, state):
    #     # choose to go left if angle is negative and right if positive
    #     pole_angle = state[2]
    #     if (pole_angle < 0):
    #         action = 0
    #     else:
    #         action = 1
    #     return action


    def run_env(self):
        # make a number of trys
        avg_total_reward = np.zeros(self.EPISODES, dtype=int)
        for episode in range(self.EPISODES):
            # reset the environment at the beginning
            state = self.env.reset()
            # make 500 timesteps --> will hardly ever be reached with random action
            for t in range(500):
                # display the game
                # agent.env.render()
                if (self.random_action == True):
                    # get an action --> random in this case
                    action = self.get_action(state)
                else:
                    # not random action, but still dumb
                    action = self.get_action_better(state)
                # update step
                state, _, done, _ = self.env.step(action)
                # print result
                # print(t, state, reward, done, action)
                # if agent fails, before the end of all timesteps, make next trail
                if done:
                    break
            avg_total_reward[episode] = t
            # close environment
            self.env.close()
        # print mean and standart deviation of the reward
        print("Summary: \tmean = {} \tstandart deviation = {}".format(np.mean(avg_total_reward), np.std(avg_total_reward)) )



if __name__ == "__main__":
    number_of_tries = 100
    action_choice_random = True
    # create agent object
    agent = Agent_rand(number_of_tries, action_choice_random)


    # run_env(number_of_tries, action_choice_random)
    