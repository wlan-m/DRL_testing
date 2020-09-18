import gym
import numpy as np


class Agent_rand():

    def __init__(self, n_tries, random_action):
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.n_tries = n_tries
        self.random_action = random_action

        self.run_env()
    

    def get_action(self, state):
        # sample a random action from the action space = {0,1}
        # action = random.choice(range(self.action_size))
        action = self.env.action_space.sample()
        return action
    

    def get_action_better(self, state):
        # choose to go left if angle is negative and right if positive
        pole_angle = state[2]
        if (pole_angle < 0):
            action = 0
        else:
            action = 1
        return action


    def run_env(self):
        # make a number of trys
        avg_total_reward = np.zeros(self.n_tries, dtype=int)
        for episode in range(self.n_tries):
            # reset the environment at the beginning
            state = self.env.reset()
            # make 500 timesteps
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
    