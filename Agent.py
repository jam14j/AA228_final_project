import numpy as np

class Agent():
    def __init__(self, pos, des_pos, act_num=4, map_x=20,map_y=20, res=.1):
        # self.reward = -np.inf
        self.ini_pos = pos.copy()
        self.pos = pos.copy()
        self.traj = np.array([pos.copy()])
        self.des_pos = des_pos.copy()
        self.num_actions = act_num
        self.num_states = (map_x / res) * (map_y / res)
        self.Q = np.zeros(self.num_states, self.num_actions)

    def move(self, step, noise_flag=False, wind_flag=False):
        noise_step = np.array([0.0,0.0])

        if noise_flag:
            # noise_step += np.random.normal(0,0.1*np.linalg.norm(step),len(step))
            p = np.random.randint(1, 11)
            if p == 10:
                noise_step = -1 * np.copy(step)
        if wind_flag:
            # noise_step += np.array([
            #     np.random.normal(0.01,0.1*np.linalg.norm(step),1)[0],
            #     0])
            p = np.random.randint(1, 4)
            if p == 1:
                noise_step += np.array([np.linalg.norm(step), 0])
        self.pos += np.copy(step + noise_step)
        self.traj = np.append(self.traj, [self.pos], 0)
        
    def Q_learning(self, sample, gamma=.5, lr=.5):
        # sample = {'s':val, 'a': val, 'r': val, 'sp': val}
        self.Q[sample['s'],sample['a']] = ((1-lr)*self.Q[sample['s'],sample['a']] +
                                           lr*(-1*sample['r'] + gamma*self.Q[sample['sp'],:].max()))
