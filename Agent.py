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

    def move(self, step):
        self.pos += step
        self.traj = np.append(self.traj, [self.pos], 0)
        
    def Q_learning(sample, gamma=.5, lr=.5):
        # sample = {'s':val, 'a': val, 'r': val, 'sp': val}
        self.Q[sample['s'],sample['a']] = ((1-lr)*self.Q[sample['s'],sample['a']] +
                                           lr*(row['r'] + action_reward[sample['a']] +
                                           gamma*self.Q[sample['sp'],:].max()))
