import numpy as np

class Agent():
    def __init__(self, pos, des_pos, act_num=4, map_x=20,map_y=20, res=.1):
        # self.reward = -np.inf
        self.ini_pos = pos.copy()
        self.pos = pos.copy()
        self.traj = np.array([pos.copy()])
        self.des_pos = des_pos.copy()
        self.num_actions = act_num
        self.map_x = map_x
        self.map_y = map_y
        self.res = res
        self.num_states = int((map_x / res) * (map_y / res))
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Pi = np.zeros(self.num_states)

    def move(self, step, noise_flag=False, wind_flag=False):
        noise_step = np.array([0.0,0.0])

        if noise_flag:
            # noise_step += np.random.normal(0,0.1*np.linalg.norm(step),len(step))
            p = np.random.randint(1, 10)
            if p == 1:
                noise_step = -1 * np.copy(step)
        if wind_flag:
            # noise_step += np.array([
            #     np.random.normal(0.01,0.1*np.linalg.norm(step),1)[0],
            #     0])
            p = np.random.randint(1, 4)
            if p == 1:
                noise_step += np.array([np.linalg.norm(step), 0])
        self.pos += np.copy(step + noise_step)
        self.pos[0] = np.max((self.pos[0],0))
        self.pos[1] = np.max((self.pos[1],0))
        # subtracting res for a off by one error/zero index error
        self.pos[0] = np.min((self.pos[0],self.map_x-self.res))
        self.pos[1] = np.min((self.pos[1],self.map_y-self.res))
        self.traj = np.append(self.traj, [self.pos], 0)
