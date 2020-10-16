import numpy as np

class Agent():
    def __init__(self, pos, des_pos):
        # self.reward = -np.inf
        self.ini_pos = pos.copy()
        self.pos = pos.copy()
        self.traj = np.array([pos.copy()])
        self.des_pos = des_pos.copy()


    def move(self, step):
        self.pos += step
        self.traj = np.append(self.traj, [self.pos], 0)
