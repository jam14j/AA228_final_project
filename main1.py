import Agent
import cost_functions
import numpy as np
import matplotlib.pyplot as plt
from snowflake import koch_snowflake

def print_agents(agents):
    print("idx\tpos\tdes_pos")
    for idx, a in enumerate(agents):
        print(f"{idx}\t{a.pos}\t{a.des_pos}")

def plot_agents(agents):
    for a in agents:
        print(a.ini_pos)
        plt.plot(a.ini_pos[0], a.ini_pos[1], 'bo')
        plt.plot(a.traj[:,0], a.traj[:,1])
        plt.plot(a.des_pos[0], a.des_pos[1], 'r*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def hooke_jeeves(agents):
    step = 0.1
    step_list = [np.array([step, 0.0]), np.array([-step, 0.0]),
                 np.array([0.0, step]), np.array([0.0, -step])]
    # base cost
    best_cost = cost_functions.cost_to_destination(agents)
    while best_cost > 0.5:
        for idx, a in enumerate(agents):
            best_step = np.array([0.0, 0.0])
            # step in each direction
            # greedily choose the best move
            for s in step_list:
                a.pos += s
                s_cost = cost_functions.cost_to_destination(agents)
                a.pos -= s
                if s_cost < best_cost:
                    best_cost = s_cost
                    best_step = s
            a.move(best_step)


def main():
    # snowflake
    x, y = koch_snowflake(order=2)
    agent_count = len(x)
    plt.plot(x,y)

    # agent_count = 12
    agents = [0.0 for _ in range(agent_count)]

    for idx in range(agent_count):
        pos = np.array([np.cos(idx*np.pi/6), np.sin(idx*np.pi/6)])
        # des_pos = np.array([10*np.cos((idx+6)*np.pi/6), 10*np.sin((idx+6)*np.pi/6)])
        des_pos = np.array([x[idx], y[idx]])
        agents[idx] = Agent.Agent(pos, des_pos)

    hooke_jeeves(agents)
    plot_agents(agents)

if __name__ == "__main__":
    main()