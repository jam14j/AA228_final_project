import numpy as np
# import Agent

def cost_to_destination(agents):
    individual_cost = []
    for agent in agents:
        # distance =
        individual_cost.append(np.linalg.norm(agent.pos - agent.des_pos))
        
    overall_cost = np.linalg.norm(np.array(individual_cost))
    
    return overall_cost
        
        
    
        
        
    

