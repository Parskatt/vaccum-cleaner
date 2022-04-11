import numpy as np

def simple_world():
    N = 10
    world_state = np.zeros((N,N),dtype=np.int32)
    dirt = np.array([[5,5],[7,3],[2,3]])
    world_state[dirt[:,0],dirt[:,1]] = 1
    obst = np.array([[1,1],[8,2],[9,9],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8]])
    world_state[obst[:,0],obst[:,1]] = 2
    start_pos = np.array([6,3])
    return world_state, start_pos, N, N



def difficult_world():
    H,W = 3,4
    world_state = np.zeros((H,W),dtype=np.int32)
    dirt = np.array([[0,1],[0,2],[0,3],[1,3],[2,3],[2,1]])
    world_state[dirt[:,0],dirt[:,1]] = 1
    obst = np.array([[1,1],[1,2],[2,2]])
    world_state[obst[:,0],obst[:,1]] = 2
    start_pos = np.array([0,0])
    return world_state, start_pos, H, W