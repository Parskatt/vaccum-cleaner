import numpy as np
from agents import AStar1StepAgent, StupidAgent, Djikstra1StepAgent, TravelingSalesmanBFAgent
from world import World
import cv2

def render(world, agent, h, w,):
    aspect_ratio = h/w
    size = 480
    H, W = int(aspect_ratio*size),int(size/aspect_ratio)
    rendition = world.render(agent)
    shit_cv2 = np.stack((rendition[...,2],rendition[...,1],rendition[...,0]),axis=-1)
    cv2.imshow('world', 
            cv2.resize(shit_cv2,
                (W,H),
                interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)

if __name__ == "__main__":
    from examples import simple_world, difficult_world
    world_state, start_pos, H, W = difficult_world()
    world = World(world_state)
    #agent = StupidAgent(start_pos,N)
    #agent = Djikstra1StepAgent(start_pos,N)
    agent = AStar1StepAgent(start_pos,H, W)
    #agent = TravelingSalesmanBFAgent(start_pos,H,W)
    print(world)
    while world.has_dirt():
        render(world, agent, H, W)
        agent.act(world)
    render(world, agent, H, W)        