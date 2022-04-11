from time import sleep
import numpy as np
from agents import (
    AStar1StepAgent,
    StupidAgent,
    Djikstra1StepAgent,
    TravelingSalesmanBFAgent,
)
from examples import simple_world, difficult_world, trap_world
from world import World
import cv2

world_strings = ["simple", "difficult", "trap"]
worlds = [simple_world, difficult_world, trap_world]
world_parser = {w_s: w for w_s, w in zip(world_strings, worlds)}
agent_strings = ["stupid", "djikstra", "astar", "tsastar"]
agents = [StupidAgent, Djikstra1StepAgent, AStar1StepAgent, TravelingSalesmanBFAgent]
agent_parser = {a_s: a for a_s, a in zip(agent_strings, agents)}


def render(
    world,
    agent,
    h,
    w,
    render_time,
):
    aspect_ratio = h / w
    size = 480
    H, W = int(np.sqrt(aspect_ratio) * size), int(size / np.sqrt(aspect_ratio))
    rendition = world.render(agent)
    shit_cv2 = np.stack(
        (rendition[..., 2], rendition[..., 1], rendition[..., 0]), axis=-1
    )
    cv2.imshow("world", cv2.resize(shit_cv2, (W, H), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(render_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="A Python program for simulating Vacuum Cleaner robots."
    )
    parser.add_argument(
        "--world",
        type=str,
        required=False,
        default="simple",
        choices=world_strings,
        help="Which world to simulate. Default simple.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=False,
        default="stupid",
        choices=agent_strings,
        help="Which agent to simulate. Default stupid.",
    )
    parser.add_argument(
        "--time_horizon",
        type=int,
        required=False,
        default=1000,
        help="Time horizon of agent planning, only applicable to A* agent. Default 1000.",
    )
    parser.add_argument("--dont_render", action='store_true', default=True, help="Whether to render the world graphically.")
    parser.add_argument(
        "--render_time",
        type=int,
        required=False,
        default=100,
        help="Wait time for each rendered frame. If set to 0 will wait until window is closed. Default 100.",
    )

    args = parser.parse_args()
    args.world = world_parser[args.world]
    args.agent = agent_parser[args.agent]
    world_state, start_pos, H, W = args.world()
    world = World(world_state)
    agent = args.agent(start_pos, H, W, time_horizon=args.time_horizon)
    render(world, agent, H, W, args.render_time) if not args.dont_render else None
    sleep(1)
    while world.has_dirt():
        agent.act(world)
        render(world, agent, H, W, args.render_time) if not args.dont_render else None
    print(f"Agent sucked up all dirt, and used {agent.used_energy} energy.")