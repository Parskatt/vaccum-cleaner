from vacuum_cleaner.world import World
from vacuum_cleaner.examples import random_world
from vacuum_cleaner.agents import AStar1StepAgent, TravelingSalesmanMCTSAgent

num_runs = 10
for agent_class in (AStar1StepAgent, TravelingSalesmanMCTSAgent):
    agent_energy = 0
    for _ in range(num_runs):
        world_state, start_pos, H, W = random_world()
        world = World(world_state)
        agent = agent_class(start_pos, H, W)
        while world.has_dirt():
            agent.act(world)
        agent_energy += agent.used_energy
    print(f"{agent} used on average {agent_energy/num_runs} energy.")
