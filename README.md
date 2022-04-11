# Vacuum Cleaner Planner

## Install
``pip3 install -r requirements.txt``

## Run
``python3 main.py --help``

## Game Rules

The world consists of vacuum cleaner (agent), in a world consisting of clean walkable tiles, dirty walkable tiles, and non-walkable tiles (osbtacles).
The agent can see the entire world, i.e. the world is fully observable.

The task of the agent is to clean the world with as little energy as possible. 
Each action costs the robot 1 energy.
We do not force the agent to return to its original position.

## Agents

- Stupid: The stupid agent finds the closest dirt, and assumes that by just walking straight in x and then in y it will reach the dirt. This often fails if there is a wall in the way.
- Djikstra: The djisktra agent finds the closest dirt, sets it as a goal, and then runs Djikstras algorithm until the path with lowest cost is found.
- A*: The A* agent additionally uses the L1 distance as heuristic to find the goal faster. Can additionally use the time_horizon argument to only search a specific amount of moves. If this horizon is reached, it will choose the horizon action which minimizes the heuristic (behaviour is ill-defined if multiple actions get the same heuristic value).
- Traveling Salesman: Uses A* to find the distances between all pairs of agent,dirt dirt,dirt. Then solves the TSP by brute force. Typically gives the best results, but scales horribly with num dirt.

## Worlds

- Simple: A small world, which most agents should be able to solve.
- Difficult: A world in which agents targeting the closest dirt usually has to backtrack.
- Trap: A world where agents with a small time horizon typically fail.