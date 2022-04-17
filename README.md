# Vacuum Cleaner Planner

## Install
``pip3 install -e .``

## Run
``python3 vacuum_cleaner.py --help``
if you want to try out different agents and worlds.

``python3 comparison.py``
if you want to compare the performance of astar to a (monte-carlo approximation) of the optimal traveling salesman solution.

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
- Traveling Salesman MCTS: Same as TS but uses Monte-Carlo Tree search for a set number of iterations to find a good solution to TSP. Scales quadratically with num dirt.

## Worlds

- Simple: A small world, which most agents should be able to solve.
- Difficult: A world in which agents targeting the closest dirt usually has to backtrack.
- Trap: A world where agents with a small time horizon typically fail.
- Random: A large randomly generated world, with lots of dirt and some obstacles. Good for comparing performance of algorithms.

## Note
The random world may sometimes generate degenerate problems. For example when a dirt is surrounded by 4 obstacles. This typically causes the program to either crash or run forever.