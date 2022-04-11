from functools import partial
import warnings
import numpy as np
from defs import left, right, up, down, clean, dirty
from utils import bisect_right
import itertools


class Agent:
    def __init__(self, start_pos, H, W, *args, **kwargs) -> None:
        self.pos = start_pos
        self.plan = []
        tmp = np.meshgrid(np.arange(W), np.arange(H))
        self.world_grid = np.stack((tmp[1], tmp[0]), axis=-1)
        self.used_energy = 0
    def allowed_directions(self, pos, world):
        return [
            direction
            for direction in (left, right, up, down)
            if world.no_obstacle(pos + direction)
        ]

    def path_to_actions(self, path):
        actions = [partial(self._move, pos=pos) for pos in path[1:]]
        return actions

    def _move(self, pos=None, world=None):
        if world.no_obstacle(pos):
            self.pos = pos
            self.used_energy += 1
    def move_left(self, world):
        new_pos = self.pos + left
        self._move(new_pos, world)

    def move_right(self, world):
        new_pos = self.pos + right
        self._move(new_pos, world)

    def move_up(self, world):
        new_pos = self.pos + up
        self._move(new_pos, world)

    def move_down(self, world):
        new_pos = self.pos + down
        self._move(new_pos, world)

    def suck(self, world):
        world[self.pos[0], self.pos[1]] = clean
        self.used_energy += 1 # maybe dont use energy for suck?

    def act(self, world):
        if self.plan == []:
            self.make_plan(world)
        self.plan.pop(0)(world=world)

    def make_plan(self, world):
        raise NotImplementedError


class StupidAgent(Agent):
    def make_plan(self, world):
        dirts = self.world_grid[world == 1]
        dirt_rel_pos = dirts - self.pos
        dirt_dists = np.abs(dirt_rel_pos).sum(axis=-1)
        goal = dirts[np.argmin(dirt_dists), :]
        if (self.pos == goal).all():
            self.plan = [self.suck]
        else:
            rel_pos = goal - self.pos
            path_y = [
                self.pos + (down if rel_pos[0] > 0 else up)
                for k in range(abs(rel_pos[0]))
            ]
            path_x = [
                self.pos + (right if rel_pos[1] > 0 else left)
                for k in range(abs(rel_pos[1]))
            ]
            self.plan = self.path_to_actions([self.pos] + path_x + path_y)


class Djikstra1StepAgent(Agent):
    def path_from_costs(self, costs, start, goal, world):
        reverse_path = [goal]
        while True:
            pos = reverse_path[-1]
            if (pos == start).all():
                break
            prev_positions = pos + self.allowed_directions(pos, world)
            prev_costs = costs[prev_positions[..., 0], prev_positions[..., 1]]
            reverse_path.append(prev_positions[np.argmin(prev_costs)])
        reverse_path.reverse()
        return reverse_path

    def djikstra(self, start, goal, world):
        costs = 100000 * np.ones_like(world.state)
        costs[start[0], start[1]] = 0
        nodes = [(start, 0)]
        while True:
            p, c = nodes.pop(0)
            if (p == goal).all():
                break
            new_positions = p + self.allowed_directions(p, world)
            new_costs = c + 1
            for new_pos in new_positions:
                if new_costs < costs[new_pos[0], new_pos[1]]:
                    nodes.append((new_pos, new_costs))
                    costs[new_pos[0], new_pos[1]] = new_costs
        return self.path_from_costs(costs, start, goal, world), costs[goal[0], goal[1]]

    def make_plan(self, world):
        dirts = self.world_grid[world == dirty]
        dirt_rel_pos = dirts - self.pos
        dirt_dists = np.abs(dirt_rel_pos).sum(axis=-1)
        goal = dirts[np.argmin(dirt_dists), :]
        if (goal == self.pos).all():
            self.plan = [self.suck]
        else:
            path, cost = self.djikstra(self.pos, goal, world)
            # print(cost)
            self.plan = self.path_to_actions(path)


class AStar1StepAgent(Agent):
    def __init__(self, start_pos, H, W, time_horizon=1000) -> None:
        super().__init__(start_pos, H, W)
        self.time_horizon = time_horizon

    def path_from_costs(self, costs, start, goal, world):
        reverse_path = [goal]
        while True:
            pos = reverse_path[-1]
            if (pos == start).all():
                break
            prev_positions = pos + self.allowed_directions(pos, world)
            prev_costs = costs[prev_positions[..., 0], prev_positions[..., 1]]
            reverse_path.append(prev_positions[np.argmin(prev_costs)])
        reverse_path.reverse()
        return reverse_path

    def astar(self, start, goal, world):
        costs = 100000 * np.ones_like(world.state)
        h_0 = np.abs(goal - start).sum()
        nodes = [(start, 0, 0 + h_0)]
        costs[start[0], start[1]] = 0
        horizon_nodes = []
        at_horizon = False
        while True:
            p, c, _ = nodes.pop(0)

            if (p == goal).all():
                break
            new_positions = p + self.allowed_directions(p, world)
            new_costs = c + 1
            if new_costs > self.time_horizon:
                horizon_nodes.append((p, c, _))
                if nodes:
                    continue
                else:
                    at_horizon = True
                    break
            for new_pos in new_positions:
                if new_costs < costs[new_pos[0], new_pos[1]]:
                    h = np.abs(goal - new_pos).sum()  # could precompute this
                    n = (new_pos, new_costs, new_costs + h)
                    nodes.insert(bisect_right(nodes, n), n)
                    costs[new_pos[0], new_pos[1]] = new_costs
        if at_horizon:
            best_node = min(horizon_nodes, key=lambda x: x[-1])
            best_node_pos = best_node[0]
            return (
                self.path_from_costs(costs, start, best_node_pos, world),
                costs[best_node_pos[0], best_node_pos[1]],
            )
        else:
            return (
                self.path_from_costs(costs, start, goal, world),
                costs[goal[0], goal[1]],
            )

    def make_plan(self, world):
        dirts = self.world_grid[world == dirty]
        dirt_rel_pos = dirts - self.pos
        dirt_dists = np.abs(dirt_rel_pos).sum(axis=-1)
        goal = dirts[np.argmin(dirt_dists), :]
        if (goal == self.pos).all():
            self.plan = [self.suck]
        else:
            path, cost = self.astar(self.pos, goal, world)
            self.plan = self.path_to_actions(path)


class AbstractTravelingSalesmanAgent(AStar1StepAgent):
    def solve_ts(self, cost_matrix, path_matrix):
        raise NotImplementedError
    def make_plan(self, world):
        dirts = self.world_grid[world == dirty]
        cost_matrix = np.zeros((len(dirts) + 1, len(dirts) + 1))
        path_matrix = np.zeros((len(dirts) + 1, len(dirts) + 1), dtype=np.object_)
        for j, j_dirt in enumerate(dirts):
            path, cost = self.astar(self.pos, j_dirt, world)
            path_matrix[j + 1, 0] = path
            cost_matrix[j + 1, 0] = cost
            for i, i_dirt in enumerate(dirts):
                path, cost = self.astar(i_dirt, j_dirt, world)
                cost_matrix[j + 1, i + 1] = cost
                path_matrix[j + 1, i + 1] = path
        best_plan = self.solve_ts(cost_matrix, path_matrix)
        self.plan = best_plan

class TravelingSalesmanBFAgent(AbstractTravelingSalesmanAgent):
    def solve_ts(self, cost_matrix, path_matrix):
        if len(cost_matrix) > 7:
            warnings.warn("Running the brute force method on the TSP is very slow for problems > 7, please reconsider :(")
        perms = [
            list(t) for t in list(itertools.permutations(range(1, len(cost_matrix))))
        ]  # yes, this is not a good idea
        p_costs, p_plans = [], []
        for perm in perms:
            perm.insert(0, 0)
            p_cost = 0
            for departure, destination in zip(perm[:-1], perm[1:]):
                p_cost += cost_matrix[destination, departure]
            p_costs.append(p_cost)
        best_perm = min(zip(perms, p_costs), key=lambda x: x[-1])[0]
        best_plan = []
        for departure, destination in zip(best_perm[:-1], best_perm[1:]):
            best_plan += self.path_to_actions(path_matrix[destination, departure]) + [
                self.suck
            ]
        return best_plan

class TravelingSalesmanRandPermAgent(AbstractTravelingSalesmanAgent):
    def solve_ts(self, cost_matrix, path_matrix):
        inds = range(1, len(cost_matrix))
        perms = [np.random.choice(inds) for _  in min(len(cost_matrix), 7*6*5*4*3*2*1)] 
        p_costs, p_plans = [], []
        for perm in perms:
            perm.insert(0, 0)
            p_cost = 0
            for departure, destination in zip(perm[:-1], perm[1:]):
                p_cost += cost_matrix[destination, departure]
            p_costs.append(p_cost)
        best_perm = min(zip(perms, p_costs), key=lambda x: x[-1])[0]
        best_plan = []
        for departure, destination in zip(best_perm[:-1], best_perm[1:]):
            best_plan += self.path_to_actions(path_matrix[destination, departure]) + [
                self.suck
            ]
        return best_plan
