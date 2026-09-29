"""Observation-only geometric control baseline; no simulator truth access."""
import math
import heapq
import numpy as np


class GeometricController:
    def predict(self, observation, deterministic=True):
        bearing = math.atan2(float(observation[7]), float(observation[8]))
        speed_mps = 0.6 if abs(bearing) > 0.35 else 1.2
        surge_action = speed_mps - 1.0  # max speed is 2 m/s
        yaw_action = max(-1., min(1., 0.8 * bearing / 0.25))
        return np.asarray([surge_action, yaw_action], dtype=np.float32), None


class ObstacleGeometricController:
    """Observation-only grid route planner used solely as an S1 solvability control."""

    def __init__(self):
        self.route = []
        self.route_index = 0

    @staticmethod
    def _decode(observation):
        o = np.asarray(observation, dtype=float)
        x, y = 50 * o[0], 50 * o[1]
        heading = math.atan2(o[4], o[5])
        distance = 100 * math.sqrt(2) * o[6]
        bearing = math.atan2(o[7], o[8]) + heading
        goal = (x + distance * math.cos(bearing), y + distance * math.sin(bearing))
        obstacles = []
        for i in range(10):
            fields = o[27 + 5 * i:32 + 5 * i]
            if fields[4] < 0.5:
                continue
            r = 30 * fields[0]
            angle = math.atan2(fields[1], fields[2]) + heading
            obstacles.append((x + r * math.cos(angle), y + r * math.sin(angle),
                              30 * fields[3]))
        return (x, y), heading, goal, obstacles

    @staticmethod
    def _clear(a, b, obstacles, margin=2.2):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        for x, y, radius in obstacles:
            t = max(0., min(1., ((x-a[0])*dx + (y-a[1])*dy) / max(length2, 1e-9)))
            if math.hypot(a[0] + t*dx - x, a[1] + t*dy - y) <= radius + margin:
                return False
        return True

    def _plan(self, start, goal, obstacles):
        # Visibility graph around circles; shortest clear line segments only.
        nodes = [start, goal]
        for x, y, radius in obstacles:
            for i in range(24):
                angle = 2 * math.pi * i / 24
                p = (x + (radius + 3.0) * math.cos(angle),
                     y + (radius + 3.0) * math.sin(angle))
                if max(abs(p[0]), abs(p[1])) < 47:
                    nodes.append(p)
        adjacency = [[] for _ in nodes]
        for i, a in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if self._clear(a, b, obstacles):
                    d = math.dist(a, b)
                    adjacency[i].append((j, d))
                    adjacency[j].append((i, d))
        distance = [float('inf')] * len(nodes)
        parent = [-1] * len(nodes)
        distance[0] = 0.
        queue = [(0., 0)]
        while queue:
            cost, i = heapq.heappop(queue)
            if cost != distance[i]:
                continue
            if i == 1:
                break
            for j, weight in adjacency[i]:
                candidate = cost + weight
                if candidate < distance[j]:
                    distance[j], parent[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        if parent[1] < 0:
            return [goal]
        result = []
        at = 1
        while at != 0:
            result.append(nodes[at])
            at = parent[at]
        return result[::-1]

    def predict(self, observation, deterministic=True):
        position, heading, goal, obstacles = self._decode(observation)
        if float(observation[-1]) > 0.999 or not self.route:
            self.route = self._plan(position, goal, obstacles)
            self.route_index = 0
        while self.route_index < len(self.route) - 1 and math.dist(
                position, self.route[self.route_index]) < 3.0:
            self.route_index += 1
        target = self.route[self.route_index]
        bearing = math.atan2(target[1]-position[1], target[0]-position[0]) - heading
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        speed = 0.55 if abs(bearing) > 0.55 else (0.85 if abs(bearing) > 0.25 else 1.2)
        if math.dist(position, goal) < 6:
            speed = min(speed, 0.75)
        return np.asarray([speed - 1., np.clip(1.6 * bearing / 0.25, -1., 1.)],
                          dtype=np.float32), None
