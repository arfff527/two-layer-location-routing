import json
import math
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import random

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        print(tour,nodes_glo)
        if len(tour) < len(nodes_glo):
            # add subtour elimination constr. for every pair of cities in subtour
            model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                         <= len(tour)-1)

# Given a tuplelist of edges, find the shortest subtour
def subtour(edges):
    unvisited = nodes_glo[:]
    cycle = nodes_glo[:] # Dummy - guaranteed to be replaced
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(thiscycle) <= len(cycle):
            cycle = thiscycle # New shortest subtour
    return cycle

def solve_tsp(nodes, distances, cluster_coordinates):
    global nodes_glo
    nodes_glo = nodes
    dist = distances #{(c1, c2): distance(c1, c2) for c1, c2 in combinations(nodes, 2)}
    # tested with Python 3.7 & Gurobi 9.0.0
    m = gp.Model()
    # Variables: is city 'i' adjacent to city 'j' on the tour?
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
    # Symmetric direction: use dict.update to alias variable with new key
    vars.update({(j, i) : vars[i, j] for i, j in vars.keys()})
    # Constraints: two edges incident to each city
    cons = m.addConstrs(vars.sum(c, '*') == 2 for c in nodes)
    # Callback - use lazy constraints to eliminate sub-tours
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)
    obj = m.ObjVal
    # Retrieve solution
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    tour = subtour(selected)
    assert len(tour) == len(nodes)
    m.dispose()
    gp.disposeDefaultEnv()
    return tour, obj
    # 创建绘图对象
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 6))
    # points = cluster_coordinates
    # tsp_result = tour
    # 绘制连接各个点的线
    # points = [[sublist[1], sublist[0]] for sublist in points]
    # for i in range(len(tsp_result) - 1):
    #     plt.plot([points[tsp_result[i]][0], points[tsp_result[i + 1]][0]],
    #              [points[tsp_result[i]][1], points[tsp_result[i + 1]][1]], 'bo-')
    # plt.plot([points[tsp_result[-1]][0], points[tsp_result[0]][0]],
    #          [points[tsp_result[-1]][1], points[tsp_result[0]][1]], 'bo-')
    #
    # # 标记各个点
    # for idx, point in enumerate(tsp_result):
    #     plt.text(points[point][0], points[point][1], str(points[point]), fontsize=12, ha='center', va='bottom')
    #
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('TSP Result')
    # plt.grid(True)
    # #plt.show()


npoints = 20
seed = 2
# Create n random points in 2D
random.seed(seed)
nodes = list(range(npoints))
points = [(random.randint(0, 4), random.randint(0, 4)) for i in nodes]
# Dictionary of Euclidean distance between each pair of points
distances = {
    (i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
    for i, j in combinations(nodes, 2)
}
tour, m = solve_tsp(nodes, distances, points)
