import numpy as np
from gurobipy import GRB, Model, quicksum


def solve_VRPTW(
    population_matrix: np.ndarray,
    distance: np.ndarray,
    coordinate: np.ndarray,
    time_window: np.ndarray,
    demand: np.ndarray,
    infection_demand: np.ndarray,
    service_duration: np.ndarray,
    vehicle_quantity: int,
    vehicle_capacity: float,
    cost_per_distance: float,
    time_per_distance: float,
    big_m: float,
    timelimit: float
):
    """
    node quantity = customer quantity + 2 = n + 2

    the starting depot is node 0 and the ending depot is node n + 1

    time window for node 0 should be [0, 0] and for node n + 1 should be [0, max operating time]

    return: is_feasible, objective value, arc matrix, arrival time matrix

    """

    # define sets
    node_quantity = coordinate.shape[0]
    customer_quantity = node_quantity - 2

    N = range(node_quantity)
    C = range(1, customer_quantity + 1)
    V = range(vehicle_quantity)


    travel_time = np.zeros([node_quantity, node_quantity])
    for i in N:
        for j in N:
            travel_time[i, j] = 0 # time_per_distance * np.hypot(coordinate[i, 0] - coordinate[j, 0], coordinate[i, 1] - coordinate[j, 1])

    # writing mathematical formulation in code
    model = Model("VRPTW")
    # x = model.addVars(node_quantity, node_quantity, vehicle_quantity, vtype=GRB.BINARY)
    # s = model.addVars(node_quantity, vehicle_quantity, vtype=GRB.CONTINUOUS)
    # # 定义每一个点的传染病物资的载货量-+
    # q = model.addVars(node_quantity, vehicle_quantity, vtype=GRB.INTEGER)
    # # 定义每一个点的传染风险
    # ip = model.addVars(node_quantity, node_quantity, vehicle_quantity, vtype=GRB.CONTINUOUS)
    x = {}
    for i in range(node_quantity):
        for j in range(node_quantity):
            for k in range(vehicle_quantity):
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')

    s = {}
    for i in range(node_quantity):
        for k in range(vehicle_quantity):
            s[i, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f's_{i}_{k}')
    # 定义每一个点的传染病物资的载货量
    q = {}
    for i in range(node_quantity):
        for k in range(vehicle_quantity):
            q[i, k] = model.addVar(vtype=GRB.INTEGER, name=f'q_{i}_{k}')
    # 定义每一个点的传染风险
    ip = {}
    for i in range(node_quantity):
        for j in range(node_quantity):
            for k in range(vehicle_quantity):
                ip[i, j, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f'ip_{i}_{j}_{k}')
    model.modelSense = GRB.MINIMIZE

    # 增加对载货量平衡的定义
    for i in N:
        for j in N:
            for k in V:
                model.addConstr(q[j, k] + (1 - x[i, j, k]) * (100*vehicle_capacity) >=
                                 q[i, k] + demand[j], name='c1')
                #model.addConstr(q[i, k] >= q[j, k] - infection_demand[j] - (1 - x[i, j, k]) * (big_m), name='c1')
    # 车辆载量约束
    for i in N:
            for k in V:
                model.addConstr(q[i, k] >= 0, name='c1.1')
                model.addConstr(q[i, k] >= demand[i], name='c1.2')
                model.addConstr(q[i, k] <= vehicle_capacity, name='c1.3')

    # 增加感染率计算公式
    for i in N:
        for j in N:
            for k in V:
                # 这里需要修改，q[i, k]具有滞后性，滞后了一个点
                model.addConstr(ip[i, j, k] == (population_matrix[i, j]*q[i, k]*0.0001), name='c2')

    obj1 = quicksum(x[i, j, k] * distance[i, j] for i in N for j in N for k in V)
    obj2 = quicksum(-ip[i, j, k] * x[i, j, k] for i in N for j in N for k in V)
    model.setObjective(obj1 + obj2)
    model.addConstrs((quicksum(x[i, j, k] for j in N for k in V) == 1 for i in C), name='c3')
    model.addConstrs((quicksum(x[0, j, k] for j in N) == 1 for k in V), name='c4')
    model.addConstrs((quicksum(x[i, customer_quantity + 1, k] for i in N) == 1 for k in V), name='c5')
    model.addConstrs((quicksum(x[i, h, k] for i in N) - quicksum(x[h, j, k] for j in N) == 0 for h in C for k in V), name='c6')
    model.addConstrs((quicksum(demand[i] * quicksum(x[i, j, k] for j in N) for i in C) <= vehicle_capacity for k in V), name='c7')
    model.addConstrs(s[i, k] >= time_window[i, 0] for i in N for k in V)
    model.addConstrs(s[i, k] <= time_window[i, 1] for i in N for k in V)
    model.addConstrs(s[i, k] + distance[i, j]/11.11 + service_duration[i] - big_m * (1 - x[i, j, k]) <= s[j, k] for i in N for j in N for k in V)

    # set timelimit and start solving
    model.Params.Timelimit = timelimit
    model.optimize()
    try:
        model.write("result/second_vrptw.sol")
        model.write("result/second_vrptw_constraints.lp")
    except:
        model.computeIIS()
        model.write("result/second_vrptw_computeIIS.ilp")

    # obtain the results
    is_feasible = True
    obj = 0
    runtime = model.Runtime
    mip_gap = GRB.INFINITY
    result_arc = np.zeros([vehicle_quantity, node_quantity, node_quantity], dtype=int)
    result_arrival_time = np.zeros([node_quantity, vehicle_quantity])
    result_load_quantity = np.zeros([node_quantity, vehicle_quantity])
    result_infectious_risk = np.zeros([vehicle_quantity, node_quantity, node_quantity])
    for k in V:
        for i in N:
            for j in N:
                try:
                    result_infectious_risk[k, i, j] = ip[i, j, k].X * round(x[i, j, k].X)
                except:
                    is_feasible = False
                    break

    for k in V:
        for i in N:
            for j in N:
                try:
                    result_arc[k, i, j] = round(x[i, j, k].X)
                except:
                    is_feasible = False
                    break

    for k in V:
        for i in N:
            try:
                result_arrival_time[i, k] = s[i, k].X
                result_load_quantity[i, k] = q[i, k].X
            except:
                is_feasible = False
                break

    try:
        obj = model.getObjective().getValue()
        mip_gap = model.MIPGap
    except:
        is_feasible = False

    obj1_value = obj1.getValue()
    obj2_value = obj2.getValue()
    print("dis value after optimization:", obj1_value)
    print("trans risk value after optimization:", obj2_value)

    return is_feasible, obj1_value, obj2_value, result_arc, result_arrival_time, runtime, mip_gap, result_load_quantity, result_infectious_risk
