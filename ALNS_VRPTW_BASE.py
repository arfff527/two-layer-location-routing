# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time
import pickle as pkl
import os

# 数据结构：解
class Sol():
    def __init__(self):
        self.obj = None # 目标函数值
        self.node_no_seq = [] # 解的编码
        self.route_list = [] # 解的解码
        self.timetable_list = [] # 车辆访问各点的时间
        self.route_distance = None
        self.risk_obj = None

# 数据结构：需求节点
class Node():
    def __init__(self):
        self.id = 0  # 节点id
        self.type = 0  # 医院类型
        self.x_coord = 0  # 节点平面横坐标
        self.y_coord = 0  # 节点平面纵坐标
        self.demand = 0  # 节点需求
        self.infection_demand = 0
        self.start_time = 0  # 节点开始服务时间
        self.end_time = 1440  # 节点结束服务时间
        self.service_time = 0  # 单次服务时长
        self.vehicle_speed = 11.11  # 行驶速度

# 数据结构：车场节点
class Depot():
    def __init__(self):
        self.id = 0 # 节点id
        self.x_coord = 0 # 节点平面横坐标
        self.y_coord = 0  # 节点平面纵坐标
        self.start_time = 0 # 节点开始服务时间
        self.end_time = 1440 # 节点结束服务时间
        self.v_speed = 11.11 # 行驶速度
        self.v_cap = large_vehicle_capacity # 车辆容量

# 数据结构：全局参数
class Model():
    def __init__(self, node_num):
        self.best_sol = None # 全局最优解
        self.sol_list = [] # 解的集合
        self.demand_dict = {}  # 需求节点集合
        self.depot = None  # 车场节点集合
        self.demand_id_list = [] # 需求节点id集合
        self.distance_matrix = {}  # 距离矩阵
        self.population_matrix = {}  # 人口矩阵
        self.related_matrix = np.zeros((node_num, node_num))  # 相似度矩阵
        self.time_matrix = {}  # 时间矩阵
        self.number_of_demands = 0 # 需求点数量
        self.rand_d_max = 0.4  # 随机破坏最大破坏比例
        self.rand_d_min = 0.1  # 随机破坏最小破坏比例
        self.worst_d_min = max(1, int(node_num * 0.05))   # 最坏值破坏最少破坏数量
        self.worst_d_max = max(self.worst_d_min + 1, int(node_num * 0.2))   # 最坏值破坏最多破坏数量
        self.regret_n = max(1, int(node_num * 0.05))  # 后悔值破坏数量
        self.r1 = 30  # 一等得分值
        self.r2 = 18  # 二等得分值
        self.r3 = 12  # 三等得分值
        self.rho = 0.6  # 权重衰减比例
        self.d_weight = np.ones(3) * 10  # 破坏算子权重
        self.d_select = np.zeros(3)  # 破坏算子选择次数
        self.d_score = np.zeros(3)  # 破坏算子得分
        self.d_history_select = np.zeros(3)  # 破坏算子累计选择次数
        self.d_history_score = np.zeros(3)  # 破坏算子累计得分
        self.r_weight = np.ones(3) * 10  # 修复算子权重
        self.r_select = np.zeros(3)  # 修复算子选择次数
        self.r_score = np.zeros(3)  # 修复算子得分
        self.r_history_select = np.zeros(3)  # 修复算子累计选择次数
        self.r_history_score = np.zeros(3)  # 修复算子累计得分

# 读取csv文件
def readCSVFile(model):
    cluster_demand_list, centroids_infection_demand, centroids_common_demand = [0] + whole_demand_list[0], [0] + whole_demand_list[1], [0] + whole_demand_list[2]
    for i in range(1, len(cluster_start_list)):
        node = Node()
        node.id = i - 1
        node.x_coord = float(xc[i])
        node.y_coord = float(yc[i])
        node.type = 0
        node.demand = float(cluster_demand_list[i])
        node.infection_demand = float(centroids_infection_demand[i])
        node.start_time = 0 #int(cluster_start_list[i])
        # 开启关闭TW
        node.end_time = int(cluster_start_list[i])*100 # 1440 if node.infection_demand <= 0 else node.start_time + 100
        node.service_time = 20
        model.demand_dict[node.id] = node
        model.demand_id_list.append(node.id)
    model.number_of_demands = len(model.demand_id_list)
    depot = Depot()
    depot.id = 'd1'
    depot.x_coord = xc[0]
    depot.y_coord = yc[0]
    depot.start_time = 0
    depot.end_time = 1440
    depot.v_speed = 11.11
    depot.v_cap = large_vehicle_capacity
    model.depot = depot

# 初始化参数：计算距离矩阵时间矩阵及初始信息素
def calDistanceTimeMatrix(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = cluster_dict_matrix[from_node_id+1, to_node_id+1]
            population = population_matrix[from_node_id+1, to_node_id+1] #math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2+ (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            relate_fitness = (1 * (dist) + 0.2 * (abs(model.demand_dict[from_node_id].start_time - model.demand_dict[to_node_id].start_time) + 0.01 *
                   abs(model.demand_dict[from_node_id].end_time - model.demand_dict[to_node_id].end_time)) + 1 * (abs(model.demand_dict[from_node_id].demand - model.demand_dict[to_node_id].demand))
                              + 2 * (population))
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
            model.population_matrix[from_node_id, to_node_id] = population
            model.population_matrix[to_node_id, from_node_id] = population
            model.related_matrix[from_node_id, to_node_id] = relate_fitness
            model.time_matrix[from_node_id, to_node_id] = math.ceil(dist/model.depot.v_speed)
            model.time_matrix[to_node_id, from_node_id] = math.ceil(dist/model.depot.v_speed)
        dist = cluster_dict_matrix[from_node_id+1, 0]
        population = population_matrix[from_node_id+1, 0] #math.sqrt((model.demand_dict[from_node_id].x_coord - model.depot.x_coord) ** 2 +(model.demand_dict[from_node_id].y_coord - model.depot.y_coord) ** 2)
        relate_fitness = 1 * (dist) + 0.2 * (abs(
            model.demand_dict[from_node_id].start_time) + 0.01 * abs(model.demand_dict[from_node_id].end_time - model.demand_dict[0].end_time)) + 1 * (abs(model.demand_dict[from_node_id].demand))
        model.distance_matrix[from_node_id, model.depot.id] = dist
        model.distance_matrix[model.depot.id, from_node_id] = dist
        model.population_matrix[from_node_id, model.depot.id] = population
        model.population_matrix[model.depot.id, from_node_id] = population
        #model.related_matrix[from_node_id, model.depot.id] = relate_fitness
        #model.related_matrix[model.depot.id, from_node_id] = relate_fitness
        model.time_matrix[from_node_id, model.depot.id] = math.ceil(dist/model.depot.v_speed)
        model.time_matrix[model.depot.id, from_node_id] = math.ceil(dist/model.depot.v_speed)


# 计算路径费用
def calTravelCost(route_list, model):
    timetable_list=[]
    route_distance = []
    total_distance=0
    risk_obj = 0
    risk_obj_list = []
    for route in route_list:
        timetable = []
        distance = 0
        infection_weight = 0  # 目前负载的传染病物资重量
        for i in range(len(route)):
            if i == 0:
                depot_id=route[i]
                next_node_id=route[i+1]
                travel_time=model.time_matrix[depot_id,next_node_id]
                departure=max(model.depot.start_time,model.demand_dict[next_node_id].start_time-travel_time)
                timetable.append((departure, departure))
                current_risk = 0
            elif 1 <= i <= len(route)-2:
                last_node_id=route[i-1]
                current_node_id=route[i]
                current_node = model.demand_dict[current_node_id]
                travel_time=model.time_matrix[last_node_id,current_node_id]
                arrival=max(timetable[-1][1]+travel_time,current_node.start_time)
                departure=arrival+current_node.service_time
                timetable.append((arrival, departure))
                distance += model.distance_matrix[last_node_id, current_node_id]
                # 增加对于传染risk的计算
                if last_node_id != 'd1':
                    infection_weight += model.demand_dict[last_node_id].demand
                if i == 1:
                    current_risk = 0
                else:
                    # TODO: 这里计算第二个目标值, 优先送priority高, 人口密度大的点。
                    current_risk = model.population_matrix[last_node_id, current_node_id]*infection_weight*0.0001
                # distance += current_risk
                risk_obj += current_risk
                risk_obj_list.append(risk_obj)
            else:
                last_node_id = route[i - 1]
                depot_id = route[i]
                travel_time = model.time_matrix[last_node_id, depot_id]
                departure = timetable[-1][1]+travel_time
                timetable.append((departure, departure))
                distance += model.distance_matrix[last_node_id, depot_id]
                infection_weight += model.demand_dict[last_node_id].demand
                # 增加对于传染risk的计算
               # current_risk = model.population_matrix[last_node_id, depot_id] * infection_weight / (0.25 * 200)
                current_risk = model.population_matrix[last_node_id, depot_id] * infection_weight * 0.0001
                #distance += current_risk
                risk_obj += current_risk
                risk_obj_list.append(risk_obj)
        total_distance += distance
        route_distance.append(distance)
        timetable_list.append(timetable)
    # print("risk_obj",risk_obj)
    return timetable_list, total_distance, route_distance, -risk_obj

# 根据Split结果，提取路径
def extractRoutes(node_no_seq,P,depot_id):
    if len(node_no_seq) == 0:
        return []
    route_list = []
    j = len(node_no_seq)
    while True:
        i = P[node_no_seq[j-1]]
        route = [depot_id]
        route.extend(node_no_seq[k] for k in range(i+1, j))
        route.append(depot_id)
        route_list.append(route)
        j = i + 1
        if i == -1:
            break
    return route_list
# 基于图论的路径分割
def splitRoutes(node_no_seq,model):
    depot = model.depot
    V = {id : float('inf') for id in node_no_seq}
    V[depot.id] = 0
    Pred = {}
    for i in range(len(node_no_seq)):
        n_1 = node_no_seq[i]
        load = 0
        departure = 0
        j = i
        cost = 0
        while True:
            n_2 = node_no_seq[j]
            load = load + model.demand_dict[n_2].demand
            if n_1 == n_2:
                arrival = max(model.demand_dict[n_2].start_time, depot.start_time+model.time_matrix[depot.id, n_2])
                cost = model.distance_matrix[depot.id, n_2] * 2
            else:
                n_3 = node_no_seq[j-1]
                arrival = max(departure+model.time_matrix[n_3, n_2], model.demand_dict[n_2].start_time)
                cost = cost - model.distance_matrix[n_3, depot.id] + model.distance_matrix[n_3, n_2] + \
                       model.distance_matrix[n_2, depot.id]
            departure = arrival + model.demand_dict[n_2].service_time
            # print(load, model.depot.v_cap)
            # print(departure, model.demand_dict[n_2].end_time)
            # print(departure, model.time_matrix[n_2, depot.id], depot.end_time)
            if load <= model.depot.v_cap and departure <= model.demand_dict[n_2].end_time and departure+model.time_matrix[n_2,depot.id]  <= depot.end_time:

                    n_4 = node_no_seq[i-1] if i >= 1 else depot.id
                    if V[n_4]+cost <= V[n_2]:
                        V[n_2] = V[n_4]+cost
                        Pred[n_2] = i-1
                    j = j+1
            if j == len(node_no_seq) or load > model.depot.v_cap:
                break
    return extractRoutes(node_no_seq, Pred, model.depot.id)

# 计算目标函数
def calObj(node_no_seq, model):
    node_no_seq = copy.deepcopy(node_no_seq)
    route_list = splitRoutes(node_no_seq, model)
    # travel cost + risk cost
    timetables, cost, route_distance, risk_obj = calTravelCost(route_list, model)
    return cost, route_list, route_distance, timetables, risk_obj

# 随机构造初始解
def genInitialSol(node_no_seq):
    node_no_seq=copy.deepcopy(node_no_seq)
    random.shuffle(node_no_seq)
    return node_no_seq
# 随机破坏
def createRandomDestory(model):
    d = random.uniform(model.rand_d_min, model.rand_d_max)
    reomve_list = random.sample(range(model.number_of_demands), int(d*(model.number_of_demands-1)))
    return reomve_list

# 相关性破坏 关联结点，每次选择与上一个移除的结点关联度较高的结点进行移除。
def createShawDestory(model, sol):
    current_remove = random.choice(sol.node_no_seq)
    remove_list = [current_remove]
    d = random.randint(model.worst_d_min, model.worst_d_max)
    for index in range(d-1):
        next_remove_list = np.argsort(model.related_matrix[current_remove])
        for node in reversed(next_remove_list):
            if node not in remove_list:
                next_remove = node
        remove_list.append(next_remove)
        current_remove = next_remove
    return remove_list

# 最坏值破坏
def createWorseDestory(model,sol):
    deta_f = []
    for node_no in sol.node_no_seq:
        node_no_seq_ = copy.deepcopy(sol.node_no_seq)
        node_no_seq_.remove(node_no)
        obj, _, _, _, risk_obj = calObj(node_no_seq_, model)
        deta_f.append(sol.obj + sol.risk_obj - obj - risk_obj)
    sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
    d = random.randint(model.worst_d_min, model.worst_d_max)
    remove_list = sorted_id[:d]
    return remove_list

# 随机修复
def createRandomRepair(remove_list,model,sol):
    unassigned_node_no_seq = []
    assigned_node_no_seq = []
    # remove node from current solution
    for i in range(model.number_of_demands):
        if i in remove_list:
            unassigned_node_no_seq.append(sol.node_no_seq[i])
        else:
            assigned_node_no_seq.append(sol.node_no_seq[i])
    # insert
    for node_no in unassigned_node_no_seq:
        if len(assigned_node_no_seq) == 0:
            index = 0
        else:
            index = random.randint(0, len(assigned_node_no_seq) - 1)
        assigned_node_no_seq.insert(index, node_no)
    new_sol = Sol()
    new_sol.node_no_seq = copy.deepcopy(assigned_node_no_seq)
    new_sol.obj, new_sol.route_list, new_sol.route_distance, new_sol.timetable_list, new_sol.risk_obj = calObj(assigned_node_no_seq, model)
    return new_sol
# 贪婪修复
def createGreedyRepair(remove_list,model,sol):
    unassigned_node_no_seq = []
    assigned_node_no_seq = []
    # remove node from current solution
    for i in range(model.number_of_demands):
        if i in remove_list:
            unassigned_node_no_seq.append(sol.node_no_seq[i])
        else:
            assigned_node_no_seq.append(sol.node_no_seq[i])
    #insert
    while len(unassigned_node_no_seq) > 0:
        insert_node_no, insert_index = findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
        try:
            assigned_node_no_seq.insert(insert_index, insert_node_no)
        except:
            print(1)
        unassigned_node_no_seq.remove(insert_node_no)
    new_sol = Sol()
    new_sol.node_no_seq = copy.deepcopy(assigned_node_no_seq)
    new_sol.obj, new_sol.route_list, new_sol.route_distance, new_sol.timetable_list, new_sol.risk_obj = calObj(assigned_node_no_seq, model)
    return new_sol
# 搜索贪婪修复位置
def findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model):
    if len(assigned_node_no_seq) == 0:
        dis = 1000
        nearest_point = unassigned_node_no_seq[0]
        for p in unassigned_node_no_seq:
            cur_dis = model.distance_matrix['d1', p] + model.population_matrix['d1', p]
            if cur_dis < dis:
                dis = cur_dis
                nearest_point = p
        best_insert_node_no = nearest_point
        best_insert_index = 0
    else:
        best_insert_node_no = None
        best_insert_index = None
    best_insert_cost = float('inf')
    assigned_obj, _, _, _, assigned_obj_risk = calObj(assigned_node_no_seq,model)
    for node_no in unassigned_node_no_seq:
        for i in range(len(assigned_node_no_seq)):
            assigned_node_no_seq_ = copy.deepcopy(assigned_node_no_seq)
            assigned_node_no_seq_.insert(i, node_no)
            obj_, _, _, _, obj_risk_ = calObj(assigned_node_no_seq_, model)
            deta_f = obj_ + obj_risk_ - assigned_obj - assigned_obj_risk
            if deta_f < best_insert_cost:
                best_insert_index = i
                best_insert_node_no = node_no
                best_insert_cost = deta_f
    return best_insert_node_no, best_insert_index

# 最大贡献值修复
def createRegretRepair(remove_list,model,sol):
    unassigned_node_no_seq = []
    assigned_node_no_seq = []
    # remove node from current solution
    for i in range(model.number_of_demands):
        if i in remove_list:
            unassigned_node_no_seq.append(sol.node_no_seq[i])
        else:
            assigned_node_no_seq.append(sol.node_no_seq[i])
    # insert
    while len(unassigned_node_no_seq) > 0:
        insert_node_no, insert_index = findRegretInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
        assigned_node_no_seq.insert(insert_index,insert_node_no)
        unassigned_node_no_seq.remove(insert_node_no)
    new_sol = Sol()
    new_sol.node_no_seq = copy.deepcopy(assigned_node_no_seq)
    new_sol.obj, new_sol.route_list, new_sol.route_distance, new_sol.timetable_list, new_sol.risk_obj = calObj(assigned_node_no_seq, model)
    return new_sol
# 搜索最大贡献值
def findRegretInsert(unassigned_node_no_seq, assigned_node_no_seq, model):
    if len(assigned_node_no_seq) == 0:
        dis = 1000
        nearest_point = unassigned_node_no_seq[0]
        for p in unassigned_node_no_seq:
            cur_dis = model.distance_matrix['d1', p] + model.population_matrix['d1', p]
            if cur_dis < dis:
                dis = cur_dis
                nearest_point = p
        opt_insert_node_no = nearest_point
        opt_insert_index = 0
        return opt_insert_node_no, opt_insert_index
    opt_insert_node_no = None
    opt_insert_index = None
    opt_insert_cost = -float('inf')
    for node_no in unassigned_node_no_seq:
        n_insert_cost = np.zeros((len(assigned_node_no_seq), 3))
        for i in range(len(assigned_node_no_seq)):
            assigned_node_no_seq_ = copy.deepcopy(assigned_node_no_seq)
            assigned_node_no_seq_.insert(i, node_no)
            obj_, _, _, _, risk_obj_ = calObj(assigned_node_no_seq_, model)
            n_insert_cost[i, 0] = node_no
            n_insert_cost[i, 1] = i
            n_insert_cost[i, 2] = obj_ + risk_obj_
        n_insert_cost = n_insert_cost[n_insert_cost[:, 2].argsort()]
        deta_f = 0
        for i in range(1, model.regret_n):
            try:
                deta_f = deta_f + n_insert_cost[i, 2] - n_insert_cost[0, 2]
            except:
                deta_f = 0
        if deta_f > opt_insert_cost:
            opt_insert_node_no = int(n_insert_cost[0, 0])
            opt_insert_index = int(n_insert_cost[0, 1])
            opt_insert_cost = deta_f
    return opt_insert_node_no, opt_insert_index
# 选择修复破坏算子
def selectDestoryRepair(model):
    d_weight = model.d_weight
    d_cumsumprob = (d_weight / sum(d_weight)).cumsum()
    d_cumsumprob -= np.random.rand()
    destory_id = list(d_cumsumprob > 0).index(True)

    r_weight = model.r_weight
    r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
    r_cumsumprob -= np.random.rand()
    repair_id = list(r_cumsumprob > 0).index(True)

    return destory_id, repair_id
# 执行破坏算子
def doDestory(destory_id,model,sol):
    if destory_id == 0:
        reomve_list = createRandomDestory(model)
    elif destory_id == 1:
        reomve_list = createWorseDestory(model, sol)
    else:
        reomve_list = createShawDestory(model, sol)
    return reomve_list
# 执行修复算子
def doRepair(repair_id,reomve_list,model,sol):
    if repair_id == 0:
        new_sol = createRandomRepair(reomve_list, model, sol)
    elif repair_id == 1:
        new_sol = createGreedyRepair(reomve_list, model, sol)
    else:
        new_sol = createRegretRepair(reomve_list, model, sol)
    return new_sol
# 重置得分
def resetScore(model):
    model.d_select = np.zeros(3)
    model.d_score = np.zeros(3)

    model.r_select = np.zeros(3)
    model.r_score = np.zeros(3)
# 更新算子权重
def updateWeight(model):
    for i in range(model.d_weight.shape[0]):
        if model.d_select[i] > 0:
            model.d_weight[i] = model.d_weight[i]*(1-model.rho)+model.rho*model.d_score[i]/model.d_select[i]
        else:
            model.d_weight[i] = model.d_weight[i] * (1 - model.rho)
    for i in range(model.r_weight.shape[0]):
        if model.r_select[i] > 0:
            model.r_weight[i] = model.r_weight[i]*(1-model.rho)+model.rho*model.r_score[i]/model.r_select[i]
        else:
            model.r_weight[i] = model.r_weight[i] * (1 - model.rho)
    model.d_history_select = model.d_history_select + model.d_select
    model.d_history_score = model.d_history_score + model.d_score
    model.r_history_select = model.r_history_select + model.r_select
    model.r_history_score = model.r_history_score + model.r_score
# 绘制目标函数收敛曲线
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1, len(obj_list)+1), obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1, len(obj_list)+1)
    if 'random' in case_type:
        plt.savefig('result/random_result/{}_{}_{}_vrptw_alns_performance_result.png'.format(set_num, node_num, case_type))
    else:
        plt.savefig('result/case_result/{}_{}_{}_vrptw_alns_performance_result.png'.format(set_num, node_num, case_type))
    plt.close()
    # plt.show()
# 输出优化结果
def outPut(model,run_time):
    if 'random' in case_type:
        work=xlsxwriter.Workbook(f'result/random_result/alns_{set_num}_{node_num}_{case_type}_result_alns.xlsx')
    else:
        work = xlsxwriter.Workbook(f'result/case_result/alns_{set_num}_{node_num}_{case_type}_result_alns.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0, 0, 'id')
    worksheet.write(0, 1, 'route')
    worksheet.write(0, 2, 'distance')
    worksheet.write(0, 3, 'time')
    worksheet.write(0, 4, 'total_distance')
    worksheet.write(0, 5, 'run_time')
    worksheet.write(1, 4, model.best_sol.obj)
    worksheet.write(1, 5, run_time)
    for id,route in enumerate(model.best_sol.route_list):
        route_str=[str(i)for i in route]
        time_str=[str(i)for i in model.best_sol.timetable_list[id]]
        worksheet.write(id + 1, 0, 'v' + str(id + 1))
        worksheet.write(id + 1, 1, '-'.join(route_str))
        worksheet.write(id + 1, 2, model.best_sol.route_distance[id])
        worksheet.write(id + 1, 3, '-'.join(time_str))
    work.close()
# 绘制优化车辆路径，这里以3个车场为例
def plotRoutes(model):
    for route in model.best_sol.route_list:
        x_coord = [model.depot.x_coord]
        y_coord = [model.depot.y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
            plt.plot(model.demand_dict[node_id].x_coord, model.demand_dict[node_id].y_coord, marker='s', color='b', linewidth=0.5, markersize=5)
            plt.text(model.demand_dict[node_id].x_coord, model.demand_dict[node_id].y_coord, node_id+1, fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.5))
        x_coord.append(model.depot.x_coord)
        y_coord.append(model.depot.y_coord)
        plt.plot(model.depot.x_coord, model.depot.y_coord, marker='s', color='b', linewidth=0.5, markersize=5)
        plt.text(model.depot.x_coord, model.depot.y_coord, 0, fontsize=12,
                 color='red', bbox=dict(facecolor='yellow', alpha=0.5))
        plt.plot(x_coord, y_coord, marker='s', color='b', linewidth=0.5, markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    if 'random' in case_type:
        plt.savefig('result/random_result/{}_{}_{}_vrptw_alns_route_result.png'.format(set_num, node_num, case_type))
    else:
        plt.savefig('result/case_result/{}_{}_{}_vrptw_alns_route_result.png'.format(set_num, node_num, case_type))
    #plt.show()

def run_alns(vrptw_input_dict):
    """
    :param filepath: Xlsx文件路径
    :param rand_d_max:  随机破坏最大破坏比例
    :param rand_d_min:  随机破坏最小破坏比例
    :param worst_d_max: 最坏值破坏最多破坏数量
    :param worst_d_min: 最坏值破坏最少破坏数量
    :param regret_n:  后悔值破坏数量
    :param r1: 一等得分值
    :param r2: 二等得分值
    :param r3: 三等得分值
    :param rho: 权重衰减比例
    :param phi: 退火速率
    :param epochs: 迭代次数
    :param pu: 权重调整步长
    :return:
    """
    for key, value in vrptw_input_dict.items():
        globals()[key] = value
    model = Model(node_num)
    model.rand_d_max = rand_d_max
    model.rand_d_min = rand_d_min
    model.worst_d_min = worst_d_min
    model.worst_d_max = worst_d_max
    model.regret_n = regret_n
    model.r1 = r1
    model.r2 = r2
    model.r3 = r3
    model.rho = rho
    readCSVFile(model)
    calDistanceTimeMatrix(model)
    history_best_obj = []
    history_best_risk_obj = []
    sol = Sol()
    sol.node_no_seq = genInitialSol(model.demand_id_list)
    sol.obj, sol.route_list, sol.route_distance, sol.timetable_list, sol.risk_obj = calObj(sol.node_no_seq, model)
    model.best_sol = copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    history_best_risk_obj.append(sol.risk_obj)
    start_time = time.time()
    run_time = 0
    # epochs = 200
    # if set_num <= 200:
    #     epochs = 600
    #     stop_time = 500
    # else:
    #     stop_time = 100
    stop_time = 1000
    for ep in range(epochs):
        if stop_time < 0:
            break
        T = (sol.obj + sol.risk_obj)*0.2
        resetScore(model)
        for k in range(pu):
            destory_id, repair_id = selectDestoryRepair(model)
            model.d_select[destory_id] += 1
            model.r_select[repair_id] += 1
            reomve_list = doDestory(destory_id, model, sol)
            new_sol = doRepair(repair_id, reomve_list, model, sol)
            # if new_sol.obj < sol.obj or new_sol.risk_obj < sol.risk_obj:
            #     sol = copy.deepcopy(new_sol)
            #     # 找到的解支配最优解
            #     if new_sol.obj < model.best_sol.obj and new_sol.risk_obj < model.best_sol.risk_obj:
            #         model.best_sol = copy.deepcopy(new_sol)
            #         model.d_score[destory_id] += model.r1
            #         model.r_score[repair_id] += model.r1
            #     # 找到的解支配上一个解
            #     elif new_sol.obj < sol.obj and new_sol.risk_obj < sol.risk_obj:
            #         model.d_score[destory_id] += model.r2
            #         model.r_score[repair_id] += model.r2
            #     # 找到的解和上一个解互不支配
            #     elif new_sol.obj < sol.obj or new_sol.risk_obj < sol.risk_obj:
            #         if new_sol.obj < sol.obj:
            #             model.d_score[destory_id] += model.r3
            #             model.r_score[repair_id] += model.r3
            #         else:
            #             model.d_score[destory_id] += 5
            #             model.r_score[repair_id] += 5
            # # 找到的解受最优解支配 and 满足温度接受准则，接受
            # elif abs(new_sol.obj + new_sol.risk_obj - sol.obj - sol.risk_obj) < T:
            #     sol = copy.deepcopy(new_sol)
            #     model.d_score[destory_id] += 3
            #     model.r_score[repair_id] += 3
            if new_sol.obj < sol.obj:
                sol = copy.deepcopy(new_sol)
                if new_sol.obj < model.best_sol.obj:
                    model.best_sol = copy.deepcopy(new_sol)
                    model.d_score[destory_id] += model.r1
                    model.r_score[repair_id] += model.r1
                else:
                    model.d_score[destory_id] += model.r2
                    model.r_score[repair_id] += model.r2
            elif new_sol.obj-sol.obj < T:
                sol = copy.deepcopy(new_sol)
                model.d_score[destory_id] += model.r3
                model.r_score[repair_id] += model.r3
            T = T * phi
            run_time = time.time() - start_time
            print(f"{ep}/{epochs}:{k}/{pu}， best obj: {model.best_sol.obj:.2f}, run: {run_time:.2f}s")
            if history_best_obj[-1] == model.best_sol.obj:
                stop_time -= 1
            history_best_obj.append(model.best_sol.obj)
            history_best_risk_obj.append(model.best_sol.risk_obj)
        updateWeight(model)
    # plotObj(history_best_risk_obj)
    # plotRoutes(model)
    outPut(model, run_time)
    # objective = 0
    # for route in solution.routes:
    #     a = route_cost(route)
    #     objective += a

    print(
        f"random destory weight is {model.d_weight[0]:.3f}\tselect is {model.d_history_select[0]}\tscore is {model.d_history_score[0]:.3f}")
    print(
        f"worse destory weight is {model.d_weight[1]:.3f}\tselect is {model.d_history_select[1]}\tscore is {model.d_history_score[1]:.3f} ")
    print(
        f"shaw destory weight is {model.d_weight[2]:.3f}\tselect is {model.d_history_select[2]}\tscore is {model.d_history_score[2]:.3f} ")

    print(
        f"random repair weight is {model.r_weight[0]:.3f}\tselect is {model.r_history_select[0]}\tscore is {model.r_history_score[0]:.3f}")
    print(
        f"greedy repair weight is {model.r_weight[1]:.3f}\tselect is {model.r_history_select[1]}\tscore is {model.r_history_score[1]:.3f}")
    print(
        f"regret repair weight is {model.r_weight[2]:.3f}\tselect is {model.r_history_select[2]}\tscore is {model.r_history_score[2]:.3f}")
    # 判断有没有更安全的回路
    for route_index, route in enumerate(model.best_sol.route_list):
        orin_demand = sum(model.demand_dict[node_id].infection_demand * node_index for node_index, node_id in enumerate(route[1:-1]))
        reverse_demand = sum(model.demand_dict[node_id].infection_demand * node_index for node_index, node_id in enumerate(reversed(route[1:-1])))
        if orin_demand < reverse_demand:
            model.best_sol.route_list[route_index] = list(reversed(route))
    tour_dict = {}
    tour_time_dict = {}
    for id, route in enumerate(model.best_sol.route_list):
        tour_dict[id+1] = route
        tour_time_dict[id+1] = model.best_sol.timetable_list[id][-1][-1]
    obj = sum(tour_time_dict.values())
    _, _, _, obj_risk = calTravelCost(model.best_sol.route_list, model)
    # [['d1', 0, 1, 5, 4, 'd1'], ['d1', 3, 2, 'd1']]
    # 优化回路
    #print(model.best_sol.route_list)
    #calTravelCost([['d1', 0, 1, 5, 4, 'd1'],['d1', 3, 2, 'd1']], model)
    return tour_dict, tour_time_dict, min(history_best_obj), obj_risk, run_time

import pickle

# if __name__=='__main__':
#     file_name = 'CVRPTW_500_k-means.pkl'
#     with open(file_name, 'rb') as file:
#         vrptw_input_dict = pickle.load(file)
#         data_set = vrptw_input_dict['data_set']
#         first_layer_result = vrptw_input_dict['first_layer_result']
#         a_tour_dict, a_tour_time_dict, a_vrptw_obj, a_run_time = run_alns(vrptw_input_dict)