# -*- coding: utf-8 -*-
# @Time    : 2021/10/12 18:29
# @Author  : Python助力交通
# @File    : ACO_MDVRPTW.py
# obj:
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv
import time
# 数据结构：解
class Sol():
    def __init__(self):
        self.obj=None # 目标函数值
        self.node_no_seq=[] # 解的编码
        self.route_list=[] # 解的解码
        self.timetable_list=[] # 车辆访问各点的时间
        self.route_distance = None
# 数据结构：需求节点
class Node():
    def __init__(self):
        self.id=0 # 节点id
        self.x_coord=0 # 节点平面横坐标
        self.y_cooord=0  # 节点平面纵坐标
        self.demand=0 # 节点需求
        self.start_time=0 # 节点开始服务时间
        self.end_time=1440 # 节点结束服务时间
        self.service_time=0 # 单次服务时长
        self.vehicle_speed = 11.11 # 行驶速度
# 数据结构：车场节点
class Depot():
    def __init__(self):
        self.id=0 # 节点id
        self.x_coord=0 # 节点平面横坐标
        self.y_cooord=0  # 节点平面纵坐标
        self.start_time=0 # 节点开始服务时间
        self.end_time=1440 # 节点结束服务时间
        self.v_speed = 11.11 # 行驶速度
        self.v_cap = 7000 # 车辆容量
# 数据结构：全局参数
class Model():
    def __init__(self, node_num):
        self.best_sol=None # 全局最优解
        self.sol_list=[] # 解的集合
        self.demand_dict = {}  # 需求节点集合
        self.depot = None  # 车场节点集合
        self.demand_id_list = [] # 需求节点id集合
        self.distance_matrix = {}  # 距离矩阵
        self.time_matrix = {}  # 时间矩阵
        self.number_of_demands = 0 # 需求点数量
        self.related_matrix = np.zeros((node_num, node_num))
        self.popsize=100 # 种群规模
        self.alpha=2 # 信息启发式因子
        self.beta=3 # 期望启发式因子
        self.Q=1000 # 信息素总量
        self.tau0=1000 # 路径初始信息素
        self.rho=0.5 # 信息素挥发因子
        self.tau={}  # 弧信息素集合
# 读取csv文件
# def readCSVFile(demand_file,depot_file,model):
#     with open(demand_file,'r') as f:
#         demand_reader=csv.DictReader(f)

#         for row in demand_reader:
#             node = Node()
#             node.id = int(row['id'])
#             node.x_coord = float(row['x_coord'])
#             node.y_coord = float(row['y_coord'])
#             node.demand = float(row['demand'])
#             node.start_time=float(row['start_time'])
#             node.end_time=float(row['end_time'])
#             node.service_time=float(row['service_time'])
#             model.demand_dict[node.id] = node
#             model.demand_id_list.append(node.id)
#         model.number_of_demands=len(model.demand_id_list)
#
#     with open(depot_file, 'r') as f:
#         depot_reader = csv.DictReader(f)
#         for row in depot_reader:
#             depot = Depot()
#             depot.id = row['id']
#             depot.x_coord = float(row['x_coord'])
#             depot.y_coord = float(row['y_coord'])
#             depot.start_time=float(row['start_time'])
#             depot.end_time=float(row['end_time'])
#             depot.v_speed = float(row['v_speed'])
#             depot.v_cap = float(row['v_cap'])
#             model.depot = depot
# 初始化参数：计算距离矩阵时间矩阵及初始信息素
def readCSVFile(model):
    cluster_demand_list, centroids_infection_demand, centroids_common_demand = [0] + whole_demand_list[0], [0] + whole_demand_list[1], [0] + whole_demand_list[2]
    max_dis = distance_matrix.max()
    for i in range(1, len(cluster_start_list)):
        node = Node()
        node.id = i - 1
        node.x_coord = float(xc[i])
        node.y_coord = float(yc[i])
        node.type = 0
        node.demand = float(cluster_demand_list[i])
        node.infection_demand = float(centroids_infection_demand[i])
        node.start_time = int(cluster_start_list[i])
        # if max_dis < 50:
        #     node.end_time = 1440 if node.infection_demand <= 0 else node.start_time + 180
        # else:
        #     node.end_time = max_dis*10 if node.infection_demand <= 0 else node.start_time + max_dis
        # 开启关闭TW
        node.end_time = int(cluster_start_list[i])*100 #1440 if node.infection_demand <= 0 else node.start_time + 180
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

def calDistanceTimeMatrix(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = cluster_dict_matrix[from_node_id+1, to_node_id+1]#math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2+ (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            relate_fitness = 1 * (dist) + 0.2 * (abs(model.demand_dict[from_node_id].start_time - model.demand_dict[to_node_id].start_time) + 0.01 *
                   abs(model.demand_dict[from_node_id].end_time - model.demand_dict[to_node_id].end_time)) + 1 * (abs(model.demand_dict[from_node_id].demand - model.demand_dict[to_node_id].demand))
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
            model.related_matrix[from_node_id, to_node_id] = relate_fitness
            model.time_matrix[from_node_id,to_node_id] = math.ceil(dist/model.depot.v_speed)
            model.time_matrix[to_node_id,from_node_id] = math.ceil(dist/model.depot.v_speed)
            model.tau[from_node_id, to_node_id] = model.tau0
            model.tau[to_node_id, from_node_id] = model.tau0
        dist = cluster_dict_matrix[from_node_id+1, 0]#math.sqrt((model.demand_dict[from_node_id].x_coord - model.depot.x_coord) ** 2 +(model.demand_dict[from_node_id].y_coord - model.depot.y_coord) ** 2)
        relate_fitness = 1 * (dist) + 0.2 * (abs(
            model.demand_dict[from_node_id].start_time) + 0.01 * abs(model.demand_dict[from_node_id].end_time - model.demand_dict[0].end_time)) + 1 * (abs(model.demand_dict[from_node_id].demand))
        model.distance_matrix[from_node_id, model.depot.id] = dist
        model.distance_matrix[model.depot.id, from_node_id] = dist
        #model.related_matrix[from_node_id, model.depot.id] = relate_fitness
        #model.related_matrix[model.depot.id, from_node_id] = relate_fitness
        model.time_matrix[from_node_id,model.depot.id] = math.ceil(dist/model.depot.v_speed)
        model.time_matrix[model.depot.id,from_node_id] = math.ceil(dist/model.depot.v_speed)
# 计算路径费用
def calTravelCost(route_list,model):
    timetable_list=[]
    route_distance = []
    total_distance=0
    for route in route_list:
        timetable=[]
        distance = 0
        for i in range(len(route)):
            if i == 0:
                depot_id=route[i]
                next_node_id=route[i+1]
                travel_time=model.time_matrix[depot_id,next_node_id]
                departure=max(model.depot.start_time,model.demand_dict[next_node_id].start_time-travel_time)
                timetable.append((departure,departure))
            elif 1<= i <= len(route)-2:
                last_node_id=route[i-1]
                current_node_id=route[i]
                current_node = model.demand_dict[current_node_id]
                travel_time=model.time_matrix[last_node_id,current_node_id]
                arrival=max(timetable[-1][1]+travel_time,current_node.start_time)
                departure=arrival+current_node.service_time
                timetable.append((arrival,departure))
                distance += model.distance_matrix[last_node_id, current_node_id]
            else:
                last_node_id = route[i - 1]
                depot_id=route[i]
                travel_time = model.time_matrix[last_node_id,depot_id]
                departure = timetable[-1][1]+travel_time
                timetable.append((departure,departure))
                distance +=model.distance_matrix[last_node_id,depot_id]
        total_distance += distance
        route_distance.append(distance)
        timetable_list.append(timetable)
    return timetable_list,total_distance,route_distance
# 根据Split结果，提取路径
def extractRoutes(node_no_seq,P,depot_id):
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
    depot=model.depot
    V={id:float('inf') for id in node_no_seq}
    V[depot.id]=0
    Pred={}
    for i in range(len(node_no_seq)):
        n_1=node_no_seq[i]
        load =0
        departure=0
        j=i
        cost=0
        while True:
            n_2 = node_no_seq[j]
            load += model.demand_dict[n_2].demand
            if n_1 == n_2:
                arrival= max(model.demand_dict[n_2].start_time,depot.start_time+model.time_matrix[depot.id,n_2])
                cost = model.distance_matrix[depot.id, n_2] * 2
            else:
                n_3=node_no_seq[j-1]
                arrival= max(departure+model.time_matrix[n_3,n_2],model.demand_dict[n_2].start_time)
                cost = cost - model.distance_matrix[n_3, depot.id] + model.distance_matrix[n_3, n_2] + \
                       model.distance_matrix[n_2, depot.id]
            departure = arrival + model.demand_dict[n_2].service_time
            if load <= model.depot.v_cap and departure<= model.demand_dict[n_2].end_time and departure+model.time_matrix[n_2,depot.id]  <= depot.end_time:
                n_4=node_no_seq[i-1] if i>=1 else depot.id
                if V[n_4]+cost <= V[n_2]:
                    V[n_2]=V[n_4]+cost
                    Pred[n_2]=i-1
                j=j+1
            if j==len(node_no_seq) or load > model.depot.v_cap:
                break
    return extractRoutes(node_no_seq,Pred,model.depot.id)
# 计算目标函数
def calObj(node_no_seq,model):
    node_no_seq=copy.deepcopy(node_no_seq)
    route_list = splitRoutes(node_no_seq, model)
    # travel cost
    timetables,cost,route_distance =calTravelCost(route_list,model)
    return cost,route_list,route_distance,timetables
# 蚂蚁移动
def movePosition(model):
    sol_list=[]
    local_sol=Sol()
    local_sol.obj=float('inf')
    for k in range(model.popsize):
        #随机初始化蚂蚁为止
        node_no_seq=[int(random.randint(0,len(model.demand_id_list)-1))]
        all_nodes_id=copy.deepcopy(model.demand_id_list)
        all_nodes_id.remove(node_no_seq[-1])
        #确定下一个访问节点
        while len(all_nodes_id)>0:
            next_node_no=searchNextNode(model,node_no_seq[-1],all_nodes_id)
            node_no_seq.append(next_node_no)
            all_nodes_id.remove(next_node_no)
        sol=Sol()
        sol.node_no_seq=node_no_seq
        sol.obj,sol.route_list,sol.route_distance,sol.timetable_list = calObj(sol.node_no_seq,model)
        sol_list.append(sol)
        if sol.obj<local_sol.obj:
            local_sol=copy.deepcopy(sol)
    model.sol_list=copy.deepcopy(sol_list)
    if local_sol.obj<model.best_sol.obj:
        model.best_sol=copy.deepcopy(local_sol)
# 搜索下一移动节点
def searchNextNode(model,current_node_id,SE_List):
    prob=np.zeros(len(SE_List))
    for i,node_id in enumerate(SE_List):
        eta=1/model.distance_matrix[current_node_id,node_id]
        tau=model.tau[current_node_id,node_id]
        prob[i]=((eta**model.beta)*(tau**model.alpha))
    #采用轮盘法选择下一个访问节点
    cumsumprob=(prob/sum(prob)).cumsum()
    cumsumprob -= np.random.rand()
    next_node_id= SE_List[list(cumsumprob > 0).index(True)]
    return next_node_id
# 更新路径信息素
def upateTau(model):
    rho=model.rho
    for k in model.tau.keys():
        model.tau[k]=(1-rho)*model.tau[k]
    #根据解的node_no_seq属性更新路径信息素（TSP问题的解）
    for sol in model.sol_list:
        nodes_id=sol.node_no_seq
        for i in range(len(nodes_id)-1):
            from_node_id=nodes_id[i]
            to_node_id=nodes_id[i+1]
            model.tau[from_node_id,to_node_id]+=model.Q/sol.obj
# 绘制目标函数收敛曲线
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()
# 输出优化结果
def outPut(model,run_time):
    work=xlsxwriter.Workbook('result.xlsx')
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
        x_coord=[model.depot.x_coord]
        y_coord=[model.depot.y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot.x_coord)
        y_coord.append(model.depot.y_coord)
        plt.plot(x_coord, y_coord, marker='s', color='b', linewidth=0.5, markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.show()
# 主程序
def run_aco(vrptw_input_dict,Q,tau0,alpha,beta,rho,epochs,popsize):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param Q:信息素总量
    :param tau0: 路径信息素初始值
    :param alpha:信息启发式因子
    :param beta:期望启发式因子
    :param rho:信息挥发因子
    :param epochs:迭代次数
    :param popsize:蚁群规模
    :return:
    """
    for key, value in vrptw_input_dict.items():
        globals()[key] = value
    model = Model(node_num)
    model.alpha=alpha
    model.beta=beta
    model.Q=Q
    model.tau0=tau0
    model.rho=rho
    model.popsize=popsize
    sol=Sol()
    sol.obj=float('inf')
    model.best_sol=sol
    history_best_obj = []
    readCSVFile(model)
    calDistanceTimeMatrix(model)
    start_time = time.time()
    run_time = 0
    for ep in range(epochs):
        movePosition(model)
        upateTau(model)
        history_best_obj.append(model.best_sol.obj)
        run_time = time.time() - start_time
        print(f"{ep}/{epochs}， best obj: {model.best_sol.obj:.2f}, run: {run_time:.2f}s")
    #plotObj(history_best_obj)
    #plotRoutes(model)
    outPut(model,run_time)
    return model.best_sol.route_list, model.best_sol.obj, run_time

# import pickle
# if __name__=='__main__':
#     file_name = 'CVRPTW_500_milp.pkl' # 第一层的结果文件
#     with open(file_name, 'rb') as file:
#         vrptw_input_dict = pickle.load(file)
#         data_set = vrptw_input_dict['data_set']
#         first_layer_result = vrptw_input_dict['first_layer_result']
#     aco_vrptw_tour ,aco_vrptw_obj, aco_runtime = run(vrptw_input_dict,Q=10,tau0=10,alpha=1,beta=5,rho=0.1,epochs=600,popsize=60)
