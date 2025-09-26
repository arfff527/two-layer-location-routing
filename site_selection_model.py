import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
from collections import defaultdict
from tsp_exact import *
import time
import pickle
import statistics
import math

def site_selection_milp(input_data):
    n_hospitals = input_data['dimension']
    # coordinates = input_data['node_coord']
    demands_list = input_data['demand']
    print(sum(demands_list))
    # case_type = input_data['case_type']
    small_vehicle_capacity = input_data['small_vehicle_capacity']
    # small_vehicle_endurance = input_data['small_vehicle_endurance']
    score_list = input_data['safe_score']
    # 距离矩阵
    distance_matrix = input_data['edge_weight']
    # 距离矩阵取整
    distance_matrix = np.round(distance_matrix)
    # clients list
    clients_list = [i for i in range(1, n_hospitals)]
    clients_node_pair = [(i, j) for i in clients_list for j in clients_list]
    # 确定中转点的数量，在合理范围内
    tranfer_site_num = math.ceil(sum(demands_list)/small_vehicle_capacity)
    # satisfied_dis_dict = dict()
    # num_select_percent = int(n_hospitals * 0.9)
    # for c in clients_list:
    #     distances_from_point = distance_matrix[c, :]
    #     over_dis_points = np.argsort(distances_from_point)
    #     satisfied_dis_dict[c] = over_dis_points[:num_select_percent]
    # distance matrix
    # 1. 初始化 -----------------------------------------------------------------------------------------------------------------------------------------
    print('start site selection ip')
    start_ip_time = time.time()
    model = Model('site_selection_milp')
    # model.Params.TimeLimit = 20 # limit running time
    # 2. variable -----------------------------------------------------------------------------------------------------------------------------------------
    # x_ij  Define a binary variable, taking value 1 if waste from vertice i has been delivered to transfer center j, otherwise 0
    x = {}
    for i, j in clients_node_pair:
        x[i, j] = model.addVar(vtype=GRB.BINARY, lb=0, name='x(%s,%s)' % (i, j))
    # y_j Define a binary variable, taking value 1 if vertice j has been selected as the transfer center, otherwise 0
    y = {}
    d = {}
    for j in clients_list:
        y[j] = model.addVar(vtype=GRB.BINARY, lb=0, name='y(%s)' % (j))
        d[j] = model.addVar(vtype=GRB.INTEGER, lb=0, name='d(%s)' % (j))
    # 2. constraint -------------------------------------------------------------------------------------
    # 一个医院由一个中转点服务
    model.addConstrs((quicksum(x[i, j] for j in clients_list) == 1 for i in clients_list), name='c1_service_once_contr1')
    # 装载容量限制
    model.addConstrs((quicksum(x[i, j]*demands_list[i] for i in clients_list) <= small_vehicle_capacity for j in clients_list), name='c2_capacity_contr2')
    # 路程限制，不可以超过车辆的负载
    # model.addConstrs((quicksum(distance_matrix[i, j] * x[i, j] for i in clients_list) <= small_vehicle_endurance for j in clients_list), name='c3')
    model.addConstrs((quicksum(distance_matrix[i, j] * x[i, j] for i in clients_list) <= d[j] for j in clients_list), name='c3')
    # 保证有设施的地点不会有需求点供应
    model.addConstrs((x[i, j] <= y[j] for i, j in clients_node_pair), name='c4_contr4')
    # 200米之内不可以有幼儿园和养老院
    # for i in clients_list:
    #     if score_list[i] < 0:
    #         model.addConstr((y[i] == 0), name='c7_safe_constr')
    # 二次剪枝操作
    # 1. 只考虑范围在前n%的点作为选址目标
    # if n_hospitals > 200000:
    #     satisfied_dis_list = satisfied_dis_dict[i]
    #     for j in clients_list:
    #         for i in clients_list:
    #             if i not in satisfied_dis_list:
    #                 model.addConstr((x[i, j] == 0), name='c8_second_purning')
    # 中转站数量限制
    # if tranfer_site_num <= 8:
    #     model.addConstr(5 <= sum(y[i] for i in clients_list), name='c5')
    #     model.addConstr(sum(y[i] for i in clients_list) <= 15, name='c5')
    model.addConstr(sum(y[i] for i in clients_list) <= tranfer_site_num*1.1, name='c5')
    model.addConstr(sum(y[i] for i in clients_list) >= tranfer_site_num, name='c5')
    # objectives
    obj1 = quicksum(y[j] for j in clients_list) + quicksum(distance_matrix[i, j] * x[i, j] for i in clients_list for j in clients_list)
    obj2 = quicksum(y[j] * score_list[j] for j in clients_list)
    obj3 = quicksum(d[j] for j in clients_list)
    obj4 = quicksum(distance_matrix[0, j] * y[j] for j in clients_list)
    # model.setObjective(obj2+obj1+obj3+obj4, GRB.MINIMIZE)
    # 3. 目标函数 --------------------------------------------------------------------------------------------------------
    model.setObjectiveN(obj2 + obj3, index=0, priority=4, name='obj1_min_sites_num')
    model.setObjectiveN(obj1 + obj4, index=1, priority=0, name='obj2_min_site_dis')
    model.Params.TimeLimit = 3600  # 限制运行时间
    # 4. 求解&解析 --------------------------------------------------------------------------------------------------------
    model.optimize()
    end_ip_time = time.time()
    print("ip time: ", end_ip_time - start_ip_time, model.Runtime)
    try:
        model.write("result/site_selection_model_result/{}_site_selection_solution.sol".format(n_hospitals))
        model.write("result/site_selection_model_result/{}_site_selection_constraints.lp".format(n_hospitals))
    except:
        model.computeIIS()
        model.write("result/site_selection_model_result/{}_site_selection_computeIIS.ilp".format(n_hospitals))
    print("obj2_min_site_dis", model.objVal)
    #print("obj1_min_sites_num", model.ObjNVal)
    runtime = model.Runtime
    labels, orin_labels = [0], [0]
    active_arcs = [a for a in clients_node_pair if x[a].x > 0.99]
    # 二次优化，选择距离depot最近的点
    initial_centroids_index = [a for a in clients_list if y[a].x > 0.99]
    for pair in active_arcs:
        i, j = pair
        if i not in initial_centroids_index:
            labels.append(j)
        orin_labels.append(j)
    initial_centroids = input_data['node_coord'][initial_centroids_index]
    node_coordinates = input_data['node_coord']
    n_clusters = len(initial_centroids)
    orin_labels = np.array(orin_labels)
    labels = np.array(labels)

    # # 为每个聚类选择一个颜色
    # colors = [
    #     'red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta',
    #     'lime', 'pink', 'yellow', 'brown', 'grey', 'olive'
    # ]
    # for i in range(n_clusters):
    #     cluster_center = list(set(initial_centroids_index))[i]
    #     all_cluster_points = np.where(orin_labels == cluster_center)[0]
    # # 可视化数据点和聚类中心
    # plt.figure(figsize=(8, 6))
    # for i in range(n_clusters):
    #      cluster_center = list(set(initial_centroids_index))[i]
    #      all_cluster_points = np.where(orin_labels == cluster_center)[0]
    #      for j in all_cluster_points:
    #          # if j == cluster_center:
    #          #     print(1)
    #          #     plt.scatter(node_coordinates[i][1], node_coordinates[i][0], s=200, c=colors[int((i+1)%12)], marker='X', edgecolors='black')
    #          # else:
    #          plt.scatter(node_coordinates[j][0], node_coordinates[j][1], s=50, c=colors[int((i+1)%12)])
    # # 设置图例
    # plt.legend()
    # # 设置标题
    # plt.title('First layer Clustering')
    # # 设置坐标轴标签
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # #显示图形
    # #plt.show()
    # if 'random' in case_type:
    #     plt.savefig('result/random_result/{0}_milp_cluster_result.png'.format(len(node_coordinates)))
    # else:
    #     plt.savefig('result/{0}_milp_cluster_result.png'.format(len(node_coordinates)))
    #plt.close()
    return orin_labels, initial_centroids, n_clusters, initial_centroids_index, model.Runtime

