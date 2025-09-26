import gurobipy as gp
from gurobipy import GRB
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import re
import logging
import math
import random
from collections import defaultdict
from itertools import combinations
from tsp_exact import *
from cvrp_gurobi_solve import *
from ALNS_VRPTW_BASE import *
import pandas as pd
import csv
from pathlib import Path
from typing import Literal
#import graphviz as gv
from site_selection_model import *


def first_layer_tsp(uav_position_list, cluster_center, all_cluster_points, distance_matrix, coordinates):
    # 把UAV服务的点drop掉
    all_cluster_points = [x for x in all_cluster_points if x not in uav_position_list]
    # 算一下UAV服务的点的distance
    uav_cost = 0
    if len(uav_position_list)!=0:
        for uav_point in uav_position_list:
            uav_cost += 2*distance_matrix[uav_point][cluster_center]
    node_num = len(all_cluster_points)
    nodes_list = [i for i in range(node_num)]
    # distance_matrix转换
    cluster_dict_matrix = {}  # np.zeros((node_num, node_num))
    # 保证第一个是TC
    all_cluster_points = list(all_cluster_points)
    if cluster_center in all_cluster_points:
        all_cluster_points.remove(cluster_center)
        all_cluster_points.insert(0, cluster_center)
    coor_index_dict = dict(zip(nodes_list, all_cluster_points))
    for i, j in combinations(nodes_list, 2):
        distance = distance_matrix[coor_index_dict[i]][coor_index_dict[j]]
        cluster_dict_matrix[(i, j)] = distance
    cluster_coordinates = [coordinates[i] for i in coor_index_dict.values()]
    tour, cost = solve_tsp(nodes_list, cluster_dict_matrix, cluster_coordinates)
    tour_index_list = [coor_index_dict[i] for i in tour]
    tour_dict = {i: tour_index_list[i] for i in range(len(tour))}
    return tour_dict, cost, uav_cost



def get_top_k_percent_candidates(candidate_list, score_list, percent):
    """
    从 candidate_list 中挑选得分排名前百分之 `percent` 的原始点编号（即索引），按得分从高到低排序。
    """
    # 取出 candidate_list 中每个点对应的得分
    candidate_scores = [score_list[i] for i in candidate_list]

    # 计算要保留的 top k 个点数量
    top_k = max(1, int(len(candidate_list) * percent))
    score_array = np.array(candidate_scores)
    # 找出得分排名前 top_k 的位置索引
    top_indices = np.argsort(candidate_scores)[-top_k:]
    # 返回原始的点编号
    top_candidates = [candidate_list[i] for i in top_indices]
    return top_candidates


def first_layer_test(case_type, set_num, mode):
    random.seed(42)  # 设置随机数种子
    #----------------------------Data Reader-----------------------------------
    if 'random' in case_type:
        with open(f'data/random_data/CVRP_random_{set_num}.pkl', 'rb') as file:
            input_data = pickle.load(file)
    if 'case' in case_type:
        with open(f'data/case_data/CVRP_case_{set_num}.pkl', 'rb') as file:
            input_data = pickle.load(file)
            input_data['case_type'] = 'case_study'

    input_data = {key.lower(): value for key, value in input_data.items()}
    input_data['demand'] = input_data['demand'][:set_num]
    input_data['dimension'] = set_num
    input_data['edge_weight'] = input_data['edge_weight'][:set_num, :set_num]
    input_data['population_matrix'] = input_data['population_matrix'][:set_num, :set_num]
    input_data['node_coord'] = input_data['node_coord'][:set_num]
    input_data['safe_score'] = input_data['safe_score'][:set_num]
    input_data['set_num'] = set_num
    # 客户点的坐标
    coordinates = input_data['node_coord']
    # 距离矩阵
    distance_matrix = input_data['edge_weight']
    # 人口矩阵
    population_matrix = input_data['population_matrix']
    # 设置不同医院的demand
    # first_layer_demand, centroid_demand = 65, 80
    small_vehicle_capacity, large_vehicle_capacity = 1000, 3000
    input_data['small_vehicle_capacity'] = small_vehicle_capacity
    input_data['small_vehicle_endurance'] = 500
    # 客户点的需求
    #input_data['DEMAND'] = input_data['DEMAND'].replace(500, centroid_demand)
    # input_data['demand'][input_data['demand'] == 500] = centroid_demand
    # input_data['demand'][input_data['demand'] == 200] = 65
    demands = input_data['demand']
    demands[0] = 0
    #----------------------------First layer clustering-----------------------------------
    # build file to save first layer result
    title = ['node_index', 'type', 'cluster_group_index', 'x_coor', 'y_coor', 'tour_index', 'demand(kg)', 'truck distance(m)','drone distance(m)','safety score']
    if 'random' in case_type:
        out_file = open("result/random_result/clustering_result_" + str(len(coordinates)) + ".csv", 'w', encoding='utf-8', newline='' "")
    else:
        out_file = open("result/case_result/clustering_result_" + str(len(coordinates)) + ".csv", 'w', encoding='utf-8', newline='' "")
    first_layer_writer = csv.writer(out_file)
    first_layer_writer.writerow(title)
    orin_labels, initial_centroids, n_clusters, initial_centroids_index, running_time = site_selection_milp(input_data)
    # 随机生成传染病医院index, 传染病医院总数 = 医院个数 * 20%
    infection_no_list = [] # random.sample(list(set([i for i in range(1, len(coordinates))]) - set(initial_centroids_index)), int(set_num * 0.2))
    cost_list = []
    uav_cost = []
    cluster_priority_list = []
    score_list = input_data['safe_score']
    initial_centroids_index = list(set(initial_centroids_index))
    for i in range(n_clusters):
        # TC对应的index
        cluster_center = list(set(initial_centroids_index))[i]
        # 类中所有的点对应的index
        all_cluster_points = np.where(orin_labels == cluster_center)[0]
        cluster_priority_list.append(score_list[cluster_center] + sum([score_list[p] for p in all_cluster_points]))
        # 类中所有的点对应的demand
        cluster_demands = demands[all_cluster_points]
        print(sorted(all_cluster_points))
        # 打印最终的聚类结果
        print(f"Cluster {i}: {all_cluster_points}, Demands: {cluster_demands}, Total Demand: {cluster_demands.sum()}")
        # 所有TC对应的index
        points = np.where(orin_labels == cluster_center)[0]
        # 类中小于2个点
        if len(all_cluster_points) <= 2:
            time_consumption = 0 if len(all_cluster_points) == 1 else distance_matrix[all_cluster_points[0]][all_cluster_points[1]]
            for node_index in all_cluster_points:
                if node_index == 0:
                    first_layer_writer.writerow([node_index, 'Depot', i, coordinates[node_index][0], coordinates[node_index][1], 0, 0, time_consumption, 0, score_list[node_index]])
                    continue  # first layer not consider depot
                if node_index == initial_centroids_index[i]:
                    first_layer_writer.writerow([node_index, 'Transfer Center', i, coordinates[node_index][0], coordinates[node_index][1], 0, demands[node_index], time_consumption, 0, score_list[node_index]])
                else:
                    first_layer_writer.writerow([node_index, 'Collect Station', i, coordinates[node_index][0], coordinates[node_index][1], 1, demands[node_index], time_consumption, 0, score_list[node_index]])
            cost_list.append(time_consumption)
        else:
            # 类中超过2个点，做TSP
            # 加入无人机的配送点，挑选priority前20%的点为无人机配送点，一次只配送一单。
            uav_position_list = get_top_k_percent_candidates(all_cluster_points,score_list, percent=0.2)
            uav_position_list = [x for x in uav_position_list if x != cluster_center]
            tour_dict, truck_cost, drone_cost = first_layer_tsp(uav_position_list, cluster_center, all_cluster_points, distance_matrix, coordinates)
            cost_list.append(truck_cost)
            uav_cost.append(drone_cost)
            # save first layer result, including center node demand, clustering result and tour list in each cluster.
            for node_index in all_cluster_points:
                if node_index in uav_position_list:
                    tour_index = 'u-0'
                else:
                    tour_index = [key for key, val in tour_dict.items() if val == node_index][0]
                if node_index == initial_centroids_index[i]:
                    first_layer_writer.writerow([node_index, 'Transfer Center', i, coordinates[node_index][0], coordinates[node_index][1], tour_index, demands[node_index], truck_cost, drone_cost,score_list[node_index]])
                else:

                    first_layer_writer.writerow([node_index, 'Collect Station', i, coordinates[node_index][0], coordinates[node_index][1], tour_index, demands[node_index], truck_cost, drone_cost, score_list[node_index]])
    out_file.close()
    result_file = 'random_result' if 'random' in case_type else 'case_result'
    first_layer_result = pd.read_csv(f"result/{result_file}/clustering_result_" + str(len(coordinates))+".csv", low_memory=False)
    first_layer_result = first_layer_result.sort_values('node_index')
    first_layer_result.loc[first_layer_result['node_index'].isin(infection_no_list), 'type'] = 'Infectious Collect Station'
    sum_score = sum(first_layer_result[first_layer_result['type'] == 'Transfer Center']['safety score'])
    print("sum_score", sum_score)
    first_layer_result.to_csv(f"result/{result_file}/clustering_result_" + str(len(coordinates)) + "_" + str(mode)+".csv", index=False)
    # cost_list = [int(i) if i != 0 else 0 for i in cost_list]
    # ----------------------------Second layer routing-----------------------------------
    second_clients_number = len(initial_centroids)  # number of clients
    centroids_demand, centroids_infection_demand = [], []
    for index, i in enumerate(list(set(initial_centroids_index))):
        dm = np.where(orin_labels == i)[0]
        centroids_demand.append(sum(demands[j] for j in dm))
        in_dm = first_layer_result[(first_layer_result['cluster_group_index'] == index) & (first_layer_result['type'] == 'Infectious Collect Station')]['node_index']
        centroids_infection_demand.append(sum(demands[j] for j in in_dm))
    centroids_common_demand = [a - b for a, b in zip(centroids_demand, centroids_infection_demand)]
    whole_demand_list = [centroids_demand, centroids_infection_demand, centroids_common_demand]
    xc = [coordinates[0][0]] + [x for x, y in initial_centroids]
    yc = [coordinates[0][1]] + [y for x, y in initial_centroids]
    nodes_list = [i for i in range(second_clients_number+1)]
    cluster_dict_matrix = {}  # np.zeros((node_num, node_num))
    cluster_population_matrix = {}
    coor_index_dict = dict(zip(nodes_list, [0] + list(initial_centroids_index)))
    # 需要把depot也加上
    for i, j in combinations(nodes_list, 2):
        distance = distance_matrix[coor_index_dict[i]][coor_index_dict[j]]
        population = population_matrix[coor_index_dict[i]][coor_index_dict[j]]
        cluster_dict_matrix[(i, j)] = distance
        cluster_dict_matrix[(j, i)] = distance
        cluster_population_matrix[(i, j)] = population
        cluster_population_matrix[(j, i)] = population
    # clients list
    clients_list = [i for i in range(1, second_clients_number+1)]
    # clients+depot list
    all_node_list = [0] + clients_list
    # pair
    all_node_pair = [(i, j) for i in all_node_list for j in all_node_list if i != j]
    # demand list
    cluster_demand_dict = dict(zip(clients_list, [0]+centroids_demand))
    cluster_demand_list = [0] + centroids_demand
    # 原算法是根据逆向回收医废的完成时间制定的时间窗，现在需要更改，根据priority制定
    cluster_start_list = [0] + cluster_priority_list
    vrptw_input_dict = {
        'set_num': set_num,
        'xc': xc,
        'yc': yc,
        'population_matrix': cluster_population_matrix,
        'data_set': input_data['name'],
        'case_type': case_type,
        'distance_matrix': distance_matrix,
        'second_clients_number': second_clients_number,
        'all_node_pair': all_node_pair,
        'cluster_dict_matrix': cluster_dict_matrix,
        'large_vehicle_capacity': large_vehicle_capacity,
        'whole_demand_list': whole_demand_list,
        'cluster_start_list': cluster_start_list,
        'rand_d_max': 0.4,
        'rand_d_min': 0.1,
        'worst_d_min': 5,
        'worst_d_max': 20,
        'regret_n': 5,
        'r1': 30,
        'r2': 20,
        'r3': 10,
        'rho': 0.4,
        'phi': 0.9,
        'epochs': 2000,
        'pu': 5,
        # 'sum_score': sum_score,
        'node_num': len(cluster_start_list) - 1,
         'first_layer_result':[n_clusters, sum_score, sum(cost_list), sum(uav_cost),round(running_time, 3)]
    }
    # 存储第一层的结果数据
    location_mode = 'k-means' if mode[0] == 'k' else 'milp'
    if 'random' in case_type:
        with open(f'data/random_data/CVRPTW_{set_num}_{location_mode}.pkl', 'wb') as file:
            pkl.dump(vrptw_input_dict, file)
    else:
        with open(f'data/case_data/CVRPTW_{set_num}_{location_mode}.pkl', 'wb') as file:
            pkl.dump(vrptw_input_dict, file)
    result = []
    # vrptw_car_num, vrptw_obj, vrptw_obj_risk, vrptw_time, result_load_quantity, result_infectious_risk = cvrptw_gurobi_solve_and_save(vrptw_input_dict, 3600)
    # # res_list = [0, 3, 5, 7, 1, 6, 8, 2, 4, 0]
    # result.append([mode[0]+"-milp", n_clusters, sum(cost_list), vrptw_car_num, vrptw_obj, vrptw_obj_risk, sum_score])
    # tour_dict, tour_time_dict, vrptw_obj, vrptw_obj_risk, run_time = run_alns(vrptw_input_dict)
    # vrptw_car_num = len(tour_dict)
    # result.append([mode[0] + "-alns", n_clusters, sum(cost_list), vrptw_car_num, vrptw_obj, vrptw_obj_risk, sum_score])
    # print(result)
    return result

# first_layer_test('case', 120, 'k')