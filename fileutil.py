import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from tabulate import tabulate



def load_chengdu_dataset(vrptw_input_dict):
    """
    get processed (node 0 and node n + 1 added) matrices from xml datasets

    assume node 0 is the depot in the xml file

    return: coordinate, time_window, demand, service_duration, vehicle_quantity, vehicle_capacity

    """
    #for key, value in vrptw_input_dict.items():
     #   globals()[key] = value
    xc, yc = vrptw_input_dict['xc'], vrptw_input_dict['yc']
    cluster_start_list = vrptw_input_dict['cluster_start_list']
    coordinate = [[a, b] for a, b in zip(xc, yc)]
    coordinate.append(coordinate[0])
    coordinate = np.array(coordinate)
    whole_demand_list = vrptw_input_dict['whole_demand_list']
    cluster_demand_list, centroids_infection_demand, centroids_common_demand = [0] + whole_demand_list[0]+[0], [0] + whole_demand_list[1]+[0], [0] + whole_demand_list[2]+[0]
    demand = np.array(cluster_demand_list)
    infection_demand = np.array(centroids_infection_demand)
    time_window, service_duration = [], []
    cluster_start_list = vrptw_input_dict['cluster_start_list']
    for index, start in enumerate(cluster_start_list):
        time_window.append([0, int(cluster_start_list[index]) * 100])
        service_duration.append(20)
    service_duration[0] = 0
    service_duration.append(0)
    time_window.append([0, 1440])
    time_window = np.array(time_window)
    service_duration = np.array(service_duration)
    vehicle_capacity = vrptw_input_dict['large_vehicle_capacity']
    vehicle_quantity = math.ceil(sum(cluster_demand_list)/vehicle_capacity)
    # calculate traveling distance and time from node to node
    distance_dict = vrptw_input_dict['cluster_dict_matrix']
    distance = np.zeros([len(coordinate), len(coordinate)])
    distance_dict[0, 0] = 0
    for i in range(len(coordinate)-1):
        distance_dict[len(coordinate)-1, i] = distance_dict[0, i]
        distance_dict[i, len(coordinate)-1] = distance_dict[i, 0]
    distance_dict[len(coordinate), len(coordinate)] = 0
    for i in range(len(coordinate)):
        for j in range(len(coordinate)):
            if i == j:
                distance[i, j] = 1e5
            else:
                distance[i, j] = distance_dict[i, j]
    distance_dict = vrptw_input_dict['cluster_dict_matrix']
    # calculate population  from node to node
    population_dict = vrptw_input_dict['population_matrix']
    population = np.zeros([len(coordinate), len(coordinate)])
    population_dict[0, 0] = 0
    for i in range(len(coordinate) - 1):
        population_dict[len(coordinate) - 1, i] = population_dict[0, i]
        population_dict[i, len(coordinate) - 1] = population_dict[i, 0]
    population_dict[len(coordinate), len(coordinate)] = 0
    for i in range(len(coordinate)):
        for j in range(len(coordinate)):
            if i == j:
                population[i, j] = 1e5
            else:
                population[i, j] = population_dict[i, j]
    return distance, population, coordinate, time_window, demand, infection_demand, service_duration, vehicle_quantity, vehicle_capacity

def load_dataset(xmlpath: str):
    """
    get processed (node 0 and node n + 1 added) matrices from xml datasets
    assume node 0 is the depot in the xml file
    return: coordinate, time_window, demand, service_duration, vehicle_quantity, vehicle_capacity
    """
    try:
        tree = ET.parse(xmlpath)
    except:
        print("Cannot find file")
        exit()
    root = tree.getroot()
    coordinate = np.zeros([1, 2])
    first_iter = True
    for node in root.iter("node"):
        if first_iter:
            first_iter = False
            coordinate = np.array([float(node.find("cx").text), float(node.find("cy").text)])
        else:
            coordinate = np.vstack((coordinate, np.array([float(node.find("cx").text), float(node.find("cy").text)])))
    coordinate = np.vstack((coordinate, coordinate[0, :]))
    time_window = np.zeros([1, 2])
    demand = np.array(0)
    service_duration = np.array(0)
    for request in root.iter("request"):
        time_window = np.vstack((
            time_window,
            np.array([float(request.find("tw").find("start").text), float(request.find("tw").find("end").text)])
        ))
        demand = np.append(demand, float(request.find("quantity").text))
        service_duration = np.append(service_duration, float(request.find("service_time").text))
    time_window = np.vstack((time_window, np.array([0, float(root.find("fleet").find("vehicle_profile").find("max_travel_time").text)])))
    demand = np.append(demand, 0)
    service_duration = np.append(service_duration, 0)
    vehicle_quantity = int(root.find("fleet").find("vehicle_profile").get("number"))
    vehicle_capacity = float(root.find("fleet").find("vehicle_profile").find("capacity").text)
    return coordinate, time_window, demand, service_duration, vehicle_quantity, vehicle_capacity

def save_raw_result(
    name: str,
    is_feasible: bool,
    objective_value: float,
    arc: np.ndarray,
    arrival_time: np.ndarray,
    coordinate: np.ndarray,
    time_window: np.ndarray,
    demand: np.ndarray,
    service_duration: np.ndarray,
    vehicle_quantity: int,
    vehicle_capacity: float,
    cost_per_distance: float,
    time_per_distance: float,
    solver_runtime: float,
    mip_gap: float):
    node_quantity = coordinate.shape[0]
    customer_quantity = node_quantity - 2
    N = range(node_quantity)
    C = range(1, customer_quantity + 1)
    V = range(vehicle_quantity)
    if 'random' in name:
        if (not os.path.exists("./result/random_result")):
            os.mkdir("./result/random_result")
        f = open("./result/random_result/raw-" + name + ".txt", "w")
    else:
        if (not os.path.exists("./result/case_result")):
            os.mkdir("./result/case_result")
        f = open("./result/case_result/raw-" + name + ".txt", "w")
    print(name, file=f)
    print(is_feasible, file=f)
    print(objective_value, file=f)
    print(mip_gap, file=f)
    print(node_quantity, file=f)
    print(vehicle_quantity, file=f)
    print(vehicle_capacity, file=f)
    print(cost_per_distance, file=f)
    print(time_per_distance, file=f)
    print(solver_runtime, file=f)
    for k in V:
        for i in N:
            for j in N:
                print(arc[k, i, j], file=f)
    for k in V:
        for i in N:
            print(arrival_time[i, k], file=f)
    for i in N:
        print(coordinate[i, 0], file=f)
        print(coordinate[i, 1], file=f)
    for i in N:
        print(time_window[i, 0], file=f)
        print(time_window[i, 1], file=f)
    for i in N:
        print(demand[i], file=f)
    for i in N:
        print(service_duration[i], file=f)
    f.close()
        

def load_raw_result(txtpath: str):
    try:
        f = open(txtpath)
    except:
        print("Cannot find file")
        exit()
    name = str(f.readline().strip("\n"))
    is_feasible = bool(f.readline().strip("\n") == "True")
    objective_value = float(f.readline())
    mip_gap = float(f.readline())
    node_quantity = int(f.readline())
    vehicle_quantity = int(f.readline())
    vehicle_capacity = float(f.readline())
    cost_per_distance = float(f.readline())
    time_per_distance = float(f.readline())
    solver_runtime = float(f.readline())
    customer_quantity = node_quantity - 2
    N = range(node_quantity)
    C = range(1, customer_quantity + 1)
    V = range(vehicle_quantity)

    arc = np.zeros([vehicle_quantity, node_quantity, node_quantity], dtype=int)
    for k in V:
        for i in N:
            for j in N:
                arc[k, i, j] = int(f.readline())

    arrival_time = np.zeros([node_quantity, vehicle_quantity])
    for k in V:
        for i in N:
            arrival_time[i, k] = float(f.readline())

    coordinate = np.zeros([node_quantity, 2])
    for i in N:
        coordinate[i, 0] = float(f.readline())
        coordinate[i, 1] = float(f.readline())

    time_window = np.zeros([node_quantity, 2])
    for i in N:
        time_window[i, 0] = float(f.readline())
        time_window[i, 1] = float(f.readline())

    demand = np.zeros(node_quantity)
    for i in N:
        demand[i] = f.readline()

    service_duration = np.zeros(node_quantity)
    for i in N:
        service_duration[i] = f.readline()

    f.close()

    return name, is_feasible, objective_value, arc, arrival_time, coordinate, time_window, demand, service_duration, vehicle_quantity, vehicle_capacity, cost_per_distance, time_per_distance, solver_runtime, mip_gap


def pretty_print(
    title: str,
    customer_quantity: int,
    is_feasible: bool,
    objective_value: float,
    vehicle_quantity: float,
    vehicle_capacity: float,
    cost_per_distance: float,
    time_per_distance: float,
    solver_runtime: float,
    chrono_info: list,
    result_load_quantity,
    result_infectious_risk,
    mip_gap: float,
):

    V = range(vehicle_quantity)


    f = open("./result/pretty-" + title + ".txt", "w")
    if 'random' in title:
        f = open("./result/random_result/pretty-" + title + ".txt", "w")
    else:
        f = open("./result/case_result/pretty-" + title + ".txt", "w")
    print(title, file=f)
    print("====================================================================================", file=f)
    print("customer quantity:", customer_quantity, file=f)
    print("vehicle quantity:", vehicle_quantity, file=f)
    print("vehicle capacity:", vehicle_capacity, file=f)
    print("cost per distance:", cost_per_distance, file=f)
    print("time per distance:", time_per_distance, file=f)
    print("====================================================================================", file=f)
    print("feasible:", is_feasible, file=f)
    print("solver runtime:", solver_runtime, file=f)
    if(is_feasible):
        print("objective function value:", objective_value, file=f)
        print("MIP gap:", mip_gap, file=f)
        print("====================================================================================", file=f)
        print("{:<13} {:<13} {:<13} {:<13} {:<13} {:<13}".format("|vehicle no.", "|time", "|node no.", "|cargo", "|X", "|Y"), file=f)
        print("------------------------------------------------------------------------------------", file=f)
        for k in V:
            if(chrono_info[k].shape[0] > 2):
                for i in range(chrono_info[k].shape[0]):
                    print(
                        "{:<13} {:<13} {:<13} {:<13} {:<13} {:<13}".format(
                            "|" + str(k),
                            "|" + str(round(chrono_info[k][i, 0], 3)),
                            "|" + str(int(chrono_info[k][i, 1])),
                            "|" + str(chrono_info[k][i, 2]),
                            "|" + str(chrono_info[k][i, 3]),
                            "|" + str(chrono_info[k][i, 4])
                        ),
                        file=f
                    )
                print("------------------------------------------------------------------------------------", file=f)
    
    f.close()


def load_pretty_result(txtpath: str):
    try:
        f = open(txtpath)
    except:
        print("Cannot find file")
        exit()

    feasible = False
    solver_runtime = 0
    obj = 0
    mip_gap = 0
    activated_vehicle = 0

    v_separation = "------------------------------------------------------------------------------------"
    while True:
        line = f.readline().strip("\n")
        if not line:
            break
        if(line[0:10] == "feasible: "):
            feasible = bool(line[10:] == "True")
        if(line[0:16] == "solver runtime: "):
            solver_runtime = float(line[16:])
        if(line[0:26] == "objective function value: "):
            obj = float(line[26:])
        if(line[0:9] == "MIP gap: "):
            mip_gap = float(line[9:])
        if(line == v_separation):
            activated_vehicle += 1
    
    activated_vehicle -= 1

    if(not feasible):
        mip_gap = math.nan
        obj = math.nan
        activated_vehicle = math.nan

    if(mip_gap < 1e-4 and solver_runtime < 3559):
        mip_gap = 0
    
    f.close()

    return activated_vehicle, obj, solver_runtime, mip_gap


def append_row_names(table: np.ndarray, setname: str):
    i = 0
    while(i < table.shape[0]):
        if i//3 < 9:
            table[i, 0] = setname.upper() + "0" + str(i//3 + 1) + "_025"
            table[i + 1, 0] = setname.upper() + "0" + str(i//3 + 1) + "_050"
            table[i + 2, 0] = setname.upper() + "0" + str(i//3 + 1) + "_100"
        else:
            table[i, 0] = setname.upper() + str(i//3 + 1) + "_025"
            table[i + 1, 0] = setname.upper() + str(i//3 + 1) + "_050"
            table[i + 2, 0] = setname.upper() + str(i//3 + 1) + "_100"
        i += 3


def relative_err(a, b):
    return (a - b) / a


def survey_results(table: np.ndarray, normalize: bool):

    nan_opt = 0
    nan_nan = 0
    inc_nan = 0
    inc_opt = 0
    popt_opt = 0
    opt_opt = 0

    total = 0

    for i in range(table.shape[0]):
        total += 1
        if(math.isnan(table[i, 0])):
            if(math.isnan(table[i, 1])):
                nan_nan += 1
            else:
                nan_opt += 1
        else:
            if(math.isnan(table[i, 1])):
                inc_nan += 1
            else:
                if(table[i, 4] == 0):
                    opt_opt += 1
                else:
                    if(table[i, 0] == table[i, 1] and relative_err(table[i, 2], table[i, 3]) < 0.02):
                        popt_opt += 1
                    else:
                        inc_opt += 1
    
    assert total == nan_opt + nan_nan + inc_nan + inc_opt + popt_opt + opt_opt

    if(normalize):
        return [nan_opt/total, nan_nan/total, inc_nan/total, inc_opt/total, popt_opt/total, opt_opt/total]
    else:
        return [nan_opt, nan_nan, inc_nan, inc_opt, popt_opt, opt_opt]



