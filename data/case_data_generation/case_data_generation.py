import random
import math
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import pickle as pkl


# ---------------------- 数据读取与处理 ---------------------- #
def read_and_format_data(input_path):
    case_df = pd.read_excel(input_path, sheet_name="Sheet1")
    location_index = case_df['Index'].tolist()
    location_list = list(zip(case_df['longitude'], case_df['latitude']))
    time_window_list = list(zip(case_df['startTime'], case_df['endTime']))
    demand_list = case_df['demand'].tolist()
    demand_list = [int(d * 5) for d in demand_list]
    building_damage_list = case_df['Building_Damage_degree'].tolist()
    road_damage_list = case_df['Road_Damage_degree'].tolist()
    popul_number_list = case_df['popul_number'].tolist()
    victim_number_list = case_df['victim_number'].tolist()
    priority_list = case_df['Prority'].tolist()
    formatted_data = []
    for i, (loc, index, demand, time_win, bd, rd, popu, victim, prio) in enumerate(
        zip(location_list, location_index, demand_list, time_window_list,
            building_damage_list, road_damage_list, popul_number_list,
            victim_number_list, priority_list)
    ):
        formatted_data.append({
            'CUST NO': i + 1,
            'Index': index,
            'XCOORD': loc[0],         # longitude = X
            'YCOORD': loc[1],         # latitude  = Y
            'DEMAND': demand,
            'READY TIME': time_win[0],
            'DUE TIME': time_win[1],
            'SERVICE TIME': 10,
            'TYPE': 0,                # 默认类型为0，可根据需要调整
            'Building_Damage': bd,
            'Road_Damage': rd,
            'Population': popu,
            'Victim': victim,
            'Priority': prio
        })

    return pd.DataFrame(formatted_data)


def get_distance(coord1, coord2):
    try:
        return geodesic(coord1, coord2).kilometers
    except Exception:
        print(f"Error with coords: {coord1}, {coord2}")
        return 0

def normalize(values):
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return [0 for _ in values]
    return [round((v - min_val) / (max_val - min_val), 3) for v in values]

# ---------------------- 生成 CVRP 数据 ---------------------- #
def generate_cvrp_data(capacity, input_path, node_num):
    df = read_and_format_data(input_path)
    points = np.column_stack((df['XCOORD'].values, df['YCOORD'].values))
    dist_matrix = np.zeros((node_num, node_num))
    population_matrix = np.zeros((node_num, node_num))
    score_list = df['Priority'].tolist()
    population_list = df['Population'].tolist()

    for i in range(node_num):
        for j in range(i, node_num):
            if i == j:
                continue
            coord1 = (points[i][1], points[i][0])
            coord2 = (points[j][1], points[j][0])
            dist = get_distance(coord1, coord2) * 1000 # 转换成米
            dist = round(dist, 3)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    for i in range(node_num):
        for j in range(i, node_num):
            if i == j:
                continue
            val = population_list[i] + population_list[j]
            population_matrix[i, j] = val
            population_matrix[j, i] = val

    cvrp_data = {
        'NAME': f'CVRP_{node_num}',
        'COMMENT': f'CVRP_{node_num}',
        'TYPE': 'CVRP',
        'DIMENSION': node_num,
        'NODE_COORD_TYPE': 'TWOD_COORDS',
        'CAPACITY': 1000,
        'EDGE_WEIGHT': dist_matrix,
        'NODE_COORD': points[:node_num+1],
        'population_matrix': population_matrix,
        'DEMAND': df['DEMAND'].values[:node_num],
        'DEPOT': np.array([0]),
        'NODE_TYPE': df['TYPE'].values[:node_num],
        'safe_score': np.array(score_list[:node_num+1])
    }

    with open(f'../case_data/CVRP_case_{node_num}.pkl', 'wb') as f:
        pkl.dump(cvrp_data, f)
    return cvrp_data


# ---------------------- 主程序入口 ---------------------- #
if __name__ == "__main__":
    # Step 1: 格式化医院数据
    input_csv = "fin_data.xlsx"
    # Step 2: 生成CVRP数据
    node_num = 100
    capacity = 1000
    data = generate_cvrp_data(capacity, input_csv, node_num)
    print("✅ 数据处理和CVRP实例生成完成。")
