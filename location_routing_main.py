from casestudy_medical_waste_routing import *
import pickle as pkl
import os
from ACO_VRPTW_change import *
test_num_list = [100] # [100 * i for i in range(1,5)]
csv_result = []
location_mode_list = ['milp']
case_type = 'case_study' #'case_study' #'random'
if __name__ == "__main__":
    result = []
    title = ['node_index', 'type', 'route_index', 'x_coor', 'y_coor', 'current_tour_order', 'waste weight(kg)',
             'time_consumption']
    out_file = open(f"result/{case_type}_result_{str(test_num_list[0])}_{str(test_num_list[-1])}" + ".csv", 'w', encoding='utf-8', newline='' "")
    result_writer = csv.writer(out_file)
    result_writer.writerow(['No.', 'DataSet', 'Number of Hospitals', 'Location Algorithm', 'Number of TC', 'tc_score','Time','Number of vehicles',
                            'Truck_obj', 'Drone_obj', 'Gap Rate (CC-Gurobi)/Gurobi', 'Number of vehicles', 'gurobi_Obj1', 'gurobi_Obj2_risk', 'Time',
                            'Number of vehicles', 'alns_Obj1', 'alns_Obj2_risk', 'Time', 'Gap (alns-Gurobi)', 'Gap Rate (alns-Gurobi)/Gurobi'])
    for index, set_num in enumerate(test_num_list):
        for location_mode in location_mode_list:
            if 'random' in case_type:
                file_name = f'data/random_data/CVRPTW_{set_num}_{location_mode}.pkl'
            if 'case' in case_type:
                file_name = f'data/case_data/CVRPTW_{set_num}_{location_mode}.pkl'
            if os.path.isfile(file_name): # 有第一层聚类解结果
                print("start second layer routing")
            else:
                print("start first layer location")
            first_layer_test(case_type, set_num, location_mode)
            with open(file_name, 'rb') as file:
                vrptw_input_dict = pickle.load(file)
                if case_type == 'case_study':
                    data_set = f'case_{set_num}'
                else:
                    data_set = vrptw_input_dict['data_set']
                first_layer_result = vrptw_input_dict['first_layer_result']
                # TODO： 第二层ACO的测试，ACO目前obj是distance，obj risk可以仿照ALNS的代码加上，也可以根据结果把obj risk加上
                # aco_vrptw_tour, aco_vrptw_obj, aco_runtime = run_aco(vrptw_input_dict, Q=10, tau0=10, alpha=1, beta=5,rho=0.1, epochs=600, popsize=60)
                # 第二层gurobi测试
                vrptw_car_num, vrptw_obj, risk_obj, runtime = cvrptw_gurobi_solve_and_save(vrptw_input_dict, 3600)
                result.append([set_num, *first_layer_result, location_mode + "-milp", vrptw_car_num, vrptw_obj, round(runtime, 3)])
                # 第二层alns测试
                a_tour_dict, a_tour_time_dict, a_vrptw_obj, a_risk_obj, a_run_time = run_alns(vrptw_input_dict)
                result.append([set_num, *first_layer_result, location_mode + "-alns", len(a_tour_dict), a_vrptw_obj, round(a_run_time, 3)])
                print("---------------"+str(set_num)+location_mode+"----------------------")
                location_algorithm = 'Gurobi' if 'milp' in location_mode else "K-means Clustering"
                tc_num, tc_score, lc_obj, uav_obj, l_time = first_layer_result
                gap = a_vrptw_obj - vrptw_obj
                gap_percent = (gap/vrptw_obj)*100 if gap != 0 else 0
                try:
                    result_writer.writerow([index + 1, data_set, set_num, location_algorithm, tc_num, tc_score, l_time, tc_num, lc_obj, uav_obj, 0,
                                            vrptw_car_num, vrptw_obj, risk_obj, runtime, len(a_tour_dict), a_vrptw_obj, a_risk_obj, a_run_time, gap, gap_percent])
                except:
                    continue
                csv_result.append([index + 1, data_set, set_num, location_algorithm, tc_num,tc_score, l_time, tc_num, lc_obj,uav_obj, 0,
                     vrptw_car_num, vrptw_obj, risk_obj, runtime, len(a_tour_dict), a_vrptw_obj, a_risk_obj, a_run_time, gap, gap_percent])
                for i in csv_result:
                    print(i)
    out_file.close()
    # data_frame = pd.read_csv(f"result/{case_type}_result_1-25"+".csv")
    # data_frame.to_excel(f"result/{case_type}_result_1-25"+".xls", index=False)
    print("---------------all result----------------------")
    for i in result:
        print(i)

