# Project Overview

## I. Directory Structure
– data/<br>
├── case_data/ # Stores Case Study test files<br>
│ ├── CVRP_case_{node_num}.pkl # Input data file<br>
│ └── CVRPTW_{node_num}_milp.pkl # First-layer solution results<br>
├── case_data_generation/<br>
│ └── case_data_generation.py # Generate Case Study test data from fin_data.xlsx<br>
└── random_data/ # Random test datasets<br>

– result/ # Output results directory<br>
├── case_result/ # Stores Case Study results<br>
├── random_result/ # Stores random dataset results<br>
└── site_selection_model_result/ # Site selection model results<br>

– ACO_VRPTW_change.py # ACO solver (currently single-objective; can extend for multi-objective similar to ALNS or compute other metrics from results)<br>
– ALNS_VRPTW_BASE.py # ALNS solver code<br>
– case_study_medical_waste_routing.py # Site selection + truck delivery to sites<br>
– cvrp_gurobi_solve.py & core.py # Gurobi solver for depot -> site distribution routes<br>
– fileutil.py # Data reading and preprocessing for depot -> site distribution<br>
– location_routing_main.py # Main entry point (test script)<br>
– site_selection_model.py # Gurobi-based site selection model<br>
– tsp_exact.py # Gurobi solver for site -> disaster point routes<br>


---

## II. Quick Testing Instructions

### 1. Case Study Testing
- **Run Command**: Execute the main file `location_routing_main.py` with parameter:
  ```python
  case_type = 'case_study'
Batch Testing: Set<br>
test_num_list = [100, 200, ..., 1000] to specify node sizes.<br>
Data Preparation: Ensure input data exists; if not, generate via case_data_generation.py.<br>
Output File:<br>
result/case_study_result_{str(test_num_list[0])}_{str(test_num_list[-1])}.csv<br>
### 2. Random Dataset Testing
Run Command: Execute location_routing_main.py with parameter:<br>
case_type = 'random'<br>
Batch Testing: Set<br>
test_num_list = [100, 200, ..., 1400]<br>
Dataset Range: Covers node sizes from 100–1400.<br>
Output File:<br>
result/random_result/random_result_{str(test_num_list[0])}_{str(test_num_list[-1])}.csv
