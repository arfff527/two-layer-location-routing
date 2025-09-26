# Project Overview

## I. Directory Structure
– data/<br>
├── case_data/ # Stores Case Study test files<br>
│ ├── CVRP_case_{node_num}.pkl # Input data file
│ └── CVRPTW_{node_num}_milp.pkl # First-layer solution results
├── case_data_generation/
│ └── case_data_generation.py # Generate Case Study test data from fin_data.xlsx
└── random_data/ # Random test datasets

– result/ # Output results directory
├── case_result/ # Stores Case Study results
├── random_result/ # Stores random dataset results
└── site_selection_model_result/ # Site selection model results

– ACO_VRPTW_change.py # ACO solver (currently single-objective; can extend for multi-objective similar to ALNS or compute other metrics from results)
– ALNS_VRPTW_BASE.py # ALNS solver code
– case_study_medical_waste_routing.py # Site selection + UAV and truck delivery to disaster sites
– cvrp_gurobi_solve.py & core.py # Gurobi solver for depot -> site distribution routes
– fileutil.py # Data reading and preprocessing for depot -> site distribution
– location_routing_main.py # Main entry point (test script)
– site_selection_model.py # Gurobi-based site selection model
– tsp_exact.py # Gurobi solver for site -> disaster point routes


---

## II. Testing Instructions

### 1. Case Study Testing
- **Run Command**: Execute the main file `location_routing_main.py` with parameter:
  ```python
  case_type = 'case_study'
Batch Testing: Set
test_num_list = [100, 200, ..., 1000]
to specify node sizes.
Data Preparation: Ensure input data exists; if not, generate via case_data_generation.py.
Output File:
result/case_study_result_{str(test_num_list[0])}_{str(test_num_list[-1])}.csv
2. Random Dataset Testing
Run Command: Execute location_routing_main.py with parameter:
case_type = 'random'
Batch Testing: Set
test_num_list = [100, 200, ..., 1400]
Dataset Range: Covers node sizes from 100–1400.
Output File:
result/random_result_{str(test_num_list[0])}_{str(
