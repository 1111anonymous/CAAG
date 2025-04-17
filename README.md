# CAAGNN

This repository is the implementation of CAAGNN.

CAAGNN is short for '**C**ontext-**A**daptive **A**ttention for **G**raph **N**eural **N**etworks-based Next POI Recommendation'. It is a novel framework, equipped with three modules (i.e., Graph-based User Preference Extractor with Context-Adaptive Attention, Sequential User Preference Extractor, Graph-Sequential Mutual Enhancement Module), which dynamically adjusts edge attention weights based on contextual signals to produce more informative POI representations and improve recommendation performance.


### File Descriptions

- `data.zip`: datasets for PHO, NY, and SIN;
- `main.py`: main file;
- `data.py`: data processing file;
- `parameter_setting.py`: parameter configuration script;
- `GAT.py` and `GATconv.py`: implementation of the CAAGNN model;
- `result_process.py`: aggregates multiple experimental results into CSV format.




### More Experimental Settings
- Environment
  - Our proposed CAAGNN is implemented using pytorch 
  


- Data Preprocessing
  - Following state-of-the-arts, for each user, we chronologically divide his check-in records into different trajectories by day, and then take the earlier 80% of his trajectories as training set; the latest 10% of trajectories as the test set; and the rest 10% as the validation set. Besides, we filter out POIs with fewer than 10 interactions, inactive users with fewer than 5 trajectories, and trajectories with fewer than 3 check-in records.


### How To Run
```
$ python main.py --city PHO
```
