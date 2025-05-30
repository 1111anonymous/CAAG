# CAGNN

This repository is the implementation of CAGNN.

CAGNN is short for '**C**ontext-**A**daptive **G**raph **N**eural **N**etworks for Next POI Recommendation'. It is a novel framework, equipped with three modules (i.e., Graph-based User Preference Extractor with Context-Adaptive Attention, Sequential User Preference Extractor, and Graph-Sequential Mutual Enhancement Module), which jointly models the co-influence of multiple contextual factors to dynamically adjust attention weights, thereby producing more informative POI representations and improving recommendation performance.


### File Descriptions

- `data.zip`: datasets for PHO, NY, and SIN;
- `main.py`: main file;
- `data.py`: data processing file;
- `parameter_setting.py`: parameter configuration script;
- `GAT.py` and `GATconv.py`: implementation of the CAGNN model;
- `result_process.py`: aggregates multiple experimental results into CSV format.




### More Experimental Settings
- Environment
  - All experiments are conducted on Google Colab with an NVIDIA A100 GPU, using PyTorch as the implementation framework. Besides the default packages provided by Colab, we additionally install the Deep Graph Library (DGL) to support graph-based computations.
  


- Data Preprocessing
  - Following state-of-the-arts, for each user, we chronologically divide their check-in records into different trajectories by day, and then take the earlier 80% of their trajectories as the training set; the latest 10% of trajectories as the test set; and the rest 10% as the validation set. Besides, we filter out POIs with fewer than 10 interactions, inactive users with fewer than 5 trajectories, and trajectories with fewer than 3 check-in records.


### How To Run
```
$ python main.py --city PHO
```
