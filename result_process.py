import pandas as pd
import glob
import os

def cal_ave_result(file_name, save_name):
    columns = ["HR @1", "HR @5", "HR @10", "NDCG@1", "NDCG@5", "NDCG@10", "MAP@1", "MAP@5", "MAP@10"]
    results_df = pd.DataFrame(columns=columns)
    result_files = [f for f in glob.glob(file_name) if '_attention' not in f]

    for file in result_files:
        with open(file, 'r') as f:
            data = {}
            for line in f:
                line = line.strip('\n')
                if line:
                    metrics = line.split(',\t')
                    for metric in metrics:
                        key, value = metric.split(" : ")
                        key = key.strip()
                        value = float(value.strip())
                        data[key] = value
            # results_df = results_df.append(data, ignore_index=True)
            aligned_data = {col: data.get(col, None) for col in columns}
            new_row = pd.DataFrame([aligned_data])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Calculate mean values and add as a new row
    mean_values = results_df.mean().to_dict()
    mean_row = pd.DataFrame([mean_values])
    results_df = pd.concat([results_df, mean_row], ignore_index=True)

    # 计算平均值并添加为新行
    # mean_values = results_df.mean()
    # results_df = results_df.append(mean_values, ignore_index=True)

    # 保存到CSV文件
    # save_name = file_name.split("*")[0] + "final_results.csv"
    results_df.to_csv(save_name, index=False)

    # 删除原始 txt 文件
    for file in result_files:
        os.remove(file)