import json
import os
import numpy as np
import pandas as pd

exp_root_path = "exp_results_downstream"
method_list = os.listdir(exp_root_path)
total_fold_num = 5
total_fold = f'total_{total_fold_num}_fold'

df=pd.DataFrame()
target_metrics = 'balanced_accuracy'
for method in method_list:
    if '.csv' in method:
        continue
    method_path = os.path.join(exp_root_path,method)
    for ckpt in sorted(os.listdir(method_path)):
        ckpt_path = os.path.join(method_path,ckpt)
        for dataset in sorted(os.listdir(ckpt_path)):
            dataset_path = os.path.join(ckpt_path,dataset)
            exp_list = os.listdir(dataset_path)
            lr_set = set((i.split('backbone')[-1].split('_')[0]) for i in exp_list)
            lr_exp_result_dict={i:[] for i in lr_set}
            for exp in exp_list:
                lr=exp.split('backbone')[-1].split('_')[0]
                exp_path = os.path.join(dataset_path,exp,total_fold)
                fold_metrics_list=[]
                for i in range(total_fold_num):
                    metric_json_path = os.path.join(exp_path,f'{i}_fold','metrics.json')
                    if not os.path.exists(metric_json_path):
                        continue
                    with open(metric_json_path) as f:
                        fold_metrics_list.append(json.load(f))
                if len(fold_metrics_list)==0:
                    continue
                keys = fold_metrics_list[0].keys()
                with open(os.path.join(exp_path, "metrics.txt"), "w") as f:
                    for key in keys:
                        vals = [m[key] for m in fold_metrics_list]
                        mean = np.mean(vals)
                        std = np.std(vals)
                        f.write(f"{key}: {mean:.4f} /pm {std:.4f}\n")
                lr_exp_result_dict[lr]+=fold_metrics_list
            
            lr_dict={}
            best_bacc=0.0
            best_lr=None
            for lr in lr_exp_result_dict.keys():
                lr_dict[lr]={}
                if len(lr_exp_result_dict[lr])==0:
                    continue
                key_list=lr_exp_result_dict[lr][0].keys()
                for key in key_list:
                    value_list = [i[key] for i in lr_exp_result_dict[lr]]
                    mean_value = np.mean(value_list)
                    std_value = np.std(value_list)
                    lr_dict[lr][key]=(mean_value,std_value)
                if lr_dict[lr]['balanced_accuracy'][0]>best_bacc:
                    best_bacc=lr_dict[lr]['balanced_accuracy'][0]
                    best_lr=lr
            if best_lr==None:
                continue
            column_name = f"{method}_{ckpt}"
            if 'f1' not in lr_dict[best_lr].keys():
                lr_dict[best_lr]['f1']=lr_dict[best_lr]['f1_weighted']
            best_mean,best_std = lr_dict[best_lr][target_metrics]
            if dataset not in df.index:
                df.loc[dataset, column_name] = f"{best_mean:.4f}±{best_std:.4f}"
            else:
                df.at[dataset, column_name] = f"{best_mean:.4f}±{best_std:.4f}"

output_csv_path = os.path.join(exp_root_path, f"results.csv")
df.to_csv(output_csv_path)
print(f"Results saved to {output_csv_path}")