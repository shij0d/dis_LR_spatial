#%%
import pickle
import torch
with open("expriements/decentralized/CI/results.pkl", "rb") as file:
    results=pickle.load(file)

# %%
# columns: gamma(across 5 columns), delta(1/tau^2), theta (across 2 columns)
# rows: empirical std, estimated std, empirical coverage probability
std_emp=torch.tensor([0.0200, 0.0205, 0.0222, 0.0194, 0.0222, 0.0032, 0.2428, 0.0113],dtype=torch.float64)
std_est_avg=torch.tensor([0.0201, 0.0201, 0.0201, 0.0201, 0.0201, 0.0036, 0.2447, 0.0113],dtype=torch.float64)
cv_prob=torch.tensor([0.9500, 0.9700, 0.9000, 0.9400, 0.9200, 0.9500, 0.9700, 0.9300],dtype=torch.float64)