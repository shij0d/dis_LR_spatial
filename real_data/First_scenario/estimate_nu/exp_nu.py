#path_project='/home/shij0d/Documents/Dis_Spatial'
import warnings
warnings.simplefilter("error", FutureWarning)
import numpy as np
from netCDF4 import Dataset
import os
import numpy as np
#import sys
# Add the path where your Python packages are located
#sys.path.append(path_project)
import torch
from scipy.optimize import minimize
from src.estimation_torch_real_data import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.kernel import exponential_kernel,onedif_kernel,matern_kernel
from src.networks import generate_connected_erdos_renyi_graph
from src.weights import optimal_weight_matrix
import networkx as nx
import numpy as np
import random
import pickle
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from golden_section_search_optimization import golden_section_search_optimization
import time


#cpu number
cpu_number=multiprocessing.cpu_count()//6
print(f"cpu number: {cpu_number}")
os.environ["OMP_NUM_THREADS"] = str(cpu_number)  # For OpenMP (e.g., NumPy, scikit-learn)
os.environ["MKL_NUM_THREADS"] = str(cpu_number)  # If you're using MKL-based libraries
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_number)  # For OpenBLAS (NumPy, SciPy)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_number)  # For NumExpr (if used)


SEED=2024

path = 'real_data/First_scenario/data/Blended-Hydro_TPW_MAP_d20241105/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
file_path=path+files[0]
file2read = Dataset(file_path)


lats=file2read.variables['lat'][:]
lons=file2read.variables['lon'][:]
TPWs=file2read.variables['TPW'][:]
observation_ages=file2read.variables['observation_age'][:]
satellite_numbers=file2read.variables['Satellite_Number'][:]
quality_informations=file2read.variables['quality_information'][:]
quality_flags=file2read.variables['quality_flag'][:]
lon_grid, lat_grid = np.meshgrid(lons.data, lats.data)



#select the data 
lat_min, lat_max = 15, 75  # Latitude range for North America
lon_min, lon_max = -170, -50  # Longitude range for North America


lat_min, lat_max = 15, 55  
lon_min, lon_max = -160, -120  

#United states

lat_min, lat_max = 24.396308, 49.384358
lon_min, lon_max = -125.0,-66.93457
lon_grid, lat_grid = np.meshgrid(lons.data, lats.data)

region_mask = (
        (lat_grid >= lat_min) & (lat_grid <= lat_max) &
        (lon_grid >= lon_min) & (lon_grid <= lon_max)
    )
lats_NorthAmerican = lat_grid[region_mask]
lons_NorthAmerican = lon_grid[region_mask]
TPWs_NorthAmerican = TPWs[region_mask]
satellite_numbers_NorthAmerican =satellite_numbers[region_mask]

unique_satellites, counts = np.unique(satellite_numbers_NorthAmerican, return_counts=True)

# Display results
for satellite, count in zip(unique_satellites, counts):
    print(f"Satellite Number {satellite}: {count} observations")
unique_satellites=unique_satellites.compressed()
dis_data=[]
#divide the data according to sat
for sat in unique_satellites:
    mask = satellite_numbers_NorthAmerican == sat
    # if sat==501.0:
    #     continue
    lats_sat = lats_NorthAmerican[mask]
    lons_sat = lons_NorthAmerican[mask]
    TPWs_sat = TPWs_NorthAmerican[mask]
    valid_indices = ~TPWs_sat.mask
    # Filter the arrays
    lats_sat = lats_sat[valid_indices]
    lons_sat = lons_sat[valid_indices]
    TPWs_sat = TPWs_sat[valid_indices]
    X=np.ones((lats_sat.shape[0],1))
    local_data=np.hstack((lats_sat.reshape(-1,1),lons_sat.reshape(-1,1),TPWs_sat.reshape(-1,1),X))
    
    #withoutX
    local_data=np.hstack((lats_sat.reshape(-1,1),lons_sat.reshape(-1,1),TPWs_sat.reshape(-1,1)))
    
    np.random.seed(SEED)
    # local_shuffled_data = np.random.permutation(local_data)
    # local_data=local_shuffled_data[:20,:]
    
    #sample 10% data
    local_data=local_data[np.random.rand(local_data.shape[0])<0.1]
    if local_data.shape[0]==0:
        continue
    dis_data.append(local_data)
print(f"dis_data shape:{len(dis_data)}")
#randomly 
full_data=np.vstack(dis_data)#.data
#full_data=full_data[:12000,:]
#remove the mean
beta=np.mean(full_data[:,2])
full_data[:, 2] -= beta
N=full_data.shape[0]
print(f"full_data shape:{full_data.shape}")
locations=full_data[:,:2]
locations = [tuple(row) for row in locations]

for i,local_data in enumerate(dis_data):
    local_data[:,2]-=beta
    dis_data[i]=local_data


# np.random.seed(42)  # Set seed for reproducibility
# shuffled_data = np.random.permutation(full_data)  # Shuffle rows of the array
# num_parts = 5
# dis_data = np.array_split(shuffled_data, num_parts)

J=len(dis_data)
con_pro=0.5
er = generate_connected_erdos_renyi_graph(J, con_pro)
adj_matrix=nx.adjacency_matrix(er).todense()
np.fill_diagonal(adj_matrix, 1) 
weights,_=optimal_weight_matrix(adj_matrix=adj_matrix)
weights=torch.tensor(weights,dtype=torch.double)
#weights = torch.ones((J,J),dtype=torch.float64)/J

#grid knots
# lats=np.arange(lat_min, lat_max,min_dis)
# lons=np.arange(lon_min, lon_max,min_dis)
# knots = [(lat, lon) for lat in lats for lon in lons]
# random knots
random.seed(SEED)
m=100
knots = random.sample(locations, m)

#estimation

def MLE_f(nu):
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, partial(exponential_kernel,type="chordal"), knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, partial(onedif_kernel,type="chordal"), knots, weights)
    else:
        #matern_kernel_nu=matern_kernel_factory(nu)
        gpp_estimation = GPPEstimation(dis_data, partial(matern_kernel,nu=nu,type="chordal"), knots, weights)
    
    
    beta_ini=None
    delta_ini=torch.tensor(1,dtype=torch.float64)
    alpha_ini=10
    length_scale_ini=0.5
    theta_ini=torch.tensor([alpha_ini,length_scale_ini],dtype=torch.float64)
    x_ini=gpp_estimation.argument2vector_lik(beta_ini,delta_ini,theta_ini)
    
    
    _,_,_,delta,theta,result=gpp_estimation.get_minimier(x_ini)
    optimal_estimator=(beta,delta,theta,result)
    file_path_save=f"real_data/First_scenario/estimate_nu/output/02_2/optimal_estimator_nu_{nu:.2f}.pkl"
    with open(file_path_save, "wb") as file:
        pickle.dump(optimal_estimator, file)
    #print("global optimization succeed")
    print(f"nu:{nu},delta:{delta.numpy()},theta:{theta.squeeze().numpy()},MLE_f:{result.fun}")
    return result.fun



  
def De_f(nu):
    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, partial(exponential_kernel,type="chordal"), knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, partial(onedif_kernel,type="chordal"), knots, weights)
    else:
        #matern_kernel_nu=matern_kernel_factory(nu)
        gpp_estimation = GPPEstimation(dis_data, partial(matern_kernel,nu=nu,type="chordal"), knots, weights)
    beta_ini=None
    delta_ini=torch.tensor(1,dtype=torch.float64)
    alpha_ini=10
    length_scale_ini=0.5
    theta_ini=torch.tensor([alpha_ini,length_scale_ini],dtype=torch.float64)
    x_ini=gpp_estimation.argument2vector_lik(beta_ini,delta_ini,theta_ini)

    time_start=time.time()
    mu_list,Sigma_list,beta_list,delta_list,theta_list,_,_=gpp_estimation.get_local_minimizers_parallel(x_ini,job_num=-1)
    time_end=time.time()
    print(f"time to get local minimizers: {time_end-time_start}")
    
    mu=mu_list[0]
    Sigma=Sigma_list[0]
    #beta=beta_list[0]
    delta=delta_list[0]
    theta=theta_list[0]
    num=len(mu_list)
    if num>1:
        for j in range(1,num):
            mu+=mu_list[j]
            Sigma+=Sigma_list[j]
            #beta+=beta_list[j]
            delta+=delta_list[j]
            theta+=theta_list[j]
    else:
        raise ValueError("local optimization failed")
    mu=mu/num
    Sigma=Sigma/num
    beta=None
    delta=delta/num
    theta=theta/num
    print(f"initial: delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
    
    mu_list=[]
    Sigma_list=[]
    beta_list=[]
    delta_list=[]
    theta_list=[]
    for j in range(J):
        mu_list.append(mu)
        Sigma_list.append(Sigma)
        beta_list.append(beta)
        delta_list.append(delta)
        theta_list.append(theta)
    
    
    T=20
   
    de_estimators=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T=T,weights_round=6)
    de_estimators_simple=(nu,de_estimators[3],de_estimators[4],de_estimators[6])
    fun=(de_estimators[6]-m)/N
    theta=de_estimators[4][-1][0].numpy()
    print(f"nu:{nu},theta:{theta},De_f:{fun}")
    file_path_save=f"real_data/First_scenario/estimate_nu/output/02_2/de_estimator_simple_nu_{nu:.2f}.pkl"
    with open(file_path_save, "wb") as file:
        pickle.dump(de_estimators_simple, file)
    return fun





L=0.2
R=2
epsilon=0.05
MLE_nu=golden_section_search_optimization(MLE_f,L,R,epsilon)
De_nu=golden_section_search_optimization(De_f,L,R,epsilon)
print(f"MLE_nu:{MLE_nu},De_nu:{De_nu}")
