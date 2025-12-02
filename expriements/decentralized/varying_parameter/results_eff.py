import sys
#sys.path.append('/home/shij0d/Documents/Dis_Spatial')
import pickle
from src.utils import exact_parameters

with open(f'expriements/decentralized/varying_parameter/mindis_0.01_irregular/nu_0.5_length_scale_0.1_weights_round_6.pkl', 'rb') as f:
        results=pickle.load(f)
        results_new=exact_parameters(results)
    
with open(f'expriements/decentralized/varying_parameter/mindis_0.01_irregular/nu_0.5_length_scale_0.1_weights_round_6_memeff.pkl', 'wb') as f:
        pickle.dump(results_new,f)