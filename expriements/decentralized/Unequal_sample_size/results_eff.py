#%%
import sys
#sys.path.append('/home/shij0d/documents/dis_LR_spatial')
import pickle
from src.utils import exact_parameters



#%%

with open(f'/home/shij0d/documents/dis_LR_spatial/expriements/decentralized/Unequal_sample_size/res.pkl','rb') as f:
    results=pickle.load(f)
    results_new=exact_parameters(results)

with open(f'/home/shij0d/documents/dis_LR_spatial/expriements/decentralized/Unequal_sample_size/res_memeff.pkl', 'wb') as f:
        pickle.dump(results_new, f)
        
        

    