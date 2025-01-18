from joblib import Parallel, delayed
import math
results=Parallel(n_jobs=2)(delayed(math.sqrt)((i+j*0.1) ** 2) for i in range(10) for j in range(10))