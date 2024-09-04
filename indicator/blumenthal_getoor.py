# https://github.com/umutsimsekli/Hausdorff-Dimension-and-Generalization/blob/main/src/alpha.py

import torch
import math
import numpy as np

# Some divisibility BS
resolutions = { 3136: [756, [2,3,6,18,36]], 
                1574: [756, [2,3,6,18,36]], 
                1563: [756, [2,3,6,18,36]], 
                793: [756, [2,3,6,7,9,18,36]],
                782: [756, [2,3,6,7,9,18,36]], 
                402: [324, [2,3,6,9,18,36]],
                391: [324, [2,3,6,9,18,36]],
                207: [196, [2, 4, 7, 14, 28]],
                196: [196, [2, 4, 7, 14, 28]]
              }

# Corollary 2.4 in Mohammadi 2014
def alpha_estimator(m, X):
    # X is N by d matrix
    N = len(X)
    n = int(N/m) # must be an integer
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff

def get_5_good_fiv(x):
    rs = int(np.floor(np.sqrt(x)))
    div = []
    for i in range(2,rs):
        if x%i == 0:
            div.append(i)
    if len(div)>5:        
        return x, div[::int(np.ceil(len(div)/4))]+[div[-1]]
    else:
        return get_5_good_fiv(x-1)

def peek_model_size(model):
    model_sizes = []
    # Input is mode
    full_size = 0
    for p in model.parameters():
        if(len(p.shape)<2):
            full_size += p.shape[0]
            continue
        model_sizes.append(np.prod(p.shape))
        full_size += np.prod(p.shape)
    #model_sizes.append(full_size)
    return model_sizes

def get_ms(iter_size, mod_size):
    if iter_size in resolutions:
        iter_set =  resolutions[iter_size]
    else:
        raise ValueError("Not a valid iteration size")
    mod_set = get_5_good_fiv(mod_size)
    full_size = iter_set[0]*mod_set[0]
    all_ms = sorted([a*b for a in iter_set[1] for b in mod_set[1]])
    return full_size, all_ms

def estimator_vector_full(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]
    sz,ms = resolutions[iter_nums]
    
    iterate_matrix_zm = iterate_matrix - torch.mean(iterate_matrix, axis=0).view(1,-1)
   
    # print(sz,ms, iter_nums, dim)
    # print(iterate_matrix_zm[-1*sz:,:].shape)
    # print(len( iterate_matrix_zm[-1*sz:,:] ))
    est = [alpha_estimator(mm, iterate_matrix_zm[-1*sz:,:]).item() for mm in ms]

    return np.median(est)

def estimator_vector_projected(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]
    sz,ms = resolutions[iter_nums]
    
    iterate_matrix_zm = iterate_matrix - torch.mean(iterate_matrix, axis=0).view(1,-1)    

    proj_alpha = []
    for i in range(10):
        rand_direction = np.random.randn(dim,1)
        rand_direction = rand_direction / np.linalg.norm(rand_direction)
        rand_direction_t = torch.from_numpy(rand_direction).float()
        
        projected = torch.mm(iterate_matrix_zm,rand_direction_t)
        
        cur_alpha_est = [alpha_estimator(mm, projected[-1*sz:,:]).item() for mm in ms]
        
        proj_alpha.append(np.median(cur_alpha_est))
    return np.median(proj_alpha), np.max(proj_alpha)

def estimator_vector_mean(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]

    mean_over_iters = torch.mean(iterate_matrix, axis=0)
    mean_over_iters_zm = mean_over_iters - torch.mean(mean_over_iters)
    mean_over_iters_zm = mean_over_iters_zm.view(-1,1)
    sz, ms = get_5_good_fiv(dim)
    
    estimate_mean = [alpha_estimator(mm, mean_over_iters_zm[-1*sz:,:]).item() for mm in ms]
    return np.median(estimate_mean)

def estimator_scalar(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]

    sz, ms = get_ms(iter_nums, dim)
    iterate_matrix_vec = iterate_matrix.view(-1,1)
    iterate_matrix_vec_zm = iterate_matrix_vec - torch.mean(iterate_matrix_vec)
    estimate = [alpha_estimator(mm, iterate_matrix_vec_zm[-1*sz:,:]) for mm in ms[::4]]
   
    return np.median(estimate)


