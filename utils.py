import torch
import numpy as np
import nolds
import random

class DictToClass:
  def __init__(self, dictionary):
    for key, value in dictionary.items():
      setattr(self, key, value)

def cycle_loader(data_loader):
    while 1:
        for _, data in enumerate(data_loader):
            yield data

def get_grads(net): 
    # wrt data at the current step
    grads = []
    for name, p in net.named_parameters():
        if 'bn' in name:
            continue
        if p.requires_grad:
            if p.grad is None: continue
            grads.append(p.grad.data.flatten().cpu())
    grads = torch.cat(grads)
    return grads

def get_params(net): 
    # wrt data at the current step
    params = []
    for name, p in net.named_parameters():
        if 'bn' in name:
            continue
        if p.requires_grad:
            params.append(p.data.flatten().cpu())
    params = torch.cat(params)
    return params
      
def fractal_dimension(coordinate_estimates):
    coordinate_estimates = np.sort(coordinate_estimates)

    cum_sum_values = np.cumsum(coordinate_estimates)
    stop = np.where(cum_sum_values - 1.0 > 0)[0][0]

    val = 1.0
    for i in range(stop):
        val += coordinate_estimates[stop] - coordinate_estimates[i]
    val /= coordinate_estimates[stop]
    
    return val

def calc_hurst_exponent(sequence):
    try:
        nvals = nolds.logarithmic_r(min_n=50, max_n=500, factor=1.2)
        nvals = [int(n) for n in nvals]
        h = nolds.hurst_rs(sequence, corrected=False, nvals=nvals)
    except:
        h = -1
    return h

def error_func(e):
    print('error callback in pool', e)
    
def init_random_state(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
def validate(net, test_loader, test_loss_func, device, train=False):
    if train:
        # eval mode at early stage of training may result in very large loss even NaNs
        init_random_state(42)
        net.train()
    else:
        net.eval()
    tot_loss = 0.0
    tot_correct = 0.0
    loss_vec = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = test_loss_func(output, target)
            loss_vec.append(loss.detach().cpu())
            tot_loss += loss.sum().item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            tot_correct += pred.eq(target.view_as(pred)).sum().item()
    loss_vec = torch.hstack(loss_vec)
    return tot_loss / len(test_loader.dataset), tot_correct / len(test_loader.dataset), loss_vec