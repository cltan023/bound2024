import torch

def cycle_loader(data_loader):
    while 1:
        for _, data in enumerate(data_loader):
            yield data

def get_grads(model): 
    # wrt data at the current step
    res = []
    for key, p in model.named_parameters():
        if 'embedding' in key:
            pass
        elif 'bias' in key:
            pass
        else:
            if p.requires_grad:
                res.append(p.grad.data.flatten())
    grad_flat = torch.cat(res)
    return grad_flat

def get_param(model): 
    # wrt data at the current step
    res = []
    for key, p in model.named_parameters():
        if 'embedding' in key:
            pass
        elif 'bias' in key:
            pass
        else:
            if p.requires_grad:
                res.append(p.data.flatten())
    param_flat = torch.cat(res)
    return param_flat, len(param_flat)

