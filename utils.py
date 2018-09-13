import torch



def get_mean_std(input, eps=1e-5):
    assert(len(input.size()) == 4)
    n, c, h, w = input.size()
    input = input.view(n, c, -1)
    mean = input.mean(dim=2).view(n, c, 1, 1)
    var = input.var(dim=2).view(n, c, 1, 1) + eps
    std = var.sqrt()
    return mean, std



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
