from __future__ import print_function
import torch
import torch.nn as nn
import Net


if __name__ == "__main__":
    d1 = Net.d
    d2 = Net.d
    print(d1 == d2)
    print(d1 is d2)
