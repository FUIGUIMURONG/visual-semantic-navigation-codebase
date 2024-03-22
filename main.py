from __future__ import print_function, division

import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from utils.flag_parser import parse_arguments
from RL import train_A3C, test_A3C
from IL import train_IL, test_IL


def main():
    args = parse_arguments()
    if args.algorithm == "IL":
        if args.train_or_test == "train":
            train_IL(args)
        else:
            test_IL(args, args.train_or_test, args.test_setting)
    elif args.algorithm == "RL":
        if args.train_or_test == "train":
            train_A3C(args)
        else:
            test_A3C(args, args.train_or_test, args.test_setting)


if __name__ == "__main__":
    main()
