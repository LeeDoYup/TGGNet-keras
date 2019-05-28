import numpy as np
import random
import tensorflow as tf

from models.TGNet_NYC import TGNet

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if __name__ == "__main__":
    print('[*] Program Starts')
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--drop_p', type=float, default=0.1)
    parser.add_argument('--reg', type=float, default=0.0)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--save_dir', type=str, default='./model_saved/')
    parser.add_argument('--model_name', type=str, default='no_named')

    parser.add_argument('--scale', type=str, default='min_max')
    parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--thr', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--num_gpu', type=int, default=2)
    parser.add_argument('--coord', type=float, default=25.0)
    parser.add_argument('--coord_net', type=int, default=2)

    parser.add_argument('--temp', type=int, default=16)
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--enf', type=int, default=64)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--es', type=str, default='min')

    args = parser.parse_args()

    if args.dataset_name == 'NYC':
        input_shape = [10, 20, 8]
        args.dataset = './datasets/NYC_taxi/'
    elif args.dataset_name == 'NYCB':
        input_shape = [10, 20, 8]
        args.dataset = './datasets/NYC_bike/'
    else:
      raise IOError(repr("Set args.dataset"))

    if args.model_name == 'no_named':
        raise IOError(repr("NO MODEL NAME IN args: python main.py ... --model_name MODEL_NAME"))

    print('[!] Model Creation Start')
    model = TGNet(input_shape, args)
    print('[*] Model Creation End')

    if not args.test:
        print('\n [!] Train Start')
        model.train()
    else:
        print('\n [!] Test Start')
        model.test()
        print('\n [*] Test End')

    print('[!] PROGRAM ENDS')
