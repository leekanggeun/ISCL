import argparse

def parse_args():
    desc = "Official Tensorflow 2.5 implementation of ISCL by Kanggeun Lee"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iter', type=int, default=400, help='The number of iters to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch per gpu')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--kfold', type=int, default=4, help='Select kfold cross-validation')
    parser.add_argument('--clean_data', type=str, help='Directory name to load the training clean data')
    parser.add_argument('--noisy_data', type=str, help='Directory name to load the training noisy data')
    parser.add_argument('--test_data', type=str, default=None, help='Directory name to load the test noisy data')
    parser.add_argument('--result_dir', type=str, help='Directory name to save the checkpoints')

    return check_args(parser.parse_args())

def check_args(args):
    # --result_dir
    try:
        assert args.epoch >= 1
    except:
        print('The number of epochs must be larger than or equal to one')

    # --batch_size
    assert args.batch_size >= 1, ('Batch size must be larger than or equal to one')
    try:
        os.mkdir(args.result_dir)
    except:
        print('Directory already exists or cannot make')

    return args

