import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/titanic/')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=30)
    args = parser.parse_args()

    return args
