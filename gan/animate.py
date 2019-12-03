from utils import animate
import argparse
import numpy as np

def main(args):
    poses = np.load(args.data_path, allow_pickle=True)['poses']
    if args.length is not None:
        poses = poses[:args.length]
    if args.save_path is None:
        args.save_path = args.data_path[:-4] + '.mp4'
    animate(poses, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--length', type=int)
    main(parser.parse_args())

