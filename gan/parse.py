import glob
import argparse
import json
from os import path
import numpy as np

def parse(dirname, savepath, num_joints=15, conf_threshold=0.25, start=0, end=None):
    filenames = sorted(glob.glob(path.join(dirname, '*.json')))
    filenames = filenames[start:len(filenames) if end is None else end]
    poses = np.zeros([len(filenames), num_joints, 2])
    confs = np.zeros([len(filenames), num_joints])
    prev_pose = None

    for i, filename in enumerate(filenames):
        with open(filename) as json_file:
            people = json.load(json_file)['people']
        print(i, len(people))
        if len(people) == 0:
            pose = poses[i-1, :]
        else:
            points = np.array(people[0]['pose_keypoints_2d']).reshape(-1, 3)[:num_joints]
            pose = -points[:, :2]
            conf = points[:, -1]
            confs[i, :] = conf
            prev_pose = pose if prev_pose is None else prev_pose
            pose[conf < conf_threshold] = prev_pose[conf < conf_threshold]

        poses[i] = pose
        prev_pose = poses[i]
    np.savez(savepath, confidences=confs, poses=poses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num', type=int)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    vid_name = f'temp{args.num}.mp4'
    dirname = 'data/detections/' + vid_name
    savepath = 'data/parsed/' + vid_name[:-4]
    parse(dirname, savepath, start=args.start, end=args.end)

