import glob
import json
from os import path
import numpy as np

def parse(dirname, savepath, num_joints=15, conf_threshold=0.25):
    filenames = sorted(glob.glob(path.join(dirname, '*.json')))
    poses = np.zeros([len(filenames), num_joints, 2])
    confs = np.zeros([len(filenames), num_joints])
    prev_pose = None

    for i, filename in enumerate(filenames):
        with open(filename) as json_file:
            people = json.load(json_file)['people']
        if len(people) > 1:
            print(i, len(people))
        if len(people) == 0:
            print(i, len(people))
            pose = poses[i-1, :]
        else:
            points = np.array(people[0]['pose_keypoints_2d']).reshape(-1, 3)[:num_joints]
            pose = -points[:, :2]
            conf = points[:, -1]
            confs[i, :] = conf

        prev_pose = pose if prev_pose is None else prev_pose
        poses[i] = pose
        poses[i, conf < conf_threshold] = prev_pose[conf < conf_threshold]
        prev_pose = poses[i]
    np.savez(savepath, confidences=confs, poses=poses)

if __name__ == '__main__':
    dirname = ''
    savepath = ''
    parse(dirname, savepath)

