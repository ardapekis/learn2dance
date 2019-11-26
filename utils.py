import torch
import glob
from celluloid import Camera
import json
from torch import nn, autograd
from os import path
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import animation

BONE_LIST = [
    [0, 1],
    [1, 2], [2, 3], [3, 4],
    [1, 5], [5, 6], [6, 7],
    [1, 8],
    [8, 9], [9, 10], [10, 11],
    [8, 12], [12, 13], [13, 14]
]

def pose_plot(pose, show=True, pause=None):
    for i, j in BONE_LIST:
        plt.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], color='b')
    # plt.scatter(pose[:, 0], pose[:, 1], color='blue')
    plt.gca().set_aspect('equal', adjustable='box')
    for i, coordinate in enumerate(pose):
        plt.annotate(i, coordinate, fontsize=10)
    if show:
        if pause is None:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(pause)
            plt.close()

def translate(poses, idx):
    return poses - poses[:, idx:idx + 1]


def gen_seq(num_timesteps, num_cycles, shift, sin=True):
    t = np.arange(num_timesteps) / num_timesteps * num_cycles * 2 * np.pi
    t += shift * num_cycles
    return np.sin(t) if sin else np.cos(t)

def get_fake_poses(num_timesteps=50, batch_size=100, num_cycles=2, eps=0.03):
    classes = np.zeros(batch_size)
    poses = np.zeros([batch_size, num_timesteps, 2])
    for i in range(batch_size):
        shift = np.random.rand() * 2 * np.pi
        sin = gen_seq(num_timesteps, num_cycles, shift, True)
        cos = gen_seq(num_timesteps, num_cycles / (i % 2 + 1), shift, False)
        poses[i] = np.stack([sin, cos], axis=1)
        classes[i] = i % 2
    poses += np.random.randn(*poses.shape) * eps
    plt.scatter(poses[1, :, 0], poses[1, :, 1])
    plt.show()

    return poses, classes

def get_fake_x(num_timesteps=50, batch_size=100, eps=0.03):
    classes = np.zeros(batch_size)
    poses = np.zeros([batch_size, num_timesteps, 2])
    for i in range(batch_size):
        t = np.random.rand(num_timesteps) - 0.5
        # poses[i] = np.stack([t, t if i % 2 == 0 else -t], axis=1)
        poses[i] = np.stack([t, t], axis=1)
        classes[i] = i % 2
    poses += np.random.randn(*poses.shape) * eps
    plt.scatter(poses[0, :, 0], poses[0, :, 1])
    plt.show()
    return poses, classes

class Poses(Dataset):
    def __init__(self, poses, labels):
        self.poses = poses.reshape(-1, poses.shape[-1])
        self.labels = labels.repeat(poses.shape[1])

    def __getitem__(self, idx):
        return self.poses[idx].astype(np.float32), int(self.labels[idx])

    def __len__(self):
        return len(self.labels)

class SeqPoses(Dataset):
    def __init__(self, poses, labels, length=50):
        self.poses = poses.reshape(-1, poses.shape[-1])
        self.labels = labels.repeat(poses.shape[1])
        if len(self.poses) <= length:
            raise ValueError()
        self.length = length

    def __getitem__(self, idx):
        return self.poses[idx:idx+self.length].astype(np.float32), \
               int(self.labels[idx])

    def __len__(self):
        return len(self.labels) - self.length


def block(in_feat, out_feat, normalize=True, leaky=True, dropout=False):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.02, inplace=True) if leaky else nn.ReLU(True))
    if dropout:
        layers.append(nn.Dropout(0.5))
    return layers



def gradient_penalty(dsc, real, fake, fake_classes, device):
    batch_size = real.size(0)
    alpha = torch.rand((batch_size, 1))

    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    labels = torch.full((batch_size,), 0, device=device)
    validity = dsc(interpolates, fake_classes)  # todo: add classes

    gradients = autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=labels,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

class OneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.embedding_dim = num_classes

    def __call__(self, classes):
        hot = torch.zeros(*classes.shape, self.embedding_dim)
        return hot.scatter_(-1, classes.unsqueeze(-1), 1)

def parse(dirname, savepath, num_joints=15):
    filenames = sorted(glob.glob(path.join(dirname, '*.json')))
    poses = np.zeros([len(filenames), num_joints, 2])
    confidences = np.zeros([len(filenames), num_joints])
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
            confidence = points[:, -1]
            confidences[i, :] = confidence

        prev_pose = pose if prev_pose is None else prev_pose
        poses[i] = pose
        poses[i, confidence < 0.25] = prev_pose[confidence < 0.25]
        prev_pose = poses[i]
    np.savez(savepath, confidences=confidences, poses=poses)

    diff = poses[1:] - poses[:-1]
    dist = np.linalg.norm(diff, axis=2)

    for i in range(num_joints):
        plt.plot(dist[:1000, i])
        plt.plot(confidences[:1000, i] * 400)
        plt.show()

def animate(poses, savename, fps=30):
    camera = Camera(plt.figure())
    for pose in poses:
        pose_plot(pose, show=False)
        camera.snap()
    anim = camera.animate(blit=True)
    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(savename, writer=writer)

