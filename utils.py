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

def pose_plot(pose, show=True, pause=None, savepath=None):
    plt.figure()
    for i, j in BONE_LIST:
        plt.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], color='b')
    plt.scatter(pose[:, 0], pose[:, 1], color='blue')
    plt.gca().set_aspect('equal', adjustable='box')
    for i, coordinate in enumerate(pose):
        plt.annotate(i, coordinate, fontsize=10)
    if savepath is not None:
        plt.savefig(savepath)
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
    def __init__(self, poses, labels=None):
        self.poses = poses
        # self.labels = labels.repeat(poses.shape[1])

    def __getitem__(self, idx):
        return self.poses[idx].astype(np.float32), 0

    def __len__(self):
        return len(self.poses)

class SeqPoses(Dataset):
    def __init__(self, all_poses, labels, length=50):
        total = 0
        lensum = [0]
        for poses in all_poses:
            total += poses.shape[0] - length
            lensum.append(total)
        self.lensum = lensum
        self.lengths = [poses.shape[0] for poses in all_poses]
        self.all_poses = all_poses
        # self.labels = labels.repeat(poses.shape[1])
        self.length = length

    def __getitem__(self, idx):
        batch_idx = 0
        while self.lengths[batch_idx] <= idx:
            idx -= self.lengths[batch_idx]
            batch_idx += 1
        return self.all_poses[batch_idx][idx:idx + self.length].astype(
            np.float32), 0

    def __len__(self):
        return sum(self.lengths)


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
    alpha = torch.rand((batch_size, 1), device=device)

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
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.embedding_dim = num_classes
        self.device = device

    def __call__(self, classes):
        hot = torch.zeros(*classes.shape, self.embedding_dim, device=self.device)
        return hot.scatter_(-1, classes.unsqueeze(-1), 1)

def animate(poses, savename, fps=30):
    camera = Camera(plt.figure())
    for pose in poses:
        pose_plot(pose, show=False)
        camera.snap()
    anim = camera.animate(blit=True)
    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(savename, writer=writer)

