import torch
import numpy as np
from utils import block
from torch import nn

class PoseGenerator(nn.Module):
    def __init__(self, embed, pose_z_dim, pose_dim):
        super(PoseGenerator, self).__init__()

        # self.embed = embed
        self.z_dim = pose_z_dim
        # in_feat = pose_z_dim + embed.embedding_dim
        in_feat = pose_z_dim
        self.model = nn.Sequential(
            *block(in_feat, 500),
            # *block(500, 500),
            *block(500, 500, dropout=True),
            nn.Linear(500, pose_dim)
        )

    def forward(self, pose_z, classes):
        # embed = self.embed(classes)
        # if len(pose_z.shape) == 3:
        #     embed = embed.unsqueeze(1).expand(-1, pose_z.size(1), -1)
        # input = torch.cat((embed, pose_z), -1)
        input = pose_z
        shape = input.shape[:-1]
        return self.model(input.view(np.prod(shape), -1)).view(*shape, -1)


class PoseDiscriminator(nn.Module):
    def __init__(self, embed, pose_dim):
        super(PoseDiscriminator, self).__init__()
        # self.embed = embed
        # in_feat = embed.embedding_dim + pose_dim
        in_feat = pose_dim

        self.model = nn.Sequential(
            *block(in_feat, 500, normalize=False),
            # *block(500, 500, normalize=False),
            *block(500, 500, normalize=False, dropout=True),
            nn.Linear(500, 1),
        )

    def forward(self, pose, classes):
        # input = torch.cat((self.embed(classes), pose), -1)
        input = pose
        return self.model(input).view(-1)


class SeqGenerator(nn.Module):
    def __init__(self, embed, num_timesteps, pose_z_dim, z_dim):
        super(SeqGenerator, self).__init__()

        self.num_timesteps = num_timesteps
        self.pose_z_dim = pose_z_dim
        # self.embed = embed
        # in_feat = z_dim + pose_z_dim + self.embed.embedding_dim
        in_feat = z_dim + pose_z_dim
        self.model = nn.Sequential(
            *block(in_feat, 500),
            # *block(500, 500),
            *block(500, 500, dropout=True),
            nn.Linear(500, pose_z_dim * num_timesteps)
        )

    def forward(self, pose_z, z, classes):
        # emb = self.embed(classes)
        # input = torch.cat([z, pose_z, emb], dim=1)
        input = torch.cat([z, pose_z], dim=1)
        deltas = self.model(input).view(-1, self.num_timesteps, self.pose_z_dim)
        seq = torch.cat([pose_z.unsqueeze(1), deltas], dim=1)
        return seq, torch.cumsum(seq, dim=1)


class SeqDiscriminator(nn.Module):
    def __init__(self, embed, pose_dim, hidden_dim, num_layers=1):
        super(SeqDiscriminator, self).__init__()

        # self.embed = embed
        # in_feat = embed.embedding_dim + pose_dim
        in_feat = pose_dim
        self.lstm = nn.LSTM(in_feat, hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.fcs = nn.Sequential(
            *block(hidden_dim * 4, 500, normalize=False),
            # *block(500, 500, normalize=False),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )

    def forward(self, poses, classes):
        # embed = self.embed(classes).unsqueeze(1).expand(-1, poses.size(1), -1)
        # input = torch.cat([poses, embed], dim=2)
        input = poses
        out, _ = self.lstm(input)
        cat = torch.cat((out[:, 0, :], out[:, -1, :]), dim=-1)
        return self.fcs(cat)


class Total(nn.Module):
    def __init__(self, num_classes, hidden_dim, pose_z_dim, z_dim, num_joints,
                 num_timesteps, num_layers=1):
        super(Total, self).__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.pose_dim = num_joints * 2

        self.pose_gen = PoseGenerator(pose_z_dim, self.embedding,
                                      self.pose_dim)
        self.pose_dsc = PoseDiscriminator(self.embedding, self.pose_dim)
        self.seq_gen = SeqGenerator(self.embedding, num_timesteps,
                                    pose_z_dim, z_dim)

        self.seq_dsc = SeqDiscriminator(self.pose_dim, hidden_dim, num_layers)

