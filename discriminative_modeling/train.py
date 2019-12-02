import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import os.path as path
import glob
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader

class MusicEncoder(nn.Module):
    def __init__(self, music_feats, music_emb, out_size=128, hid_size=256, num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        
        self.num_layers = num_layers
        self.music_embedding = nn.Linear(music_feats, music_emb)
        self.lstm = nn.LSTM(input_size = music_emb, hidden_size = hid_size, num_layers = num_layers, batch_first = True, bidirectional = True, dropout = dropout)

    def forward(self, music):
        music = self.music_embedding(music)
        _, hid = self.lstm(music)
        return hid[0][2*self.num_layers-2, :]
    
class PoseDecoder(nn.Module):
    def __init__(self, hid_size=256, out_size=128):
        super().__init__()
        self.prev_embedder = nn.Embedding(out_size, hid_size)
        self.cls = nn.Sequential(
            nn.Linear(2*hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size)
        )
        self.out_size = out_size
        self.hid_size = hid_size
    def forward(self, music_emb, prev=None):
        prev_emb = torch.zeros((music_emb.size(0), self.hid_size), device=music_emb.device, dtype=music_emb.dtype) if prev is None else self.prev_embedder(prev)
        new_emb = torch.cat([music_emb, prev_emb], 1)
        return self.cls(new_emb)
enc = MusicEncoder(512, 256, 128, 256, 2, 0)
dec = PoseDecoder()
#enc.load_state_dict(torch.load("enc.pt"))
#dec.load_state_dict(torch.load("dec.pt"))
class PairDataset(Dataset):
    def __init__(self, poses, music):
        super().__init__()
        self.poses = poses
        self.music = music
    def __getitem__(self, index):
        return self.music[index], self.poses[index]
    def __len__(self):
        return len(self.poses)
data = PairDataset(np.load("labels.npy"), np.load("music.npy"))
enc.to("cuda")
dec.to("cuda")
enc_optim = optim.SGD(enc.parameters(), lr=3e-4)
dec_optim = optim.SGD(dec.parameters(), lr=3e-4)
loss_func = nn.CrossEntropyLoss()

def train(epoch):
    print(f"Epoch: {epoch}")
    dataloader = DataLoader(data, batch_size=4)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.0
    for i, batch in pbar:
        loss = torch.tensor(0.0).to("cuda")
        music, pose = batch
        music, pose = music.cuda(), pose.cuda()
        music = music.to(dtype=torch.float32).transpose(1, 2)
        pose = pose.to(dtype=torch.long)
        accuracy = 0.0
        
        music_enc = enc(music)
        for j in range(pose.size(1)):
            pose_pred = dec(music_enc, prev=pose[:, j-1] if i > 0 else None)
            loss += loss_func(pose_pred, pose[:, j])
            accuracy += (torch.argmax(pose_pred, 1) == pose[:, j]).sum().float() / pose.size(1)
            
        total_loss += loss.item()/pose.size(1)
        accuracy = accuracy.item()

        pbar.set_postfix(loss=loss.item()/pose.size(1), accuracy=accuracy)
        loss.backward()
        enc_optim.step()
        dec_optim.step()
    pbar.set_postfix(loss=(total_loss/len(dataloader)), accuracy=accuracy/len(dataloader))
    pbar.close()
[train(i) for i in range(10)]
torch.save(enc.state_dict(), "enc.pt")
torch.save(dec.state_dict(), "dec.pt")