from models import *
from utils import *
from torch.nn import DataParallel
from tqdm import tqdm, trange
import argparse
from torch import optim
from torch.utils.data import DataLoader


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)

    # poses, labels = get_fake_x(num_timesteps, 100)
    batch_size = 200
    filenames = glob.glob(path.join(datadir, '*.npz'))
    list_poses = []
    for filename in filenames:
        data = np.load(filename, allow_pickle=True)
        poses = data['poses']
        poses = translate(poses, 8)
        poses = poses.reshape(poses.shape[0], -1)
        list_poses.append(poses)
    train_loader = DataLoader(SeqPoses(list_poses, None), batch_size=batch_size,
                              shuffle=True)
    num_joints = list_poses[0].shape[-1] // 2

    if torch.cuda.is_available() and num_gpu > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Instantiate Models
    # embed = nn.Embedding(num_classes, num_classes)
    # embed = OneHot(num_classes)
    embed = None
    pose_dim = num_joints * 2

    pose_gen = DataParallel(PoseGenerator(embed, pose_z_dim, pose_dim))
    pose_gen.to(device)


    # TODO: make model checkpoint a param.
    pose_gen.load_state_dict(torch.load('models/pose_gen0.pt'))
    pose_gen.eval()
    seq_gen = DataParallel(SeqGenerator(embed, num_timesteps - 1, pose_z_dim,
                                        seq_z_dim))
    seq_dsc = DataParallel(SeqDiscriminator(embed, pose_dim, hidden_dim,
                                            num_layers))
    seq_gen.to(device)
    seq_dsc.to(device)
    seq_gen.train()
    seq_dsc.train()

    optim_seq_dsc = optim.Adam(seq_dsc.parameters(), lr=lr, betas=betas)
    optim_seq_gen = optim.Adam(seq_gen.parameters(), lr=lr, betas=betas)

    real_label = 1
    fake_label = 0

    bce = nn.BCELoss()

    iter = 0
    for epoch in trange(init_epoch, epochs):
        for i, (real_poses, _) in enumerate(train_loader):

            batch_size = real_poses.shape[0]
            real_poses = real_poses.to(device)
            classes = None

            # Seq discriminator
            seq_dsc.zero_grad()
            pose_z = torch.randn(batch_size, pose_z_dim, device=device)
            seq_z = torch.randn(batch_size, seq_z_dim, device=device)
            # fake_classes = torch.randint(num_classes, size=(batch_size,))
            fake_classes = None
            _, fake_seq_z = seq_gen(pose_z, seq_z, fake_classes)
            fake_seq_pose = pose_gen(fake_seq_z, fake_classes)

            real_validity = seq_dsc(real_poses, classes)
            fake_validity = seq_dsc(fake_seq_pose.detach(), fake_classes)

            real_labels = torch.full((batch_size,), real_label, device=device)
            fake_labels = torch.full((batch_size,), fake_label, device=device)
            loss_real = bce(real_validity, real_labels)
            loss_fake = bce(fake_validity, fake_labels)
            loss_dsc = loss_real + loss_fake
            loss_dsc.backward()
            optim_seq_dsc.step()

            # Seq Generator
            seq_gen.zero_grad()
            pose_z = torch.randn(batch_size, pose_z_dim, device=device)
            seq_z = torch.randn(batch_size, seq_z_dim, device=device)
            # fake_classes = torch.randint(num_classes, size=(batch_size,))
            fake_seq_dz, fake_seq_z = seq_gen(pose_z, seq_z, fake_classes)
            fake_seq_pose = pose_gen(fake_seq_z, fake_classes)
            fake_validity = seq_dsc(fake_seq_pose, fake_classes)
            loss_gen = bce(fake_validity, real_labels)
            # TODO: should i do this? test without
            # loss_gen = loss_gen + lambda_mse * torch.norm(fake_seq_dz)
            loss_gen.backward()
            optim_seq_gen.step()

            if iter % log_interval == 0:
                tqdm.write(f'Epoch {epoch} \t'
                           f'Iter: {iter: 3}\t'
                           f'Loss Dsc: {loss_dsc.item(): 7.3f}\t'
                           f'Loss Gen: {loss_gen.item(): 7.3f}')

            if iter % anim_interval == 0:
                animate(fake_seq_pose[0].view(-1, 15, 2).cpu().detach().numpy(),
                        f'vis/anim_{iter}.mp4')
                np.savez()

            if iter % save_interval == 0:
                torch.save(seq_gen.state_dict(),
                           path.join(model_path, f'gen/seq_gen{iter}.pt'))
                torch.save(seq_dsc.state_dict(),
                           path.join(model_path, f'dsc/seq_dsc{iter}.pt'))


            # if iter % show_interval == 0:
            #     pose_gen.eval()
            #     pose_z = torch.randn(h, pose_z_dim, device=device)
            #     single_classes = torch.full((1,), 0, device=device,
            #                                 dtype=int)
            #     fake_pose = pose_gen(pose_z, single_classes).detach().squeeze()
            #     fake_pose = fake_pose.view(-1, 2)
            #     pose_plot(fake_pose)
            #     pose_gen.train()

            iter += 1

if __name__ == '__main__':
    seed = 0
    num_timesteps = 50
    seq_z_dim = 100
    pose_z_dim = 100
    hidden_dim = 100
    num_layers = 2
    num_hidden = 10
    lr = 0.001
    betas = (0.5, 0.999)
    num_gpu = 4
    init_epoch = 0
    epochs = 50
    lambda_mse = 0.01
    log_interval = 10
    show_interval = 50
    anim_interval = 10
    save_interval = 500
    datadir = 'data/parsed'
    model_path = 'models/seq'

    main()

