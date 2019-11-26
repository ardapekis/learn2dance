from models import *
from utils import *
from tqdm import tqdm
import argparse
from torch import optim
from torch.utils.data import DataLoader


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)

    # poses, labels = get_fake_x(num_timesteps, 100)
    data = np.load(data_file, allow_pickle=True)
    poses = data['poses']
    poses = translate(poses, 8)
    poses = poses.reshape(1, poses.shape[0], -1)
    labels = np.zeros(1)
    train_loader = DataLoader(Poses(poses, labels), batch_size=100,
                              shuffle=True)
    num_classes = len(np.unique(labels))
    num_joints = poses.shape[2] // 2

    if torch.cuda.is_available() and num_gpu > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Instantiate Models
    # embed = nn.Embedding(num_classes, num_classes)
    embed = OneHot(num_classes)
    pose_dim = num_joints * 2

    pose_gen = PoseGenerator(embed, pose_z_dim, pose_dim)
    pose_dsc = PoseDiscriminator(embed, pose_dim)
    pose_gen.train()
    pose_dsc.train()

    optim_pose_dsc = optim.Adam(pose_dsc.parameters(), lr=lr, betas=betas)
    optim_pose_gen = optim.Adam(pose_gen.parameters(), lr=lr, betas=betas)

    iter = 0
    for epoch in range(init_epoch, epochs):
        for i, (real_poses, classes) in enumerate(train_loader):

            batch_size = real_poses.shape[0]
            real_poses = real_poses.to(device)


            # Pose discriminator
            pose_dsc.zero_grad()
            pose_z = torch.randn(batch_size, pose_z_dim, device=device)
            fake_classes = torch.randint(num_classes, size=(batch_size,))
            fake_poses = pose_gen(pose_z, fake_classes)

            real_validity = pose_dsc(real_poses, classes)
            fake_validity = pose_dsc(fake_poses.detach(), fake_classes)
            gp = gradient_penalty(pose_dsc, real_poses.detach(),
                                  fake_poses.detach(), fake_classes, device)
            dsc_loss = -torch.mean(real_validity) + torch.mean(fake_validity)\
                       + lambda_gp * gp

            dsc_loss.backward()
            optim_pose_dsc.step()

            # Pose generator
            pose_gen.zero_grad()
            pose_z = torch.randn(batch_size, pose_z_dim, device=device)
            fake_classes = torch.randint(num_classes, size=(batch_size,))
            fake_poses = pose_gen(pose_z, fake_classes)
            fake_validity = pose_dsc(fake_poses, fake_classes)
            gen_loss = -torch.mean(fake_validity)
            gen_loss.backward()
            optim_pose_gen.step()

            if iter % log_interval == 0:
                tqdm.write(f'Epoch {epoch} \t'
                           f'Iter: {iter: 3}\t'
                           f'Loss Dsc: {dsc_loss.item(): 7.3f}\t'
                           f'Loss Gen: {gen_loss.item(): 7.3f}')

            if iter % show_interval == 0:
                pose_gen.eval()
                pose_z = torch.randn(1, pose_z_dim, device=device)
                single_classes = torch.full((1,), 0, device=device,
                                            dtype=int)
                fake_pose = pose_gen(pose_z, single_classes).detach().squeeze()
                fake_pose = fake_pose.view(-1, 2)
                pose_plot(fake_pose)
                pose_gen.train()

            if iter % save_interval == 0:
                torch.save(pose_gen.state_dict(),
                           path.join(model_path, f'pose_gen{iter}.pt'))
                torch.save(pose_dsc.state_dict(),
                           path.join(model_path, f'pose_dsc{iter}.pt'))

            iter += 1

    torch.save(pose_gen.state_dict(), path.join(model_path,
                                                'pose_gen.pt'))
    torch.save(pose_dsc.state_dict(), path.join(model_path,
                                                'pose_dsc.pt'))


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
    num_gpu = 0
    init_epoch = 0
    epochs = 50
    lambda_gp = 10
    log_interval = 10
    show_interval = 2000
    save_interval = 100
    data_file = 'data/poses.npz'
    model_path = 'models/'

    main()

