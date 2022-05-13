import argparse
import os
from model import *
from metric import *
from data_loader import SingleLoader,MultiLoader
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
import imageio

def train_single(noise_dir,gt_dir,image_size,num_workers,batch_size,n_epoch,checkpoint,resume_single,loss_every,save_every,learning_rate):
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SingleLoader(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    model = SFD_C().to(device)
    epoch_start = 0
    if resume_single != '':
        save_dict = torch.load(os.path.join(checkpoint,resume_single))
        model.load_state_dict(save_dict['state_dict'])
        epoch_start = save_dict['epoch']
    loss_func = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                           amsgrad=False)

    for epoch in range(epoch_start,n_epoch):
        for step, (image_noise, image_gt) in enumerate(data_loader):
            image_noise = image_noise.to(device)
            image_gt = image_gt.to(device)
            pre = model(image_noise)
            loss = loss_func(pre, image_gt)
            if (step + 1) % loss_every == 0:
                print('single t = %d, loss = %.4f' % (step + 1, loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % save_every == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = os.path.join("checkpoint", "SFD_C_{}.pth.tar".format(epoch))

            torch.save(save_dict, filename)
            # torch.save(model.state_dict(), filename)

def train_multi(images_dir, num_workers, batch_size, n_epoch, checkpoint, resume_single, resume_multi, loss_every, save_every, learning_rate):
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiLoader(images_dir=images_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # for (n,g) in data_loader:
    #     print(np.moveaxis(np.squeeze(g, axis=0).cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]).shape)
    #     imageio.imwrite('teste.jpg', np.moveaxis(np.squeeze(g, axis=0).cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]))
    #     print(np.moveaxis(np.squeeze(n[:,1,:,:,:], axis=0).cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]).shape)
    #     imageio.imwrite('teste1.png', np.moveaxis(np.squeeze(n[:,1,:,:,:], axis=0).cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]))
    #     # plt.imshow(np.moveaxis(np.squeeze(g, axis=0).cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]))
    #     # plt.show()
    #     return
    # return
    model_single = SFD_C().to(device)
    epoch_start=0
    if resume_multi != '':
        model = MFD_C(model_single).to(device)
        save_dict = torch.load(os.path.join(checkpoint,resume_multi))
        model.load_state_dict(save_dict['state_dict'])
        epoch_start = save_dict['epoch']
    elif resume_single != "":
        save_dict = torch.load(os.path.join(checkpoint, resume_single))
        model_single.load_state_dict(save_dict['state_dict'])
        model = MFD_C(model_single).to(device)
    else:
        model = MFD_C(model_single).to(device)
    loss_func = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                           amsgrad=False)
    model.train()
    print('Began')
    for epoch in range(epoch_start,n_epoch):
        for step, (image_noise, image_gt) in enumerate(data_loader):
            image_noise_batch = image_noise.to(device)
            image_gt = image_gt.to(device)
            # print("image_noise_batch  : ",image_noise_batch.size())
            # print("image_gt   : ",image_gt.size())
            # print(image_noise_batch.size())
            batch_size_i = image_noise_batch.size()[0]
            burst_size = image_noise_batch.size()[1]
            mfinit1, mfinit2, mfinit3,mfinit4,mfinit5,mfinit6,mfinit7 = torch.zeros(7, batch_size_i, 64, 64, 64).to(device)
            mfinit8 = torch.zeros(batch_size_i, 3, 64, 64).to(device)
            i = 0
            for i_burst in range(burst_size):
                frame = image_noise_batch[:,i_burst,:,:,:]
                if i == 0:
                    i += 1
                    dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8 = model(
                        frame, mfinit1, mfinit2, mfinit3, mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                    loss_sfd = loss_func(dframe, image_gt)
                    loss_mfd = loss_func(mf8, image_gt)

                else:
                    dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8= model(frame, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8)
                    loss_sfd += loss_func(dframe, image_gt)
                    loss_mfd += loss_func(mf8, image_gt)
            loss = loss_sfd + loss_mfd
            if (step) % loss_every == 0:
                print('epoch %d  multi t = %d, loss = %.4f' % (epoch,step + 1, loss.data))
                imageio.imwrite(f'Images/GT_{step+1}_{epoch}.jpg', np.moveaxis(image_gt[0].cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]))
                imageio.imwrite(f'Images/PRED_{step+1}_{epoch}.jpg', np.moveaxis(dframe[0].cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % save_every == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = os.path.join("checkpoint", "CNN" ,"MFD_C_{}.pth.tar".format(epoch))

            torch.save(save_dict, filename)
    pass


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--images_dir','-n', default='D:/DeepLearning/Projeto/Zurich/train/huawei_raw_noisy_burst', help='path to the burst directory')
    parser.add_argument('--num_workers', '-nw', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--batch_size', '-bs', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--n_epoch', '-ep', default=10, type=int, help='number of workers in data loader')
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='number of workers in data loader')
    parser.add_argument('--loss_every', '-le', default=1000, type=int, help='number of inter to print loss')
    parser.add_argument('--save_every', '-se', default=1, type=int, help='number of epoch to save checkpoint')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the folder checkpoint to save')
    parser.add_argument('--resume_multi', '-rm', type=str, default='',
                        help='file name of checkpoint')
    parser.add_argument('--resume_single', '-rs', type=str, default='',
                        help='file name of checkpoint')
    parser.add_argument('--type_model', '-t', type=str, default='multi',help='type model train is single or multi')
    args = parser.parse_args()
    #
    if args.type_model == 'single':
        train_single(args.images_dir,args.num_workers,args.batch_size,args.n_epoch,args.checkpoint,args.resume_single,args.loss_every,args.save_every,args.learning_rate)
    elif args.type_model == 'multi':
        train_multi(args.images_dir,args.num_workers,args.batch_size,args.n_epoch,args.checkpoint,args.resume_single,args.resume_multi,args.loss_every,args.save_every,args.learning_rate)



