import argparse
import os
from model import *
from metric import *
from data_loader import SingleLoader,MultiLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from metric import  psnr
import torchvision.transforms as transforms
import imageio

def test_single(noise_dir,gt_dir,image_size,num_workers,checkpoint,resume):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SingleLoader(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=num_workers)
    model = SFD_C().to(device)
    if resume != '':
        save_dict = torch.load(os.path.join(checkpoint,resume))
        # model.load_state_dict(save_dict['state_dict'])
        model.load_state_dict(save_dict['state_dict'])
    for step, (image_noise, image_gt) in enumerate(data_loader):
        image_noise = image_noise.to(device)
        image_gt = image_gt.to(device)

        pre = model(image_noise)
        image_gt = np.array(np.transpose(image_gt[0].detach().numpy(), (1, 2, 0))*255,dtype=int)
        image_noise = np.array(np.transpose(image_noise[0].detach().numpy(), (1, 2, 0))*255,dtype=int)
        pre = np.array(np.transpose(pre[0].detach().numpy(), (1, 2, 0))*255,dtype=int)
        print(pre)
        print(" Noise : ",psnr(image_noise,image_gt), "   pre : ",psnr(pre,image_gt))
        plt.subplot(1,2,1)
        plt.imshow(image_noise)
        plt.subplot(1, 2, 2)
        plt.imshow(pre)
        plt.show()

# def denoise_burst(burst_image,gt):


def test_multi(images_dir,num_workers,checkpoint,resume):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiLoader(images_dir=images_dir)
    data_loader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=num_workers)
    model_single = SFD_C().to(device)
    model = MFD_C(model_single).to(device)
    if resume != '':
        print(device)
        # save_dict = torch.load(os.path.join(checkpoint, resume), map_location=torch.device('cpu'))

        if device == "cpu":
            save_dict = torch.load(os.path.join(checkpoint,resume),map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(os.path.join(checkpoint, resume))
        model.load_state_dict(save_dict['state_dict'])
    model.eval()
    #trans = transforms.ToPILImage()
    #for i in range(10):
    for step, (image_noise, image_gt) in enumerate(data_loader):

        image_noise_batch = image_noise.to(device)
        image_gt = image_gt.to(device)
        # print(image_noise_batch.size())
        burst_size = image_noise_batch.size()[0]
        mfinit1, mfinit2, mfinit3,mfinit4,mfinit5,mfinit6,mfinit7 = torch.zeros(7, 1, 64, 64, 64).to(device)
        mfinit8 = torch.zeros(1, 3, 64, 64).to(device)
        i = 0
        for i_burst in range(burst_size):
            frame = image_noise_batch[:,i_burst,:,:,:]
            # print(frame.size())
            if i == 0:
                i += 1
                dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8 = model(
                    frame, mfinit1, mfinit2, mfinit3, mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
            else:
                dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8= model(dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8)
        # # print(np.array(trans(mf8[0])))
        # print(np.array(trans(dframe[0])).shape)
        # print(np.array(trans(image_gt[0])).shape)
        # plt.imshow(np.array(trans(dframe[0])))
        # plt.show()
        # plt.imshow(np.array(trans(image_gt[0])))
        # plt.show()
        #print(psnr(np.array(trans(dframe[0])),np.array(trans(image_gt[0]))))
        os.mkdir(os.path.join(r'D:\DeepLearning\Projeto\Zurich\train\Test_result_Modified_CNN', str(step)))
        imageio.imwrite(os.path.join(r'D:\DeepLearning\Projeto\Zurich\train\Test_result_Modified_CNN', str(step), 'GT.jpg'), np.moveaxis(image_gt[0].cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]))
        imageio.imwrite(os.path.join(r'D:\DeepLearning\Projeto\Zurich\train\Test_result_Modified_CNN', str(step), 'PRED.jpg'), np.moveaxis(dframe[0].cpu().detach().numpy(), [0, 1, 2], [2, 0, 1]))

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--images_dir','-n', default='D:/DeepLearning/Projeto/Zurich/train/huawei_raw_noisy_burst_test', help='path to noise image file')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint/CNN',
                        help='the folder checkpoint to save')
    parser.add_argument('--resume', '-r', type=str, default="MFD_C_9.pth.tar",
                        help='file name of checkpoint')
    parser.add_argument('--type_model', '-t', type=str, default='multi',help='type model train is single or multi')
    args = parser.parse_args()
    #
    if args.type_model == 'single':
        test_single(args.images_dir,args.num_workers,args.checkpoint,args.resume)
    elif args.type_model == 'multi':
        test_multi(args.images_dir,args.num_workers,args.checkpoint,args.resume)



