import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
import glob
from PIL import Image
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms





if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def load(name):
        state_dicts = torch.load(name)#state_dict就是一个简单的Python dictionary，其功能是将每层与层的参数张量之间一一映射
        network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
        net.load_state_dict(network_state_dict)
        try:
            optim.load_state_dict(state_dicts['opt'])
        except:
            print('Cannot load optimizer for some reason or other')


    def gauss_noise(shape):

        noise = torch.zeros(shape).cuda()
        for i in range(noise.shape[0]):
            noise[i] = torch.randn(noise[i].shape).cuda()

        return noise


    def computePSNR(origin,pred):
        origin = np.array(origin)
        origin = origin.astype(np.float32)
        pred = np.array(pred)
        pred = pred.astype(np.float32)
        mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
        if mse < 1.0e-10:
          return 100
        return 10 * math.log10(255.0**2/mse)


##########3
    # def to_rgb(image):
    #     rgb_image = Image.new("RGB", image.size)
    #     rgb_image.paste(image)  # paste中文就是粘贴的意思，所以该方法就是将paste方法中，传入的图像粘贴在原图像上。
    #     return rgb_image
    #
    #
    # def read_stegofile():
    #     # stego.file = sorted(glob.glob(c.stego_PATH + "/*." + c.format_val))
    #     image_path = "D:\\HiNet-main\\HiNet-main\\Dataset\\DIV2K\\DIV2K_stego_HR\\000.png"
    #     steg_img = Image.open(image_path)
    #     steg_img = to_rgb(steg_img)
    #     trans = transforms.ToTensor()
    #     steg_img = trans(steg_img)
    #     steg_img = steg_img.unsqueeze(dim=0)
    #     steg_img = steg_img.to(device)
    #     return steg_img

###########





    net = Model()
    net.cuda()
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    load(c.MODEL_PATH + c.suffix)

    net.eval()

    dwt = common.DWT()
    iwt = common.IWT()


    with torch.no_grad():
        for i, data in enumerate(datasets.testloader):
            data = data.to(device)
            cover = data[data.shape[0] // 2:, :, :, :]  #image.shape[0]——图片高,image.shape[1]——图片长,image.shape[2]——图片通道数
            secret = data[:data.shape[0] // 2, :, :, :]#data.shape返回的是元组data.shape[0]是行数data.shape[1]是列数
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)#将两个张量（tensor）拼接在一起

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)#可以理解为对张量的一种剪裁narrow(dim,start,length) # dim代表沿着哪个维度剪裁。start代表从dim维的第几位维开始剪裁。length代表沿着这一维度剪裁的长度
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)
            backward_z = gauss_noise(output_z.shape)
            torch.save(steg_img, "D:\\HiNet-main\\HiNet-main\\Dataset\\DIV2K\\DIV2K_stego_HR\\tensor.pth")
            torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)  # 图像保存，直接将tensor保存为图像
            torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
            torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
            # # backward_z = gauss_noise(steg_img.shape)
            #####################
            # steg_img = read_stegofile()
            # output_steg = dwt(steg_img)
            # backward_z = dwt(backward_z)
            # output_r=iwt(output_z)
            ######################
            #################
            #   backward:   #
            #################

            # output_rev = torch.cat((output_steg, backward_z), 1)
            # bacward_img = net(output_rev, rev=True)
            # secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
            # secret_rev = iwt(secret_rev)
            # cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
            # cover_rev = iwt(cover_rev)
            # resi_cover = (steg_img - cover) * 20
            # resi_secret = (secret_rev - secret) * 20
            #
            #
            # torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)
            # torchvision.utils.save_image(cover_rev, c.IMAGE_PATH_cover_rev + '%.5d.png' % i)
            # torchvision.utils.save_image(output_r, c.IMAGE_PATH_output_r + '%.5d.png' % i)






