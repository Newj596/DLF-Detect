import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from torch.nn import functional

class BlockMean(nn.Module):
    def __init__(self, block_size=16):
        super(BlockMean, self).__init__()
        self.block_size = block_size
        self.unfold = nn.Unfold(kernel_size=block_size, stride=block_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x=torch.mean(x,dim=1,keepdim=True)
        x_unf = self.unfold(x)  # [b, c * block_size * block_size, num_blocks]
        num_blocks = x_unf.shape[2]
        x_unf = x_unf.view(b, 1, self.block_size, self.block_size, -1)  # [b, c, block_size, block_size, num_blocks]
        block_mean = x_unf.mean(dim=(2, 3))  # [b, c, num_blocks]
        h_out = w_out = int(h / self.block_size)  
        block_mean = block_mean.view(b, 1, h_out, w_out)  # [b, c, h_out, w_out]

        return block_mean

class BlockStd(nn.Module):
    def __init__(self, block_size=16):
        super(BlockStd, self).__init__()
        self.block_size = block_size
        self.unfold = nn.Unfold(kernel_size=block_size, stride=block_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x=torch.mean(x,dim=1,keepdim=True)
        x_unf = self.unfold(x)  # [b, c * block_size * block_size, num_blocks]
        num_blocks = x_unf.shape[2]
        x_unf = x_unf.view(b, 1, self.block_size, self.block_size, -1)  # [b, c, block_size, block_size, num_blocks]
        block_mean = x_unf.std(dim=(2, 3))  # [b, c, num_blocks]
        h_out = w_out = int(h / self.block_size)  
        block_mean = block_mean.view(b, 1, h_out, w_out)  # [b, c, h_out, w_out]

        return block_mean
        
class L_Std(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(L_Std, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)

    def forward(self, x):
        x = torch.mean(x,dim=1,keepdim=True)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))

class L_Dcp(nn.Module):

    def __init__(self):
        super(L_Dcp, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
    
        dark_prior,_=torch.min(x,dim=1,keepdim=True)
        dark=torch.zeros_like(dark_prior)
    
        return self.mse(dark_prior,dark)

class L_Rec(nn.Module):

    def __init__(self):
        super(L_Rec, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x0,x1):
        return self.mse_loss(x0,x1)
        
class L_Reg(nn.Module):

    def __init__(self):
        super(L_Reg, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        sorted_tensor, _ = torch.sort(x.view(x.size(0), x.size(1), -1), dim=-1, descending=True)
        top_k_index = int(sorted_tensor.size(-1) * 1.5 / 100)
        top_k_values = sorted_tensor[:, :, :top_k_index]
        mean_values = top_k_values.mean(dim=-1, keepdim=True)
        mean_values = torch.unsqueeze(mean_values,-1)
        A=mean_values.expand_as(x)
        return self.mse_loss(x,A)
        
        

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
        
class L_expa(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_expa, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d

class L_exp(nn.Module):

    def __init__(self):
        super(L_exp, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.block_mean=BlockMean()

    def forward(self, x,t):
        x=self.block_mean(x)
        return self.mse_loss(x,t)
        
class L_std(nn.Module):

    def __init__(self):
        super(L_std, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.block_std=BlockStd()

    def forward(self, x,t):
        x=self.block_std(x)
        return self.mse_loss(x,t)


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3

