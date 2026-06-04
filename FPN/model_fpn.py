import torch.nn as nn
import torch.nn.functional as F
import torch


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))

        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))

        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))

        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))

        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))

        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)

        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)

        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)

        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)

        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)

        b6 = self.block6(mp5)

        return b1,b2,b3,b4,b5,b6


#6 tensori in uscita come i blocchi dell'encoder
#non devo generare le mappe di feature a diversa scala perchè uso già quelle dell'encoder
#quindi parto dalle convoluzione laterali per ridimensionare in profondità
class DecoderFPN(nn.Module):
    def __init__(self, base_width=64, fpn_channels=256, out_channels=2):
        super(DecoderFPN, self).__init__()
        #conv 1x1
        self.depth_conv6 = nn.Conv2d(base_width * 8, fpn_channels, kernel_size=1)
        self.depth_conv5 = nn.Conv2d(base_width * 8, fpn_channels, kernel_size=1)
        self.depth_conv4 = nn.Conv2d(base_width * 8, fpn_channels, kernel_size=1)
        self.depth_conv3 = nn.Conv2d(base_width * 4, fpn_channels, kernel_size=1)
        self.depth_conv2 = nn.Conv2d(base_width * 2, fpn_channels, kernel_size=1)
        self.depth_conv1 = nn.Conv2d(base_width * 1, fpn_channels, kernel_size=1)

        #convluzioni 3x3 di smoothing per eliminare errori di upsampling
        self.smooth6 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
       
        #Output
        #fpn_channels * 6 perchè riceve sei mappe di feature
        self.fin_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * 6, fpn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fpn_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, b1, b2, b3, b4, b5, b6):

        #Somma mappe di feature + smoothin 3x3
        p6 = self.depth_conv6(b6)

        p5 = self.depth_conv5(b5) + F.interpolate(p6, scale_factor=2, mode='bilinear', align_corners=False)
        p5 = self.smooth5(p5)

        p4 = self.depth_conv4(b4) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)
        p4 = self.smooth4(p4)

        p3 = self.depth_conv3(b3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.smooth3(p3)

        p2 = self.depth_conv2(b2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.smooth2(p2)

        p1 = self.depth_conv1(b1) + F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p1 = self.smooth1(p1)

        #Upsample a 256x256
        original_size = (256,256)

        p6_up = F.interpolate(p6, size=original_size, mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=original_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=original_size, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=original_size, mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=original_size, mode='bilinear', align_corners=False)
        p1_up = F.interpolate(p1, size=original_size, mode='bilinear', align_corners=False)

        #Concatenazione mappe
        out_concat = torch.cat((p1_up,p2_up,p3_up,p4_up,p5_up,p6_up),dim=1)

        return self.fin_conv(out_concat)

#Wrapper per la classe
class FPNNetwork(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, base_width=64):
        super(FPNNetwork, self).__init__()
        self.encoder = EncoderDiscriminative(in_channels, base_width)
        self.decoder = DecoderFPN(base_width, fpn_channels=256, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder(x)
        fpn_output = self.decoder(b1,b2,b3,b4,b5,b6)
        return fpn_output
    
    def getName(self):
        return self.__class__.__name__
    