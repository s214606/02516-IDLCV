# from turtle import forward
from numpy import concat
import torch 
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling + skip connections)
        # an alternative can be self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False) LOOK into MODE and ALIGN_CORNER
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        x0 = F.relu(self.enc_conv0(x))
        e0 = self.pool0(x0)
        x1 = F.relu(self.enc_conv1(e0))
        e1 = self.pool1(x1)
        x2 = F.relu(self.enc_conv2(e1))
        e2 = self.pool2(x2)
        x3 = F.relu(self.enc_conv3(e2))
        e3 = self.pool3(x3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        upsampling0 = self.upsample0(b)
        concat_0 = torch.concat([upsampling0, x3], 1)
        d0 = F.relu(self.dec_conv0(concat_0))
        upsampling1 = self.upsample1(d0)
        concat_1 = torch.concat([upsampling1, x2], 1)
        d1 = F.relu(self.dec_conv1(concat_1))
        upsampling2 = self.upsample2(d1)
        concat_2 = torch.concat([upsampling2, x1], 1)
        d2 = F.relu(self.dec_conv2(concat_2))
        upsampling3 = self.upsample3(d2)
        concat_3 = torch.concat([upsampling3, x0], 1)
        d3 = self.dec_conv3(concat_3)      
        return d3



class UNet2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) #128
        self.pool1 = nn.Conv2d(64, 64, 3, stride = 2, padding=1) #128->64
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.Conv2d(64, 64, 3, stride = 2, padding=1) #64->32
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.Conv2d(64, 64, 3, stride = 2, padding=1) #32->16
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.Conv2d(64, 64, 3, stride = 2, padding=1) #16->8

        self.bottleneck = nn.Conv2d(64,64,3,padding=1)

        self.up1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1) #8->16
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1) #16->32
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 64, 3, stride=2,padding=1, output_padding=1) #32->64
        self.dec_conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.up4 = nn.ConvTranspose2d(64, 64, 3, stride=2,padding=1, output_padding=1) #64->128
        self.dec_conv4 = nn.Conv2d(128, 64, 3, padding=1)

    def forward(self, x):
        #Encoder
        x0 = F.relu(self.conv1(x))
        e0 = self.pool1(x0)
        x1 = F.relu(self.conv2(e0))
        e1 = self.pool2(x1)
        x2 = F.relu(self.conv3(e1))
        e2 = self.pool3(x2)
        x3 = F.relu(self.conv4(e2))
        e3 = self.pool4(x3)

        b = self.bottleneck(e3)

        #Decoder
        upsamp0 = self.up1(b)
        concat0 = torch.concat([upsamp0,x3], 1)
        dec_conv0 = F.relu((self.dec_conv1(concat0)))
        upsamp1 = self.up2(dec_conv0)
        concat1 = torch.concat([upsamp1,x2], 1)
        dec_conv1 = F.relu((self.dec_conv2(concat1)))
        upsamp2 = self.up3(dec_conv1)
        concat2 = torch.concat([upsamp2,x1], 1)
        dec_conv2 = F.relu((self.dec_conv3(concat2)))
        upsamp3 = self.up4(dec_conv2)
        concat3 = torch.concat([upsamp3,x0], 1)
        dec_conv3 = self.dec_conv2(concat3)
        return dec_conv3

# alternative transpose conv is self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
class UNet256(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        """
        U-Net architecture for image segmentation
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            init_features: Number of features in first layer (doubles each level)
        """
        super(UNet256, self).__init__()
        
        features = init_features
        
        # Encoder (downsampling path)
        self.encoder1 = self._make_encoder_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._make_encoder_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._make_encoder_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._make_encoder_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(features * 8, features * 16)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._make_decoder_block(features * 2, features)
        
        # Final output layer
        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def _make_encoder_block(self, in_channels, out_channels):
        """Double convolution block for encoder"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Double convolution block for decoder"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path - save skip connections
        enc1 = self.encoder1(x)      # 64 channels
        enc2 = self.encoder2(self.pool1(enc1))  # 128 channels
        enc3 = self.encoder3(self.pool2(enc2))  # 256 channels
        enc4 = self.encoder4(self.pool3(enc3))  # 512 channels
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # 1024 channels
        
        # Decoder path - concatenate skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concatenate along channel dimension
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        return self.out_conv(dec1)

