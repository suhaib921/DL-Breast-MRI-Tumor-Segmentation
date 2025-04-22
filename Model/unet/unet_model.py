""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Elu = nn.ReLU()
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        # self.outc = (OutConvElu(64,n_classes))
        self.b_values_no0 = torch.FloatTensor([5, 50, 100, 200, 500, 800, 1000]).to(torch.device("cuda"))
        
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)        
        x = self.up3(x, x2)
        x = self.up4(x, x1)        
        logits = self.outc(x)
        # out =   input - logits
        # return out
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        


class UNet_denoise(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Elu = nn.ReLU()
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        # self.outc = (OutConvElu(64,n_classes))
        self.b_values_no0 = torch.FloatTensor([5, 50, 100, 200, 500, 800, 1000]).to(torch.device("cuda"))
        
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)        
        x = self.up3(x, x2)
        x = self.up4(x, x1)        
        logits = self.outc(x)
        out =   input - logits
        return out
        # return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
        
# Define the CNN regression model
class CNNRegressor(nn.Module):
    # def __init__(self):
    def __init__(self, n_channels, n_classes, layer=6):        
        super(CNNRegressor, self).__init__()
        self.b_values_no0 = torch.FloatTensor([5, 50, 100, 200, 500, 800, 1000]).to(torch.device("cuda"))
        self.layer = layer
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.GELU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.GELU()
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.GELU()

        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.GELU()
        
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.GELU()        

        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.GELU()
        
        self.conv8 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.GELU()
        
        self.fc1 = nn.Linear(128 *12 *12, 64)
        self.relu3 = nn.GELU() 
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        if self.layer==6 or self.layer==8:
            x = self.conv3(x)
            x = self.relu3(x)

        if self.layer==8:
            x = self.conv4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.relu5(x)

        if self.layer==6 or self.layer==8:
            x = self.conv6(x)
            x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        logits = self.relu8(x)

        # print("logits.shape",logits.shape)
        b_values_no0 = self.b_values_no0.unsqueeze(1)
        Dp = logits[:,0,:,:].reshape([-1,1,200*200])
        Dt = logits[:,1,:,:].reshape([-1,1,200*200])#.unsqueeze(1)
        Fp = logits[:,2,:,:].reshape([-1,1,200*200])#.unsqueeze(1)
        pred_x = torch.empty([logits.size()[0],len(self.b_values_no0),200*200]).to(torch.device("cuda"))
        for i in range(logits.size()[0]):
            X = torch.mul(torch.exp(-b_values_no0 * Dp[i,:,:]), Fp[i,:]) + torch.mul( torch.exp(-b_values_no0 * Dt[i,:,:]) ,(1-Fp[i,:]))
            pred_x[i] = X

        # X = Fp * torch.exp(-Dp * b_values_no0) + (1-Fp) * torch.exp(-Dt * b_values_no0)
        # pred_x = pred_x.reshape([])
        # X = X.transpose(1, 2).reshape([-1,7,200,200])
        pred_x = pred_x.reshape([-1,len(self.b_values_no0),200,200])
        Fp = Fp.reshape([-1,1,200,200])
        Dt = Dt.reshape([-1,1,200,200])
        Dp = Dp.reshape([-1,1,200,200])
        return logits, pred_x, Dp, Dt, Fp