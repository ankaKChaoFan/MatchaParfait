import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import interpolate

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups= out_channels//16, num_channels= out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,3, padding=1),
            nn.GroupNorm(num_groups= out_channels//16, num_channels= out_channels),
            nn.ReLU(),
        )
    def forward(self,x):
        if self.down_sample:
            x = F.interpolate(x,scale_factor= 0.5,mode='bicubic')
        x = self.conv(x)
        return x

def upcat(small,large):
    return torch.cat([F.interpolate(small, scale_factor= 2,mode='bicubic'), large],1)

class UNet(nn.Module):
    def __init__(self, in_channels= 1, out_channels= 10, c= None):
        super().__init__()
        if c is None:
            c = {
                512:32,
                256:32,
                128:32,
                64:32,
                32:64,
                16:64,
                8:64,
                4:64,
            }
        self.C512 = nn.Sequential(
            #nn.Conv2d(in_channels,c[512],3,padding=1),
            #ConvBlock(c[512],c[512])
            ConvBlock(in_channels=in_channels,out_channels=c[512])
        )

        self.C256 = ConvBlock(c[512],c[256],down_sample=True)
        self.C128 = ConvBlock(c[256],c[128],down_sample=True)
        self.C64 = ConvBlock(c[128],c[64],down_sample=True)
        self.C32 = ConvBlock(c[64],c[32],down_sample=True)
        self.C16 = ConvBlock(c[32],c[16],down_sample=True)
        self.C8 = ConvBlock(c[16],c[8],down_sample=True)
        self.C4 = ConvBlock(c[8],c[4],down_sample=True)

        self.U8 = ConvBlock(c[4]+c[8],c[8])
        self.U16 = ConvBlock(c[8]+c[16],c[16])
        self.U32 = ConvBlock(c[16]+c[32],c[32])
        self.U64 = ConvBlock(c[32]+c[64],c[64])
        self.U128 = ConvBlock(c[64]+c[128],c[128])
        self.U256 = ConvBlock(c[128]+c[256],c[256])

        self.U512 = nn.Sequential(
            ConvBlock(c[256]+c[512],c[512]),
            nn.Conv2d(c[512],out_channels,3,padding=1)
        )
    def half_heatmap(self, x, heatmap=True):
        x512 = self.C512(x)
        x256 = self.C256(x512)
        x128 = self.C128(x256)
        x64 = self.C64(x128)
        x32 = self.C32(x64)
        x16 = self.C16(x32)
        x8 = self.C8(x16)
        x4 = self.C4(x8)

        x8 = self.U8( upcat(x4,x8) )
        x16 = self.U16( upcat(x8,x16) )
        x32 = self.U32( upcat(x16,x32) )
        x64 = self.U64( upcat(x32,x64) )
        x128 = self.U128( upcat(x64,x128) )
        x256 = self.U256( upcat(x128,x256) )
        x512 = self.U512( upcat(x256,x512) )
        
        if heatmap:
            shape = x512.shape
            x512 = F.softmax(x512.flatten(2),dim=2).reshape(shape)
        return x512
    
    def full_heatmap(self, x, heatmap=True):
        Lpage = self.half_heatmap(x, heatmap)
        Rpage = self.half_heatmap(x.flip([3]), heatmap).flip([3])
        Rpage = torch.cat( [
            Rpage[:,:Rpage.shape[1]//2].flip([1]),
            Rpage[:,Rpage.shape[1]//2:].flip([1])
        ],1)
        return torch.cat([Lpage,Rpage],1)
    
    #ここからはバッチサイズ1で推論することを想定。すまんなｗ！
    @torch.no_grad()
    def post_proc_mask(self,p1,p2,robustness, device):
        centre = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
        length = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) + 1
        grid = torch.cat([
            torch.arange(512,device=device).reshape(1,1,1,-1).repeat(1,1,384,1).add(-centre[0]),
            torch.arange(384,device=device).reshape(1,1,-1,1).repeat(1,1,1,512).add(-centre[1])],1).to(device)

        theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        M = torch.tensor([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).float().transpose(0,1).reshape(2,2,1,1).to(device)
        grid = F.conv2d(grid,M)

        grid = torch.cat([grid[:,:1,:,:].mul(robustness/length),
                        grid[:,1:,:,:].mul(1/10)],1)

        grid = grid[:,:1,:,:].pow(2) + grid[:,1:,:,:].pow(2) 
        grid = grid.mul(-0.5).exp()
        return grid
    
    @torch.no_grad()
    def split_inference(self,heatmap,robustness):
        device = heatmap.device
        p = heatmap[0,0].argmax().item()
        p0 = [p%512,p//512]
        p = heatmap[:,4].argmax().item()
        p4 = [p%512,p//512]

        p = ( heatmap[:,2] * self.post_proc_mask(p0,p4,robustness=robustness, device=device) ).argmax().item()
        p2 = [p%512,p//512]

        p = ( heatmap[:,1] * self.post_proc_mask(p0,p2,robustness=robustness, device=device) ).argmax().item()
        p1 = [p%512,p//512]
        p = ( heatmap[:,3] * self.post_proc_mask(p2,p4,robustness=robustness, device=device) ).argmax().item()
        p3 = [p%512,p//512]
        return p0,p1,p2,p3,p4
    
    @torch.no_grad()
    def robust_inference(self, x, robustness=12):
        assert robustness>0, 'robustness must be greater than 0.'
        heatmap = self.full_heatmap(x[:1])
        
        p0x, p4x = heatmap[0,0].argmax().item()%512, heatmap[0,4].argmax().item()%512 
        p10x,p14x = heatmap[0,10].argmax().item()%512, heatmap[0,14].argmax().item()%512
        half = abs(p4x-p0x) + abs(p14x-p0x) > 2*abs(p10x-p0x) + 2*abs(p14x-p4x)
        
        if half:
            heatmap[:,:10] = heatmap[:,:10] + heatmap[:,10:]    
        """
        else:
            heatmap[:,4] = heatmap[:,4] + heatmap[:,10]
            heatmap[:,10] = heatmap[:,4]

            heatmap[:,9] = heatmap[:,9] + heatmap[:,15]
            heatmap[:,15] = heatmap[:,9]
        """
        P0_5 = self.split_inference(heatmap=heatmap[:,:5],robustness=robustness)
        P5_10 = self.split_inference(heatmap=heatmap[:,5:10],robustness=robustness)
        if half:
            return P0_5 + P5_10
        
        P10_15 = self.split_inference(heatmap=heatmap[:,10:15],robustness=robustness)
        P15_20 = self.split_inference(heatmap=heatmap[:,15:20],robustness=robustness)
        return P0_5 + P5_10 + P10_15 + P15_20
    
    @torch.no_grad()
    def half_inference(self, x, p, shape=None, mode='spline', perspective=1.0, pers_mode = 'D'):
        """
        shape: 出力のサイズ
        perspective: 0.8~1.0程度が良い。理論的根拠なしの遠近補正。実験的機能。
        mode: 横方向に拡大する方法。bilinear,spline
            bilinear: 高速。中品質。
            spline: 低速。高品質？
        pers_mode: 遠近補正の方法
            A: x^t
            B: 1-(1-x)^(1/t) ←Aの逆変換的な
            C: AとBの平均
            D: xA+(1-x)B ←一番良さそう
        """
        assert mode in ['bilinear', 'spline'], "mode must be 'bilinear' or 'spline'. "
        assert pers_mode in ['A', 'B', 'C', 'D'], "pers_mode must be one of ['A', 'B', 'C', 'D']. "
        
        grid = torch.tensor([p]).float().to(x.device).reshape(1,2,5,2)
        grid_x,grid_y = grid[:,:,:,:1], grid[:,:,:,1:]
        grid = torch.cat( [grid_x.add(-(512-1)/2).mul(1/256), grid_y.add(-(384-1)/2).mul(1/192)], 3)
        
        if shape is None:
            ratio = np.sqrt( (p[2][1] - p[7][1])**2 + (p[2][0] - p[7][0])**2 ) / 384
            height,width = ratio*x.shape[2], ratio*x.shape[2]/(2**0.5)
            shape = [int(height)+1, int(width)+1]
        
        grid = grid.permute(0,3,1,2)
        
        if mode=='bilinear':
            grid = F.interpolate(grid, [grid.shape[2],shape[1]], mode=mode,align_corners=True)
        elif mode=='spline':
            tck,u = interpolate.splprep( grid[0, :, 0, :].cpu().numpy(),k=3,s=0) 
            spline1 = torch.tensor( interpolate.splev(np.linspace(0,1,shape[1]),tck) ).float()
            tck,u = interpolate.splprep( grid[0, :, 1, :].cpu().numpy(),k=3,s=0) 
            spline2 = torch.tensor( interpolate.splev(np.linspace(0,1,shape[1]),tck) ).float()
            grid = torch.cat([spline1.reshape(1,2,1,-1),spline2.reshape(1,2,1,-1)],2).to(grid.device)
        
        alpha = torch.linspace(1,0,shape[0],device=grid.device).reshape(1,1,-1,1)
        alpha1 = alpha.pow(perspective)
        alpha2 = 1-( 1 - alpha ).pow(1/perspective)
        
        if pers_mode=='A':
            alpha=alpha1
            
        elif pers_mode=='B':
            alpha = alpha2
        
        elif pers_mode=='C':
            alpha = (alpha1+alpha2)/2
        
        elif pers_mode=='D':
            alpha = alpha*alpha1 + (1-alpha)*alpha2
        
        grid = alpha*grid[:,:,:1,:] + (1-alpha)*grid[:,:,1:,:]
        grid = grid.permute(0,2,3,1)
        return F.grid_sample(x, grid, padding_mode='zeros',mode='bicubic')
        
    @torch.no_grad()
    def forward(self, x, robustness=12, shape=None, mode='spline', perspective=1.0, pers_mode='D', shift_centre=False):
        x_scale = F.interpolate(x.mean(1,True), [384,512],mode='bicubic')
        p = self.robust_inference(x_scale, robustness)

        if shift_centre:
            min_x,min_y = np.min(p,0)
            max_x,max_y = np.max(p,0)
            shift_x = 256 - (max_x + min_x)//2
            shift_y = 192 - (max_y + min_y)//2

            x_scale = x_scale.roll( [shift_y,shift_x], [2,3])
            p = self.robust_inference(x_scale, robustness)
            p = np.array(p) - np.array([[shift_x,shift_y]])
        
        Lpage = self.half_inference(x, p[:10], shape, mode, perspective,pers_mode)
        if len(p)==10:
            return Lpage,None
        Rpage = self.half_inference(x, p[10:], shape, mode, perspective,pers_mode)
        
        return Lpage,Rpage