from codes.utils import *
from codes.model_page import *
#import os,torch
import torchvision
import torch.optim as optim
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
from torchvision.utils import save_image

try:
    import wandb
    use_wandb =True
except:
    use_wandb = False

@torch.no_grad()
def inferance(val_img,test_img):
    #普通の推定
    heatmap = detector_EMA.full_heatmap(val_img, heatmap=False).flatten(2)
    img_ = val_img[0,0].add(1).mul(255/2).unsqueeze(2).repeat(1,1,3).cpu().numpy()
    img_ = img_.astype(np.uint8)
    for i in range(20):
        p = heatmap[0,i].argmax().item()
        p = (p%512,p//512)
        cv2.drawMarker(img_, p, color=col[i], markerType=cv2.MARKER_CROSS if i<10 else cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_8)
    img = img_
    
    #ロバストな推定
    img_ = val_img[0,0].add(1).mul(255*0.5).unsqueeze(2).repeat(1,1,3).cpu().numpy()
    img_ = img_.astype(np.uint8)
    for i,p in enumerate(detector_EMA.robust_inference(val_img,robustness=24) ):
        r = int(255*(19-i)/19)
        b = int(255*i/19 )
        cv2.drawMarker(img_, tuple(p), color=col[i], markerType=cv2.MARKER_CROSS if i<10 else cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_8)
    img = cv2.hconcat([img,img_])
    
    #テスト
    t = F.interpolate(test_img[:,:1],[384,512],mode='bicubic')
    heatmap = detector_EMA.full_heatmap(t, heatmap=False).flatten(2)
    img_ = t[0,0].add(1).mul(255*0.5).unsqueeze(2).repeat(1,1,3).cpu().numpy()
    img_ = img_.astype(np.uint8)
    for i in range(20):
        p = heatmap[0,i].argmax().item()
        p = (p%512,p//512)
        cv2.drawMarker(img_, p, color=col[i], markerType=cv2.MARKER_CROSS if i<10 else cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_8)
    img = cv2.hconcat([img,img_])
    
    img_ = t[0,0].add(1).mul(255*0.5).unsqueeze(2).repeat(1,1,3).cpu().numpy()
    img_ = img_.astype(np.uint8)
    for i,p in enumerate(detector_EMA.robust_inference(t,robustness=24) ):
        r = int(255*(19-i)/19)
        b = int(255*i/19 )
        cv2.drawMarker(img_, tuple(p), color=col[i], markerType=cv2.MARKER_CROSS if i<10 else cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_8)
    img = cv2.hconcat([img,img_])
    
    cv2.imwrite('./gen_img/tmp.jpg', img)
    
    Lpage, Rpage = detector_EMA(test_img)
    save_image(Lpage.add(1).mul(0.5),'./gen_img/Lpage.jpg')
    if Rpage is not None:
        save_image(Rpage.add(1).mul(0.5),'./gen_img/Rpage.jpg')
    return img

@torch.no_grad()
def data_proc(img,keypoint,aug=True):
    keypoint = keypoint[:,:,:,::3]
    if aug:
        img = img + torch.randn_like(img)*0.1*torch.rand(img.shape[0],1,1,1,device=device)
        img = img + torch.randn(img.shape[0],device=device).reshape(-1,1,1,1)*0.5

        noise = F.interpolate(torch.rand_like(F.interpolate(img,scale_factor=0.01)),img.shape[2:],mode='bicubic')
        img = img.add(1).mul(0.5) * ((noise-1)*torch.rand(img.shape[0],1,1,1,device=device) + 1)

        img = img.mul(2).add(-1)
        img = img.clamp(-1,1)

        scale= np.abs( np.random.randn() )*0.15
        scale = 1 - np.clip(scale,0,0.9)

        img = F.interpolate(img,scale_factor=scale,mode='bicubic')
        h1,w1 = int((384-img.shape[2])*np.random.rand() ),int( (512-img.shape[3])*np.random.rand() )
        h2,w2 = (384-img.shape[2])-h1,(512-img.shape[3])-w1
        img = F.pad(img,[w1,w2,h1,h2,],mode='replicate')
        img = img.clamp(-1,1)

        keypoint = torch.cat([
            keypoint[:,:,:,:,:1]*scale+w1,
            keypoint[:,:,:,:,1:]*scale+h1],4)

    label = F.relu( (keypoint-0.5).long() )
    label = torch.cat([
        label[:,:,:,:,:1].clamp(0,511),
        label[:,:,:,:,1:].clamp(0,383)],4)
    label = label.flatten(1,3)
    label = label[:,:,0] + label[:,:,1]*img.shape[3]

    keycoord = keypoint.flatten(1,3)
    return img, keypoint, label, keycoord

    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    
    dataset = Dataset(args.train_dir)
    print('学習画像枚数:',len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    
    val_loader = torch.utils.data.DataLoader(Dataset(args.val_dir), batch_size=args.val_batch_size, shuffle=False, drop_last=True, num_workers=2)
    val_img, val_keypoint = next( iter(val_loader) )
    val_img, val_keypoint = val_img.to(device), val_keypoint.to(device)
    val_img, val_keypoint, val_label, val_keycoord = data_proc(val_img,val_keypoint,False)

    use_wandb = use_wandb and args.use_wandb
    if use_wandb:
        wandb.init(project="自炊")
        
    detector = UNet().to(device)
    detector_EMA = UNet().to(device)
    detector_EMA.eval()

    opt = optim.AdamW(detector.parameters(), lr=args.learning_rate)
    
    losses = []
    col = {}
    for i in range(5):
        col[i] = (int(128 + 127*i/4),0,0)
    for i in range(5):
        col[5+i] = (0,int(128 + 127*i/4),0)
    for i in range(5):
        col[10+i] = (0,0,int(128 + 127*i/4))
    for i in range(5):
        col[15+i] = (int(255 - 127*i/4),0,int(128 + 127*i/4))        
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5), (0.5), inplace=True),
    ])
    test_img = trans(PIL.Image.open('./test_img.jpg') ).unsqueeze(0).to(device)

    with torch.no_grad():
        coord_x = torch.arange(512,device=device).reshape(1,1,512)
        coord_y = torch.arange(384,device=device).reshape(1,1,384)
    
    iteration=0
    pbar = tqdm(total= args.max_iterations, dynamic_ncols= True)
    while True:
        for img,keypoint in loader:
            img, keypoint = img.to(device), keypoint.to(device)
            img,keypoint,label,keycoord = data_proc(img,keypoint)
            
            heatmap = detector.full_heatmap(img,False).flatten(2)

            loss_CE = 0
            for i in range(20):
                loss_CE += F.cross_entropy(heatmap[:,i],label[:,i] )/20

            heatmap = detector.full_heatmap(img)
            keycoord_x = (heatmap.sum(2)*coord_x ).sum(2,True)
            keycoord_y = (heatmap.sum(3)*coord_y ).sum(2,True)
            detect_keycoord = torch.cat([keycoord_x,keycoord_y],2)

            loss_coord = detect_keycoord.add(-keycoord).pow(2).mean()

            loss = loss_CE
            loss += loss_coord/400
            opt.zero_grad()
            loss.backward()
            opt.step()
            param_ema(detector_EMA,detector)   
            if use_wandb:
                wandb.log( {
                    'train_loss':loss.item(),
                    'train_loss_CE':loss_CE.item(),
                    'train_loss_coord':loss_coord.item(),
                           }, commit=False )
                
            iteration += 1
            pbar.update(1)
            if iteration%100==0:
                img_=inferance(val_img[:1],test_img)
                if use_wandb:
                    with torch.no_grad():
                        heatmap = detector_EMA.full_heatmap(val_img,False).flatten(2)

                        loss_CE = 0
                        for i in range(20):
                            loss_CE += F.cross_entropy(heatmap[:,i],val_label[:,i] )/20

                        heatmap = detector_EMA.full_heatmap(val_img)
                        keycoord_x = (heatmap.sum(2)*coord_x ).sum(2,True)
                        keycoord_y = (heatmap.sum(3)*coord_y ).sum(2,True)
                        detect_keycoord = torch.cat([keycoord_x,keycoord_y],2)
                        
                        loss_coord = detect_keycoord.add(-val_keycoord).pow(2).mean()

                        loss = loss_CE
                        loss += loss_coord/400

                        wandb.log( {
                            'val_loss':loss.item(),
                            'val_loss_CE':loss_CE.item(),
                            'val_loss_coord':loss_coord.item(),
                                   }, commit=False )
            if use_wandb:
                wandb.log({})
                    
                    
                    
                    
            if iteration==args.max_iterations:
                break
        if iteration==args.max_iterations:
                break
    pbar.close()
    torch.save(detector_EMA.to('cpu').state_dict(),"./checkpoint/model.pth" )