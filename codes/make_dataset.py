import cv2
import numpy as np
import os,glob
import copy

import argparse

import tqdm
import json

class Book:
    def __init__(self,canvas_height, canvas_width,frac_num=5):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.frac_num = frac_num
    
    def book_condition(self):
        self.bezier_sz = np.clip(0.3*np.abs(np.random.randn()),0,0.9)#np.random.uniform(0,0.5)
        self.up_scale = 1 - 0.25*self.bezier_sz#ページ上部の遠近感
        
        self.page_width = self.canvas_width*0.5 * np.random.uniform(0.8,0.9)

        self.down_centre_x = (np.random.randn()*0.05+0.5) *self.canvas_width
        self.down_centre_y = (np.random.randn()*0.05+0.85 )*self.canvas_height

        self.up_centre_x = self.down_centre_x + (np.random.randn()*0.025)*self.canvas_width
        self.up_centre_y = (np.random.randn()*0.04+0.15 )*self.canvas_height

    def bezier_function(self,a,t):
        return a[0]*(1-t)**3 + 3*a[1]*((1-t)**2)*t + 3*a[2]*(1-t)*t**2 + a[3]*t**3
    
    def bezier(self):
        ax = np.array([0,np.random.uniform(0.0,0.4),np.random.uniform(0.5,0.8),1])
        az = np.array([0,np.random.uniform(0.1,0.6),np.random.uniform(-0.2,0.3),0])
        t = np.linspace(0,1,1+512)
        px = self.bezier_function(ax,t)
        pz = self.bezier_function(az,t)

        l = np.sqrt( (px[1:]-px[:-1])**2 + (pz[1:]-pz[:-1])**2 )
        px = px/sum(l)
        pz = pz/sum(l)

        l = l/sum(l)
        l = np.cumsum(l)
        return px,pz,l
    
    def render_half_mask(self,px,pz,l):
        up_lattice = []
        down_lattice = []
        record_idx = []
        for i in np.linspace(0,1,self.frac_num)[1:-1]:
            record_idx.append( np.where(l>=i)[0][0] )
        record_idx = [0] + record_idx + [len(l)]

        for i in range(len(px) ):
            start = ( self.down_centre_x + px[i]*self.page_width, self.down_centre_y - pz[i]*self.page_width) 
            end   = ( self.up_centre_x + px[i]*self.page_width*self.up_scale, self.up_centre_y - pz[i]*self.page_width) 
            cv2.line(self.mask, (int(start[0]),int(start[1])), (int(end[0]),int(end[1])), 255, thickness=2)
            if i in record_idx:
                up_lattice.append(end)
                down_lattice.append(start)
        return [up_lattice,down_lattice]
    
    def render_mask(self):
        self.mask = np.full((self.canvas_height, self.canvas_width), 0, dtype=np.uint8)

        self.book_condition()
        
        #左ページをレンダリング    
        px,pz,l = self.bezier()
        px = -np.flip(px)
        pz = np.flip(pz)
        l = 1 - np.flip(l)
        pz = pz*self.bezier_sz

        lattice_L = self.render_half_mask(px,pz,l,)
        
        #右ページ
        px,pz,l = self.bezier()
        pz = pz*self.bezier_sz
        
        lattice_R = self.render_half_mask(px,pz,l,)
        self.lattice = np.concatenate([[lattice_L],[lattice_R]],0)

        self.canvas = copy.deepcopy(self.mask)
    
    def make_mul(self,src):
        h,w = src.shape
        mul = np.linspace(0,10,w).reshape(1,-1)
        mul = np.repeat(mul,h, axis=0)
        mul = np.clip(mul,0,1)*0.7+0.3
        return mul**0.5
    
    def render_half_canvas(self,lattice,src, L_page=False):
        src = cv2.copyMakeBorder(src, 0, 0, 0, int(src.shape[1]*0.05), borderType=cv2.BORDER_CONSTANT,value=255)
        ambient_occlusion = self.make_mul(src)
        if L_page is True:
            ambient_occlusion = np.flip( ambient_occlusion ) 
        src = (src* ambient_occlusion).astype('uint8')
        
        flags = cv2.INTER_LANCZOS4

        src_h,src_w = src.shape
        src_div = np.linspace(0,1,self.frac_num)*(src_w-1)

        pad = 4
        src = cv2.copyMakeBorder(src, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT, value=0)

        for idx in range(self.frac_num-1):
            src_p = np.array([ [src_div[idx]+pad,pad], [src_div[idx+1]+pad,pad], [src_div[idx+1]+pad,src_h-1+pad], [src_div[idx]+pad,src_h-1+pad] ], dtype=np.float32)
            trg_p = np.array([ lattice[0,idx], lattice[0,idx+1], lattice[1,idx+1], lattice[1,idx]  ], dtype=np.float32)

            alpha = np.full((src_h + 2*pad, src_w//(self.frac_num-1)+2*pad ), 255, dtype=np.uint8) 

            alpha = cv2.copyMakeBorder(alpha, 0, 0, src_w//(self.frac_num-1)*idx, self.canvas_width*2-src_w//(self.frac_num-1)*(self.frac_num-idx-2), borderType=cv2.BORDER_CONSTANT, value=0)

            mat = cv2.getPerspectiveTransform(src_p, trg_p)
            
            p_img = cv2.warpPerspective(src, mat, (self.canvas_width, self.canvas_height),flags=flags )
            alpha = cv2.warpPerspective(alpha, mat, (self.canvas_width, self.canvas_height),flags=flags )
            alpha = alpha.astype(np.float32)/255
            self.canvas = (1-alpha)*self.canvas + alpha * p_img
        self.canvas = self.canvas * (self.mask/255)
        self.canvas = np.clip( self.canvas,0,255)
        self.canvas = self.canvas.astype('uint8')
    
    def render_book_side(self,cof=0.01,color=None):
        color = color if color is not None else np.random.randint(224,256)

        h_size = int(self.canvas_height*cof*self.bezier_sz)
        w_size = int(self.canvas_width*cof*self.bezier_sz)//2*2

        side_mask = cv2.resize( self.mask,
                               (self.canvas_width+w_size, self.canvas_height+h_size)
                               )[:self.canvas_height, w_size//2: self.canvas_width+w_size//2]/255
        side_mask = side_mask * (1 - self.mask/255)

        self.canvas = self.canvas*(self.mask/255) + (1-self.mask/255)*side_mask*color
        self.canvas = self.canvas.astype(np.uint8)

        self.mask = (self.mask/255 + side_mask)*255
        self.mask = self.mask.astype(np.uint8)
    
    def render_total_book(self):
        M = np.float32([[1,0,int(np.random.uniform(-self.canvas_width*0.05,self.canvas_width*0.05))],[0,1,0*int(np.random.uniform(-self.canvas_height*0.05,self.canvas_height*0.05))]])
        noise = cv2.warpAffine(self.canvas,M,(self.canvas_width,self.canvas_height) )*1.0
        noise = noise*(0.9+0.1*np.random.rand())
        noise = noise.astype(np.uint8)
        noise_mask = cv2.warpAffine(self.mask,M,(self.canvas_width,self.canvas_height) )

        if np.random.rand()<0.5:
            M = np.float32([[1,0,int(np.random.uniform(-self.canvas_width,self.canvas_width))],[0,1,int(np.random.uniform(-self.canvas_height,self.canvas_height))]])
            noise2 = cv2.warpAffine(self.canvas,M,(self.canvas_width,self.canvas_height) )*1.0
            noise2 = noise2*np.random.rand()
            noise2 = noise2.astype(np.uint8)
            noise_mask2 = cv2.warpAffine(self.mask,M,(self.canvas_width,self.canvas_height) )

            noise = (noise_mask/255)*noise + (1-noise_mask/255)*noise2
            noise = noise.astype(np.uint8)
            noise_mask = (noise_mask/255)*(noise_mask) + (1-noise_mask/255)*(noise_mask2)
            noise_mask = noise_mask.astype(np.uint8)

        for i in range(np.random.randint(5,10)):
            self.render_book_side(color=255)
        
       
        self.canvas = self.canvas * (self.mask/255) + noise*(1-self.mask/255)
        self.mask = self.mask/255 +  (1-self.mask/255)*noise_mask/255
        self.mask = (self.mask*255).astype(np.uint8)
        self.canvas = self.canvas.astype(np.uint8)

        self.render_book_side(0.05,np.random.randint(64,255))
    
    def render_floor(self, src, data_aug=False):
        flags = cv2.INTER_LANCZOS4
        src_h,src_w = src.shape

        src_p = np.array([ [0,0], [src_w-1,0], [src_w-1,src_h-1], [0,src_h-1] ], dtype=np.float32)
        w = int( self.canvas_width *(1/self.up_scale - 1)/2 )
        trg_p = np.array([ [0,0], [self.canvas_width-1,0], [self.canvas_width-1+w,self.canvas_height-1], [-w, self.canvas_height-1] ], dtype=np.float32)

        mat = cv2.getPerspectiveTransform(src_p, trg_p)
        
        p_img = cv2.warpPerspective(src, mat, (self.canvas_width, self.canvas_height),flags=flags )

        self.canvas = (self.mask/255)*self.canvas + (1-self.mask/255) * p_img
        self.canvas = self.canvas.astype('uint8')

        if data_aug:
            for i in range(np.random.randint(5)):
                self.canvas = cv2.ellipse(self.canvas,
                                        (
                                            ( np.random.randint(self.canvas_width), np.random.randint(self.canvas_height) ),
                                            ( np.random.randint(int(self.canvas_width*0.4)), np.random.randint(int(self.canvas_height*0.4)) ),
                                            np.random.randint(360)
                                            ) ,
                                            (np.random.randint(255)), thickness=-1, lineType=cv2.LINE_8)
            for i in range(np.random.randint(5)):
                rect = cv2.rectangle(np.full((self.canvas_height, self.canvas_width), 255, dtype=np.uint8),
                                            (np.random.randint(self.canvas_width), np.random.randint(self.canvas_height)), 
                                            (np.random.randint(self.canvas_width), np.random.randint(self.canvas_height)), 
                                            np.random.randint(255), 
                                            thickness=-1 if np.random.rand()<0.5 else np.random.randint(1,5),
                                            lineType=cv2.LINE_8, shift=0)
                self.canvas = self.canvas * (rect/255)
                self.canvas = self.canvas.astype(np.uint8)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--data_num", type=int, default=5000)
    parser.add_argument("--data_aug", action="store_true")
    args = parser.parse_args()

    data_dir = glob.glob(os.path.join(args.img_dir, '*.jpg')) + glob.glob(os.path.join(args.img_dir, '*.png'))
    try:
        os.mkdir( args.save_dir )
    except:
        pass
    try:
        os.mkdir( f'{args.save_dir}/Image' )
    except:
        pass
    print(args.img_dir,data_dir)
    book = Book(384*2,512*2,13)

    image_dir = []
    key_point = []

    for i in tqdm.tqdm( range(args.data_num) ):
        book.render_mask()

        src = cv2.imread( np.random.choice(data_dir) , cv2.IMREAD_GRAYSCALE)
        book.render_half_canvas(book.lattice[0], src, L_page=True)
        src = cv2.imread( np.random.choice(data_dir) , cv2.IMREAD_GRAYSCALE)
        book.render_half_canvas(book.lattice[1], src)

        book.render_total_book()

        cof = np.random.rand()#*0.5+0.5
        src = np.random.rand(50,10)*(1-cof)*0.05+cof
        src = src *255
        book.render_floor(src.astype(np.uint8),data_aug=args.data_aug)

        img = copy.deepcopy( book.canvas ).reshape(book.canvas_height, book.canvas_width, 1)
        img = cv2.resize(img,(512,384),interpolation=cv2.INTER_LANCZOS4)
        
        image_dir.append( f'Image/{i:05d}.jpg' )
        key_point.append( (book.lattice/2).tolist() )
        
        cv2.imwrite(f'{args.save_dir}/Image/{i:05d}.jpg', img)

    data = {'image_dir':image_dir, 'key_point':key_point}
    with open(f'{args.save_dir}/dataset.json','w') as f:
        json.dump(data,f)
