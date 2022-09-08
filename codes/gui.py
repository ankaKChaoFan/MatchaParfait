import warnings
warnings.simplefilter('ignore')
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import tkinter.font as font

import os,glob
from model_page import *

#import PIL
from PIL import ImageTk,Image
import torchvision.transforms as T
from torchvision.utils import save_image

#from tqdm import tqdm
import re


def blur(x,kernel=[0.25,0.5,0.25],padding=1, stride=1):
    kernel = torch.tensor(kernel,device= x.device).float().unsqueeze(0)# (1,k)
    kernel = kernel * kernel.T# (k,k)
    kernel = kernel.unsqueeze(0).unsqueeze(0)# (1,1,k,k)
    
    channel = x.shape[1]
    kernel = kernel.repeat(channel,1,1,1)
    p = padding
    return F.conv2d( F.pad(x,[p,p,p,p], mode='replicate'), kernel, stride=stride, groups=channel)

def remove_shadow(x,alpha=0.75):
    x = x.add(1).mul(0.5)
    if alpha==0.0:
        return x
    
    shape = [80,int(80*x.shape[3]/x.shape[2])+1]
    shadow = F.interpolate(x,shape,mode='bicubic')
    
    for i in range(2):#膨張
        shadow = F.max_pool2d(shadow,kernel_size=[3,3],stride=1,padding=1) 
    
    for i in range(3):#縮小
        shadow = -F.max_pool2d(-shadow,kernel_size=[3,3],stride=1,padding=1) 

    for i in range(2):
        shadow = blur(shadow)

    shadow = F.interpolate(shadow,x.shape[2:],mode='bicubic')
    shadow = (1-alpha) + alpha*shadow
    return  (x / shadow ).clamp(0,1)

def askdir():
    path = filedialog.askdirectory(initialdir = './')
    folder_path.set(path)

def search_img_path():
    path  = glob.glob(os.path.join(folder_path.get(), '*.png')) + glob.glob(os.path.join(folder_path.get(), '*.jpg')) 
    path.sort()
    return path

def warning():
    msg = ''
    if len(search_img_path())==0:
        msg += '画像が見つかりません\n'

    if re.search(r'[\\/:*?"<>|]',prefix.get() ) is not None:
        msg+= '接頭辞に使用不可な文字列が含まれています\n'

    try:
        float( remove_value.get() )
    except:
        msg += '影除去レベルが数値ではありません\n'
    
    if not mode_cbox.get() in ['bilinear', 'spline']:
        msg += 'サンプリングモードが不正です\n'

    try:
        float( pers_value.get() )
    except:
        msg += '遠近補正値が数値ではありません\n'

    if not pers_cbox.get() in ['A','B','C','D']:
        msg += '遠近補正モードが不正です\n'
    
    if not auto_shape.get():
        try:
            w = int(shape_width.get() )
            h = int(shape_height.get() )
        except:
            w = 0
            h = 0
        if w<=0 or h<=0:
            msg += '画像サイズが不正です\n'
            
    
    if len(msg)!=0:
        messagebox.showerror('エラー', msg)
        return True
    
    return False

@torch.no_grad()
def preview():
    if warning():
        return
    img_path  = search_img_path()[0]
    img = Image.open(img_path)

    img = trans(img).unsqueeze(0).to(device)
    if monochrome.get():
        img = img.mean(1,True)

    shape = None if auto_shape.get() else [int(shape_height.get() ), int(shape_width.get() )]
    
    Lpage,Rpage = model(
        img,
        shape = shape,
        mode= mode_cbox.get(),
        perspective= float( pers_value.get() ),
        pers_mode= pers_cbox.get(),
        shift_centre = shift_centre.get()
        )
    if open_right.get() and Rpage is not None:
        img = Rpage
    else:
        img = Lpage
    img = remove_shadow(img,alpha=float( remove_value.get() ) )
    org_h, org_w =  img.shape[2:]

    img = F.interpolate(img,scale_factor=800/img.shape[2])
    if monochrome.get():
        img = img.repeat(1,3,1,1)
    
    img  = img[0].mul(255).cpu()
    img = np.transpose(img.numpy(),[1,2,0])
    img = Image.fromarray( img.astype('uint8') )

    w,h = img.size

    window = tk.Tk()
    window.geometry(f'{w}x{h}')
    window.title(f'プレビュー　保存サイズ(幅:{org_w},高さ:{org_h}), プレビューサイズ(幅:{w},高さ:{h})')
    
    canvas = tk.Canvas(window, width= w, height= h)
    canvas.grid(row=0,column=0)
    
    preview_img = ImageTk.PhotoImage( img, master=window )
    canvas.create_image(w//2, h//2, image=preview_img)
    window.mainloop()


@torch.no_grad()
def runnning():
    if warning():
        return

    path = folder_path.get() + '/proc'
    try:
        os.mkdir( path )
    except:
        pass

    img_paths  = search_img_path()
    idx = 0
    shape = None if auto_shape.get() else [int(shape_height.get() ), int(shape_width.get() )]

    for img_path in img_paths:#tqdm( img_paths ):
        img = Image.open(img_path)

        img = trans(img).unsqueeze(0).to(device)
        if monochrome.get():
            img = img.mean(1,True)

        Lpage, Rpage = model(
            img,
            shape=shape,
            mode= mode_cbox.get(),
            perspective= float( pers_value.get() ),
            pers_mode= pers_cbox.get(),
            shift_centre = shift_centre.get()
            )
        if open_right.get() and Rpage is not None:
            Lpage,Rpage = Rpage,Lpage
        
        Lpage = remove_shadow(Lpage,alpha=float( remove_value.get() ) )
        save_image(Lpage, f'{path}/{prefix.get()}{idx:04}.jpg')
        idx +=1
        if Rpage is not None:
            Rpage = remove_shadow(Rpage,alpha=float( remove_value.get() ) )
            save_image(Rpage, f'{path}/{prefix.get()}{idx:04}.jpg')
            idx +=1

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet().to(device)
    model.load_state_dict(torch.load('./checkpoint/best_model.pth'))

    trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5), (0.5), inplace=True),
    ])
    
    root = tk.Tk()
    Meirio = font.Font(root,family='メイリオ',size=11)

    root.title(f'まっちゃぱふぇ　[device={device}]')
    root.resizable(width=False, height=False)

    setting_frame = ttk.Frame(root)
    setting_frame.grid(row=0,column=0)

    running_frame = ttk.Frame(root)
    running_frame.grid(row=1,column=0)

    row = 0
    tk.Label(setting_frame, text="フォルダ",font=Meirio).grid(row=row, column=0)

    folder_path = tk.StringVar()
    ttk.Entry(setting_frame, textvariable=folder_path, width=25).grid(row=row, column=1)
    ttk.Button(setting_frame, text="参照", command=askdir).grid(row=row, column=2)

    row+=1
    tk.Label(setting_frame, text=" オプション　",font=Meirio).grid(row=row, column=0)

    row += 1
    shift_centre = tk.BooleanVar()
    tk.Checkbutton(
        setting_frame,
        text='画像シフト(実験的・低速) ',
        font=Meirio,
        variable=shift_centre, 
        onvalue=True,
        offvalue=False,
        ).grid(row=row, column=1,columnspan=2)

    row += 1
    monochrome = tk.BooleanVar()
    tk.Checkbutton(
        setting_frame,
        text='モノクロで保存',
        font=Meirio,
        variable=monochrome, 
        onvalue=True,
        offvalue=False,
        ).grid(row=row, column=1, pady=5)

    open_right = tk.BooleanVar()
    tk.Checkbutton(
        setting_frame,
        text='右開き',
        font=Meirio,
        variable=open_right, 
        onvalue=True,
        offvalue=False,
        ).grid(row=row, column=2)
    
    
    row += 1
    tk.Label(setting_frame, text="接頭辞",font=Meirio).grid(row=row, column=1)
    prefix = tk.StringVar()
    ttk.Entry(setting_frame, textvariable=prefix, width=8,font=Meirio).grid(row=row, column=2)

    row += 1
    tk.Label(setting_frame, text="影除去レベル",font=Meirio).grid(row=row, column=1)
    remove_value = tk.DoubleVar()
    remove_value.set(0.0)
    tk.Entry(setting_frame,font=Meirio,width=8, textvariable=remove_value ).grid(row=row, column=2)

    row += 1
    tk.Label(setting_frame, text="サンプリングモード",font=Meirio).grid(row=row, column=1, pady=20)
    mode_cbox = ttk.Combobox(
        setting_frame,
        values=['bilinear','spline'],
        font=Meirio,
        width=6)
    mode_cbox.set('spline')
    mode_cbox.grid(row=row,column=2)

    row += 1
    tk.Label(setting_frame, text="遠近補正",font=Meirio).grid(row=row, column=1)
    pers_value = tk.DoubleVar()
    pers_value.set(1.0)
    tk.Entry(setting_frame,font=Meirio,width=8, textvariable=pers_value ).grid(row=row, column=2)

    row += 1
    tk.Label(setting_frame, text="遠近補正モード",font=Meirio).grid(row=row, column=1)
    pers_cbox = ttk.Combobox(
        setting_frame,
        values=['A','B','C','D'],
        font=Meirio,
        width=6)
    pers_cbox.set('D')
    pers_cbox.grid(row=row,column=2)

    row +=1
    setting_shape_frame = ttk.Frame(setting_frame)
    setting_shape_frame.grid(row=row,column=1, columnspan=2, pady=20)
    
    tk.Label(setting_shape_frame, text="幅",font=Meirio).grid(row=1, column=0)
    shape_width = tk.IntVar()
    shape_width.set(1000)
    shape_width_entry = ttk.Entry(setting_shape_frame, textvariable=shape_width, width=6,font=Meirio)
    shape_width_entry.grid(row=1, column=1)
    shape_width_entry.configure(state='readonly')
    tk.Label(setting_shape_frame, text="px ",font=Meirio).grid(row=1, column=2)

    tk.Label(setting_shape_frame, text="高さ",font=Meirio).grid(row=1, column=3)
    shape_height = tk.IntVar()
    shape_height.set(1414)
    shape_height_entry = ttk.Entry(setting_shape_frame, textvariable=shape_height, width=6,font=Meirio)
    shape_height_entry.grid(row=1, column=4)
    shape_height_entry.configure(state='readonly')
    tk.Label(setting_shape_frame, text="px",font=Meirio).grid(row=1, column=5)


    auto_shape = tk.BooleanVar()
    auto_shape.set(True)
    def read_only():
        if auto_shape.get():
            shape_width_entry.configure(state='readonly')
            shape_height_entry.configure(state='readonly')
        else:
            shape_width_entry.configure(state='normal')
            shape_height_entry.configure(state='normal')

    tk.Checkbutton(
        setting_shape_frame,
        text='画像サイズを自動設定',
        font=Meirio,
        variable=auto_shape, 
        onvalue=True,
        offvalue=False,
        command = read_only
        ).grid(row=0, column=0,columnspan=6)

    ttk.Button(running_frame, text="プレビュー", command=preview).grid(row=1, column=0,padx=20,pady=10)
    ttk.Button(running_frame, text="実行", command=runnning).grid(row=1, column=2)
    
    root.mainloop()
