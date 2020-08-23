import torch

def resize(img, dsize, fx=1, fy=1):
    c, h, w = img.shape[-3], img.shape[-2], img.shape[-1]

    if dsize == None:
        # dsize = (int(img.shape[0]*fx), int(img.shape[1]*fy)) if len(img.shape) == 2 else (int(img.shape[1]*fx), int(img.shape[2]*fy))
        dsize = (c, h*fy, w*fx)

    new_img = torch.zeros(dsize)
    hs = torch.linspace(0, h-1, h, dtype=torch.long)*fy
    ws = torch.linspace(0, w-1, w, dtype=torch.long)*fx
    mesh_h, mesh_w = torch.meshgrid(hs, ws)
    
    new_img[:,mesh_h, mesh_w] = 1
    print(new_img[0,:,:])
    # TODO: Interpolate between pixels
    
    pass


if __name__ == "__main__":
    img = torch.ones((3,5,5))
    resize(img, None, fx=2, fy=2)