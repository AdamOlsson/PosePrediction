import torch

def resize(img, dsize, fx=1, fy=1):
    c, h, w = img.shape[-3], img.shape[-2], img.shape[-1]
    no_dims = len(img.shape)

    if dsize == None:
        dsize = (c, int(h*fy), int(w*fx)) if no_dims == 3 else (img.shape[0], c, int(h*fy), int(w*fx))

    new_img = torch.zeros(dsize)
    hs = torch.floor(torch.linspace(0, h-1, h)*fy).long()
    ws = torch.floor(torch.linspace(0, w-1, w)*fx).long()

    mesh_h, mesh_w = torch.meshgrid(hs, ws)
    
    if no_dims == 3:
        new_img[:,mesh_h, mesh_w] = img
    elif no_dims == 4:
        new_img[:,:,mesh_h, mesh_w] = img
    else:
        raise NotImplementedError("Error: Number of dimesion in input image is not 3 or 4.")

    print(new_img[0,:,:])

    # TODO: Interpolate between pixels
    
    pass


if __name__ == "__main__":
    img = torch.ones((3,5,5))*5
    percent = 2.5
    resize(img, None, fx=percent, fy=percent)