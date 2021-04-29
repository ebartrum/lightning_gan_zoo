import torch
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer import FoVOrthographicCameras

def convert_cam_pred(cam_pred, device):
    scale, tx, ty, quats = cam_pred[:,0], cam_pred[:,1],\
            cam_pred[:,2], cam_pred[:,3:]
    
    R = torch.eye(3).unsqueeze(0)
    R[:,1]*=-1
    R[:,0]*=-1
    
    T = torch.stack([-tx,-ty,5*torch.ones_like(tx)],1)
    R = torch.cat(len(tx)*[R]).to(device)
    newR = quaternion_to_matrix(quats)
    R = torch.inverse(newR)@R
    
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    return cameras, scale
