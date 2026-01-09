import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py as h5
from utils import mask_generator, get_perf_metrics, get_SMS_file_list
from SPSG.spsg_utils import get_grappa_kernel, fill_grappa_kspace
from Unrolled_Net.data_consistency import Data_consistency
import sys

np.set_printoptions(precision=3)

dc = Data_consistency()

def get_spsg_file():
    data_path = Path("/home/range6-raid17/merve-data/fMRI_dat2mat/RawData/")

    file_list = []
    for sub in range(5,14):
        for r in range(1,11):
            file_list.append(data_path / f"subject_{sub:02d}_run_{r:02d}.mat")
    
    return file_list
 
files = get_spsg_file()
files2 = get_SMS_file_list()
file_idx = 0
with h5.File(files[file_idx], 'r', swmr=True) as f:
    acs = f["acs_for_spsg"][:]
    data = f["MB"][0]
    fov_PE = f["param"]["fov_PE"][:]

with h5.File(files2[file_idx], 'r', swmr=True) as f:
    coils = f["sense_maps_all_small"][:]

acs = acs["real"] + 1j* acs["imag"]
data = data['real'] + 1j* data["imag"]
coils = coils['real'] + 1j* coils["imag"]

grappa_type = sys.argv[1]

#Define size parameters
n_grps, ns, nc, nx, ny = acs.shape
cx, cy = 30,30#100, 120
center = [55,64]
kx, ky = 7,7
ks = [kx, ky]
a,b = cx//2, cy//2
p_fourier_pad = 18
nx = nx - p_fourier_pad
l = 1e-10
sl_grp = 10
r = 2

def cus_ifft(img, dims):
    return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(img, axes=dims), axes=dims, norm = 'ortho'), axes=dims)

#Crop ACS measurement to desired size
acs = acs[sl_grp,:,:,p_fourier_pad:,:]
acs = acs[:,:,center[0]-a:center[0]+a,center[1]-b:center[1]+b]
data = data[sl_grp,:,p_fourier_pad:,:]
coils = coils[sl_grp]

def recon(grappa_type, data, r=2, acs = None, coils = None):
    mask = mask_generator(nx,ny,r)
    if grappa_type in ["spsg", "sg"]:
        kernels = get_grappa_kernel(acs, mask, ks, lmbd = l, grappa_type = grappa_type)
        data = fill_grappa_kspace(data, kernels, ks, ns)
        data = data/np.max(np.abs(data))
        #img_data = cus_ifft(data.squeeze(), dims = (1,2))
        #recon = np.sqrt(np.sum(img_data**2, axis=3))
        #return recon.squeeze()
        
        data = data.transpose(0,3,1,2)
        recon = dc.single_slice_EH(torch.from_numpy(data), torch.from_numpy(coils))
    elif grappa_type == "cg-sense":
        data = data/np.max(np.abs(data))
        zerofilled= dc.EH(torch.from_numpy(data), torch.from_numpy(coils), torch.from_numpy(mask))
        recon = dc(zerofilled.unsqueeze(0), torch.from_numpy(coils), torch.from_numpy(mask), CG_iter = 25)
    elif grappa_type == "zerofilled":
        data = data/np.max(np.abs(data))
        recon = dc.EH(torch.from_numpy(data), torch.from_numpy(coils), torch.from_numpy(mask))
    
    return recon.numpy().squeeze()


#Pad ACS and data
#Get R = 2 results
r_list = [2,4,6]

ref_recon = np.zeros((ns,nx,ny),dtype=float)#recon("spsg", data, r = 2, acs = acs, coils=coils)
fig, ax = plt.subplots(3,5, figsize = (15,3*len(r_list)))

for j, r in enumerate(r_list):
    img_data = recon(grappa_type, data, r=r, acs=acs, coils=coils)
    if r==2: ref_recon = img_data
    for i in range(5):
        PSNR, SSIM = get_perf_metrics(img_data[i], ref_recon[i])
        img_plot = np.roll(img_data[i], fov_PE[i], axis = 0)

        ax[j, i].imshow(np.abs(img_plot), cmap="gray")
        if r > 2:
            ax[j,i].text(0.02, 0.02,
            f'PSNR: {PSNR:.3f}\nSSIM: {SSIM:.3f}',
            color='white', fontsize=9, fontweight='bold',
            ha='left', va='bottom', transform=ax[j,i].transAxes)

        if i == 0: ax[j,i].set_ylabel(f"R = {r}", fontsize = 18)

for ax in fig.get_axes():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
fig.suptitle(f"{grappa_type.upper()} Reconstruction", fontsize = 25)
fig.tight_layout()
fig.savefig(f"temp_images/{grappa_type}_{sys.argv[2]}.png")


'''

#Get unique patches from the mask
#mask = np.abs(data_pad[0,0,:,:])>0
mask_patches = np.lib.stride_tricks.sliding_window_view(mask, ks) #[nx, ny, kx, kx] 
mask_patches = mask_patches.transpose(0,1,3,2)
mask_patches = mask_patches.reshape(-1,kn) #[nx*ny, ky*kx] 
P, p_idx = np.unique(mask_patches, axis=0, return_inverse=True)
#Get valid patches (all that have non-zero elements)
validP = np.argwhere(~np.all(P, axis=1)).squeeze()

#Select slice group (Just using 0 for now)
sl_grp = 0
acs = acs[sl_grp,...]
data_pad = data_pad[sl_grp,...]
#Get patches from ACS and data to be filled
acs_patches = np.lib.stride_tricks.sliding_window_view(acs, ks, axis=(-1,-2))
acs_patches = acs_patches.reshape(ns,nc,cx*cy,kn) #[ns, nc, cy*cx, ky*kx] 
acs_patches = acs_patches.transpose(0,2,3,1) #[ns, cy*cx, ky*kx, nc] 

data_patches = np.lib.stride_tricks.sliding_window_view(data_pad, ks, axis=(-1,-2)) #[nc, nx,ny, kx,ky] 
data_patches = data_patches.reshape(nc, nx*ny, kn) #[nc, nx*ny, ky*kx] 
data_patches = data_patches.transpose(1,2,0) #[nx*ny, ky*kx, nc] 

grappa_data = np.zeros((ns,nx*ny,nc), dtype=data_patches.dtype)

def cus_ifft(img, dims):
    return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(img, axes=dims), axes=dims, norm = 'ortho'), axes=dims)

if grappa_type == "sg":
    acs_scr = np.sum(acs_patches, axis = 0) #[cy*cx, ky*kx, nc] 
elif grappa_type == "spsg":
    acs_scr = acs_patches

#Get kernels
for z in range(ns):
    print(z)
    for i in validP:
        p = P[i]
        acs_trg = acs_patches[z]
        trg = acs_trg[:,kc,:]
        nonzero_idx = np.nonzero(p)[0]

        if grappa_type == "spsg":
            src = acs_scr[:,:,nonzero_idx,:]
            src = src.reshape(ns, cy*cx, len(nonzero_idx)*nc)
            trg = trg[np.newaxis,...]
            trg = np.tile(trg,(ns,1,1))
            zero_slices = np.ones(ns,dtype=int)
            zero_slices[z] = 0
            trg[zero_slices==1,:,:] = 0
            src = src.reshape(ns*cx*cy,len(nonzero_idx)*nc)
            trg = trg.reshape(ns*cx*cy,nc)
        elif grappa_type == "sg":
            src = acs_scr[:,nonzero_idx,:]
            src = src.reshape(src.shape[0], -1)

        AhA = np.conj(src.T) @ src
        Ahb = np.conj(src.T) @ trg

        n = AhA.shape[0]
        x = np.linalg.solve(AhA + np.eye(n)*l, Ahb).T
        patch_indices = (p_idx == i)
        n_idx = np.count_nonzero(patch_indices)
        y = data_patches[:, nonzero_idx, :]
        y = y[patch_indices, :, :]
        
        y = y.reshape(n_idx,-1)
        y = y[:,np.newaxis,:] * x
        grappa_data[z,patch_indices, :] = np.sum(y, axis=-1)
         
       

grappa_data = grappa_data.reshape(ns, 110, 128, nc)
test2d = grappa_data[0,:,:,0].squeeze()
quick_plot(np.log10(np.abs(test2d)), "kspace.png")

for i in range(5):
    img_data = cus_ifft(grappa_data[i,:,:,:].squeeze(),dims = (0,1))
    img_data = np.sqrt(np.sum(img_data**2, axis=2))
    img_data = np.roll(img_data, fov_PE[i], axis = 0)
    quick_plot(np.abs(img_data), f"img{i}_{grappa_type}.png")

'''