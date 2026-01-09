from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from Unrolled_Net.data_consistency import Data_consistency
from Unrolled_Net.UnrolledNet import UnrolledNet
from configs import Config
import h5py as h5
import hdf5storage
from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import wandb
import io
from SPSG.spsg_utils import get_grappa_kernel, fill_grappa_kspace


def get_perf_metrics(x_ref, x):
    """
    Get PSNR and SSIM for input images
    """
    if(torch.is_tensor(x_ref)): 
        x_ref = x_ref.cpu().detach().numpy()
    if(torch.is_tensor(x)): 
        x = x.cpu().detach().numpy()

    x_ref = x_ref.squeeze()
    x = x.squeeze()
    psnr_val = psnr(np.abs(x_ref), np.abs(x), data_range=np.abs(x_ref).max() - np.abs(x_ref).min())
    ssim_val = ssim(np.abs(x_ref), np.abs(x), data_range=np.abs(x_ref).max() - np.abs(x_ref).min())
    return [psnr_val, ssim_val]


#L1 L2 norm loss
def L1_L2_norm(output, ref):
    L1 = torch.norm(ref-output, p=1)/torch.norm(ref, p=1)
    L2 = torch.norm(ref-output, p=2)/torch.norm(ref, p=2)
    return L1 + L2


def get_SMS_file_list(category="train", path = None):

    train_path = Path("/home/daedalus1-raid1/omer-data/fMRI_FOV3/training_corrected/")
    test_path = Path("/home/daedalus1-raid1/omer-data/fMRI_FOV3/testing_corrected/")
    grappa_path = Path("/home/range6-raid17/merve-data/fMRI_dat2mat/RawData/")
    rng = np.random.default_rng(0)
    subj_random_select = rng.permutation(range(5,9))
    run_random_select = rng.permutation(range(1,11))

    file_list = []
    if category in ["train", "val"]:
        for sub in subj_random_select:
            for r in run_random_select:
                path1 = train_path / f"Csubject_{sub}_run_{r}.mat"
                path2 = grappa_path / f"subject_{sub:02d}_run_{r:02d}.mat"
                file_list.append((path1, path2))
    elif category == "test":
        for r in range(1,11):
            for subj in [9,10,11,13]:
                slice_list = []
                for s in range(1,16):
                    path1 = test_path / f"Csubject_{subj}_slice_{s}_run_{r}.mat"
                    slice_list.append(path1)
                path2 = grappa_path / f"subject_{subj:02d}_run_{r:02d}.mat"
                file_list.append((path1, path2))


    return file_list


class SMSFileCache:
    def __init__(self, n_sl_grps, file_list):
        self.cache = {}   # per worker
        self.n_sl_grps = n_sl_grps
        self.file_list = file_list

    def get(self, file_idx):
        if file_idx not in self.cache:
            path = self.path_list[file_idx]
            self.cache[file_idx] = h5.File(path, 'r', swmr=True)[:self.n_sl_grps][:]
        return self.cache[path]

    def __del__(self):
        for f in self.cache.values():
            try:
                f.close()
            except:
                pass

def get_SMS_slice(category, file_list, idx, acc_rate, sms_cache):
    
    if category == "val":
        idx += Config().parse().n_train

    slice_group_idx = idx%16
    file_idx = idx//16
    
    path = file_list[file_idx]
    f = sms_cache.get(path)

    if category in ["train", "val"]:
        kspace = f[f'kspace_all_r{acc_rate}'][slice_group_idx]   
    else:
        kspace = f[f'kspace_all_r{acc_rate}_small'][0]

    coils = f[f'sense_maps_all_small'][slice_group_idx]

    kspace = kspace['real'] + 1j * kspace['imag']
    coils = coils['real'] + 1j * coils['imag']

    return kspace, coils

def gauss_dist(mask, samp_rat, center):
    s = 86
    h,w = mask.shape
    c2, c1 = center
    locs = mask.abs()==1

    x1 = np.linspace(0, w, w)
    x2 = np.linspace(0, h, h) 
    y1 = np.exp(-(x1 - c1)**2/s**2)[np.newaxis, :]
    y2 = np.exp(-(x2 - c2)**2/s**2)[np.newaxis, :]

    d = y2.T @ y1
    d = torch.from_numpy(d)
    d = d/torch.mean(d[locs]) * (1-samp_rat)
    d = 1-d
    return d*mask.abs()

# =======================================================
#                   Dataset Class
# =======================================================

class kspace_SMS_dataset(Dataset):

    def __init__(self, category, device):
        self.conf = Config().parse()
        self.device = device
        self.category = category
        self.temp_ksp = None
        self.temp_coils = None
        self.n_slice_grps = 16
        self.gauss_mask = None
        self.cache = {}   # per worker
        self.ksp_str = f'kspace_all_r2'
        nx,ny = 110,128

        # Determine number of slices
        if category == "train":
            self.num_slices = self.conf.n_train
            self.ordered = self.conf.ordered
        elif category == "val":
            self.conf.n_masks = 1
            self.num_slices = self.conf.n_val
            self.ordered = True 
        elif category == "test":
            self.conf.n_masks = 1
            self.n_slice_grps = 1
            self.num_slices = self.conf.n_test
            self.ksp_str = f'kspace_all_r2_small'
            self.ordered = True 

        self.file_list = get_SMS_file_list(category)

        self.omega_mask_r4 = torch.from_numpy(mask_generator(110,128,4,32)).to(device).to(torch.complex64)
        self.omega_mask_r2 = torch.from_numpy(mask_generator(110,128,2,32)).to(device).to(torch.complex64)

    # ---------------------------------------------------
    # Mask generation helper
    # ---------------------------------------------------
    def _create_disjoint_masks(self, kspace, seed_val = 1):
        
        Nc, Nx, Ny = self.omega_mask_r4.shape

        #Find kspace maximum
        flat_idx = torch.argmax(kspace[0,:,:].squeeze().abs()) 
        row_idx = flat_idx.item() // Ny
        col_idx = flat_idx.item() % Ny
        temp_mask = self.omega_mask_r4[0,:,:].detach().clone().cpu()
        c = self.conf.center_size
        temp_mask[row_idx-c:row_idx+c, col_idx-c:col_idx+c] = 0

        if self.conf.gauss:
            if self.gauss_mask is None:
                self.gauss_mask = gauss_dist(temp_mask, self.conf.lambda_ratio, (row_idx, col_idx))
            temp_mask = self.gauss_mask
        else:
            temp_mask = temp_mask.abs() * self.conf.lambda_ratio

        temp_mask[temp_mask>1] = 1

        #Set seed and sample
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed_val)
        lambda_mask = torch.bernoulli(temp_mask).to(torch.complex64)
        torch.set_rng_state(rng_state)

        #Center of lambda mask is False
        lambda_mask = lambda_mask.unsqueeze(0).repeat(Nc, 1, 1).to(self.device)

        theta_mask = self.omega_mask_r4 - lambda_mask
        
        return lambda_mask, theta_mask.to(self.device)

    # ---------------------------------------------------
    def __len__(self):
        return self.num_slices * self.conf.n_masks
    
    def read_grappa_mat(self, f):
        kspace_list = []
        for i in range(2):
            full_kspace = f[f"R{i}"]
            full_kspace = full_kspace['real'][:] + 1j* full_kspace["imag"][:]
            full_kspace = full_kspace.transpose(3,2,1,0)
            kspace_list.append(torch.from_numpy(full_kspace).to(self.device).to(torch.complex64))
        return kspace_list

    def get_grappa_data(self, sampled_kspace, sl_grp_idx, grappa_path, save_path):
        if save_path.exists():
            data_dict = h5.File(save_path)
            return self.read_grappa_mat(data_dict)
        else:
            f = h5.File(grappa_path)
            acs = f["acs_for_spsg"][:]
            acs = acs["real"] + 1j* acs["imag"]
            ns, nx, ny, cx, cy = 5, 110, 128, 100, 120
            kx, ky = 7,7
            ks = [kx, ky]
            a,b = cx//2, cy//2
            p_fourier_pad = 18
            l = 1e-10
            
            #Crop ACS measurement to desired size
            acs = acs[15,:,:,p_fourier_pad:,:]
            acs = acs[:,:,nx//2-a:nx//2+a,ny//2-b:ny//2+b]
            mask_r2 = self.omega_mask_r2.cpu().numpy()[0,:,:]
            mask_r4 = self.omega_mask_r4.cpu().numpy()[0,:,:]
             
            kernels_r2 = get_grappa_kernel(acs, mask_r2, ks, lmbd = l)
            kernels_r4 = get_grappa_kernel(acs, mask_r4, ks, lmbd = l)
            mask_list = [mask_r2, mask_r4]
            data_dict = {}
            kspace_list = []
            for i, kernel in enumerate([kernels_r2, kernels_r4]):
                full_kspace = fill_grappa_kspace(sampled_kspace*mask_list[i], kernel, ks, ns)
                full_kspace = full_kspace/np.max(np.abs(full_kspace))
                full_kspace = full_kspace.transpose(0,3,1,2)
                kspace_list.append(torch.from_numpy(full_kspace).to(self.device).to(torch.complex64))
                data_dict[f"R{i}"] = {"real": full_kspace.real, "imag": full_kspace.imag}
            hdf5storage.savemat(save_path, data_dict, matlab_compatible=True, store_python_metadata=True)
            return kspace_list


    # ---------------------------------------------------
    def __getitem__(self, idx):

        if self.category == "val":
            n_train = self.conf.n_train
            idx += n_train

        sl_idx = idx // self.conf.n_masks #Slice index
        sl_grp_idx = sl_idx % self.n_slice_grps #Slice group index
        file_idx = sl_idx//self.n_slice_grps #File index
        tot_reads = self.n_slice_grps*self.conf.n_masks #Total reads per file
        cache_idx = file_idx*int(not self.ordered)
        read_condition = (self.category == "test") or \
                         (self.ordered and idx%tot_reads==0) or \
                         (not self.ordered and cache_idx not in self.cache.keys())

        if read_condition:
            file_path = self.file_list[file_idx][0]
            f = h5.File(file_path)
            self.cache[cache_idx] = {}
            self.cache[cache_idx]['ksp'] = f[self.ksp_str][:self.n_slice_grps][:] 
            self.cache[cache_idx]['coils'] = f['sense_maps_all_small'][:self.n_slice_grps][:]

        kspace = self.cache[cache_idx]['ksp'][sl_grp_idx]
        coils =  self.cache[cache_idx]['coils'][sl_grp_idx]

        kspace = kspace['real'] + 1j * kspace['imag']
        coils = coils['real'] + 1j * coils['imag']
        
        if self.category == "test":
            #Run grappa recons for comparison
            grappa_path = Path(f"SPSG/grappa_ksp/{str(file_path)[60:]}")
            grappa_kspace = self.get_grappa_data(kspace, sl_grp_idx, self.file_list[idx][1], grappa_path)

        # Convert to tensor
        kspace = torch.from_numpy(kspace).to(self.device).to(torch.complex64)
        coils  = torch.from_numpy(coils).to(self.device).to(torch.complex64)
        kspace = kspace / kspace.abs().max() #normalize

        if self.category in ['train', 'val']:
            lambda_mask, theta_mask = self._create_disjoint_masks(kspace, idx)
            return kspace, coils, theta_mask, lambda_mask, idx
        else:
            return kspace, coils, grappa_kspace


def validate_model(model, dataloader):

    metrics_list = np.zeros(len(dataloader))
    dc = Data_consistency()
    test_bar = tqdm(dataloader, desc=f"[Validation]")
    i = 0
    with torch.no_grad():
        for ksp, coils, theta_mask, lambda_mask, _ in test_bar:

            # Get undersampled measurements
            y_theta  = ksp*theta_mask
            y_lambda = ksp*lambda_mask

            # Forward pass
            output, _ = model(y_theta, coils, theta_mask)
            enc_output = dc.E(output, coils, lambda_mask)

            loss = L1_L2_norm(enc_output, y_lambda)

            metrics_list[i] = loss
            test_bar.set_postfix(loss=loss.item())
            i+=1

    return metrics_list

def test_model(model, dataset, idx_range):

    dc = Data_consistency()
    i = 0

    _, coil_example, _, _, _  = dataset[0]
    Nb, Nc, Nx, Ny = coil_example.shape
    metrics = np.zeros((2, Nb, len(idx_range)))
    with torch.no_grad():
        omega_mask_r4 = dataset.omega_mask_r4.unsqueeze(0)
        omega_mask_r2 = dataset.omega_mask_r2.unsqueeze(0)

        for idx in idx_range:

            ksp, coils, _, _, _  = dataset[idx]

            ksp = ksp.unsqueeze(0)
            coils = coils.unsqueeze(0)
            
            # Forward pass
            output, _ = model(ksp*omega_mask_r4, coils, omega_mask_r4)

            zerofilled_r2 = dc.EH(ksp*omega_mask_r2, coils, omega_mask_r2)
            ref_recon = dc(zerofilled_r2, coils, omega_mask_r2, CG_iter = 25)

            rssq_coils = torch.sum(torch.square(torch.abs(coils)), dim = -3)

            ref_recon = ref_recon * rssq_coils
            output = output * rssq_coils

            for j in range(Nb):
                metrics[:,j,i] = get_perf_metrics(ref_recon[0,j,:,:], output[0,j,:,:])

            i+=1
    return metrics

def plot_masks(SMS_dataset):
    
    kspace, coils, theta_mask, lambda_mask, _ = SMS_dataset[0]
    omega_mask = SMS_dataset.omega_mask_r4

    fig, ax = plt.subplots(1,3,figsize=(10,5))
    ax[0].imshow(lambda_mask[0,:,:].squeeze().cpu().abs(), cmap = "gray")
    ax[0].set_title(f"Loss mask, ratio = {(torch.sum(lambda_mask)/torch.sum(omega_mask)).abs():.3f}")
    ax[1].imshow(theta_mask[0,:,:].squeeze().cpu().abs(), cmap = "gray")
    ax[1].set_title("Data consistency mask")
    ax[2].imshow(omega_mask[0,:,:].squeeze().cpu().abs(), cmap = "gray")
    ax[2].set_title("Full mask")

    plt.gray()
    for ax in fig.get_axes():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)

    fig.tight_layout()
    conf = Config().parse()
    if conf.wandb:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        wandb.log(({"masks": wandb.Image(Image.open(buf))}))




def plot_examples(model, dataset, slice_range, save_path = None, epoch = 0):
    
    j = 0
    fs = 17
    conf = Config().parse() 
    dc = Data_consistency()
    plt.gray()
    model.eval()
    caipi_shift = np.array([3,2,1,0,-1]) * 42

    with torch.no_grad():
        omega_mask_r4 = dataset.omega_mask_r4.unsqueeze(0)
        omega_mask_r2 = dataset.omega_mask_r2.unsqueeze(0)
        for i in slice_range:
            if i > len(dataset): continue
            ksp, coils, grappa_kspace = dataset[i]

            ksp = ksp.unsqueeze(0)
            coils = coils.unsqueeze(0)
            grappa_kspace_r2 = grappa_kspace[0].unsqueeze(0)
            grappa_kspace_r4 = grappa_kspace[1].unsqueeze(0)

            Nbatches, Nb, Nc, Nx, Ny = coils.shape
            
            rss_coils = torch.sum(torch.square(torch.abs(coils)), dim = -3)

            grappa_recon_r4 = dc.single_slice_EH(grappa_kspace_r4, coils)
            grappa_recon_r4 = (grappa_recon_r4 * rss_coils).abs().squeeze()

            grappa_recon_r2 = dc.single_slice_EH(grappa_kspace_r2, coils)
            grappa_recon_r2 = (grappa_recon_r2 * rss_coils).abs().squeeze()
            

            cnn_recon, _ = model(ksp, coils, omega_mask_r4)
            cnn_recon = cnn_recon * rss_coils
            cnn_recon = cnn_recon.abs().squeeze()
            
            fig, ax = plt.subplots(3, Nb, figsize=(16, 7.5))
                
            for i in range(Nb):
                psnr_val_grappa, ssim_val_grappa = get_perf_metrics(grappa_recon_r4[i,:,:], grappa_recon_r2[i,:,:])
                psnr_val_cnn, ssim_val_cnn = get_perf_metrics(cnn_recon[i,:,:], grappa_recon_r2[i,:,:])
                cnn_recon[i,:,:] = cnn_recon[i,:,:].roll(caipi_shift[i], 0)
                grappa_recon_r4[i,:,:] = grappa_recon_r4[i,:,:].roll(caipi_shift[i], 0)
                grappa_recon_r2[i,:,:] = grappa_recon_r2[i,:,:].roll(caipi_shift[i], 0)

                ax[0,i].imshow(grappa_recon_r2[i,:,:].cpu())
                if i==0: ax[0,i].set_ylabel('SPSG R=2', fontsize=fs)

                ax[1,i].imshow(grappa_recon_r4[i,:,:].cpu())
                ax[1,i].text(0.02, 0.02,
                f'PSNR: {psnr_val_grappa:.3f}\nSSIM: {ssim_val_grappa:.3f}',
                color='white', fontsize=fs-7, fontweight='bold',
                ha='left', va='bottom', transform=ax[1,i].transAxes)
                if i==0: ax[1,i].set_ylabel('SPSG R=4', fontsize=fs)

                ax[2,i].imshow(cnn_recon[i,:,:].cpu())
                if i==0: ax[2,i].set_ylabel('PDDL recon R=4', fontsize=fs)
                ax[2, i].text(0.02, 0.02,
                f'PSNR: {psnr_val_cnn:.3f}\nSSIM: {ssim_val_cnn:.3f}',
                color='white', fontsize=fs-7, fontweight='bold',
                ha='left', va='bottom', transform=ax[2, i].transAxes)

            for ax in fig.get_axes():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='x', length=0)
                ax.tick_params(axis='y', length=0)
            
            fig.suptitle(f"Reconstruction results | Epoch = {epoch}")
            fig.tight_layout()
            if save_path is not None and conf.plot_local:
                directory_path = Path(save_path)
                directory_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path / f"epch_{epoch}_{j}.png")

            if conf.wandb:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                wandb.log(({"epoch": epoch, f"example_{j}": wandb.Image(Image.open(buf))})) 
            j += 1




def get_unrolled_model(model_path, device):
    model = UnrolledNet().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def plot_metrics(save_path, train_metrics, val_metrics, epoch):
    conf = Config().parse()

    fig, ax = plt.subplots(1,2,figsize = (6,4))

    ax[0].plot(np.linspace(0, epoch+1, epoch+1), train_metrics[0,:epoch+1], label = "Training")
    ax[0].plot(np.linspace(0, epoch+1, epoch//conf.val_freq+1), val_metrics[:epoch//conf.val_freq+1], label = "Validation")
    ax[0].set_title("Training and Validation loss", fontsize = 15)
    ax[0].legend(loc='best')

    ax[1].plot(train_metrics[1,:epoch+1])
    ax[1].set_title("MU value", fontsize = 15)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path / "metrics_plots.png")
        plt.close('all')



def quick_plot(img, path = "test.png"):
    fig, ax = plt.subplots(1,1)
    ax.imshow(img, cmap = "gray")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    ax.axis('off')
    fig.savefig(path,bbox_inches='tight', pad_inches=0)
    plt.close('all')

def mask_generator(nx,ny,r,coil_dim = 0):
    mask = np.zeros((r,ny))
    mask[-1,:] = 1
    mask = np.tile(mask,(nx//r+1, 1))
    mask = mask[:nx,:ny]
    if coil_dim > 0:
        mask = np.tile(mask[np.newaxis,:,:], (coil_dim,1,1))
    return mask