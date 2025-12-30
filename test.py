from utils import *
from utils import *
from Unrolled_Net.data_consistency import Data_consistency
from configs import Config
import torch
import os
import h5py 

conf = Config().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = conf.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_path = Path(
    f"/home/naxos2-raid40/avela019/Dinge/Forschung/Parallel_reconstruction/SMS/"
    f"saved_results/{conf.out_path}/{conf.train_type}/R{conf.acc_rate}"
)
results_path = output_path / f"example_slices/"
test_path = output_path / f"test_slices/"
test_path.mkdir(exist_ok=True, parents=True)

dc = Data_consistency()

SMS_dataset =  kspace_SMS_dataset("train", device, acc_rate = 4)
val_dataset_2 =  kspace_SMS_dataset("val", device, acc_rate = 2)
val_dataset_4 =  kspace_SMS_dataset("val", device, acc_rate = 4)

# with h5py.File(SMS_dataset.file_list[0], 'r') as f:
#     print("Keys (headers) in the MAT file:")
#     for name, item in zip(f.keys(), f.values()):
#         print(f"- {name}, shape {item.shape}")


# with h5py.File(SMS_dataset.file_list[0], 'r') as f:
#     print(f['mr2'][:], f['mr4'][:])
#         #    

unrolled_model = UnrolledNet(test = True).to(device)
unrolled_model.load_state_dict(torch.load(output_path/"unrolled_model.pth", weights_only=True))
unrolled_model.eval()

kspace, coils, theta_mask, lambda_mask = val_dataset_4[0]
omega_mask = SMS_dataset.omega_mask


fig, ax = plt.subplots(1,3,figsize=(10,5))
ax[0].imshow(lambda_mask[0,:,:].squeeze().cpu().abs(), cmap = "gray")
ax[0].set_title(f"Loss mask, ratio = {torch.sum(theta_mask)/torch.sum(omega_mask):.3f}")
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
fig.savefig(test_path / "masks.png")


for train in [True, False]:
    for DC_or_resnet in [True, False]:
        l1 = "TRAINING" if train else "TESTING"
        l2 = "BEFORE" if DC_or_resnet else "AFTER"
    
        kspace = kspace.unsqueeze(0)
        coils = coils.unsqueeze(0)
        theta_mask = theta_mask.unsqueeze(0)
        lambda_mask = lambda_mask.unsqueeze(0)
        omega_mask = omega_mask.unsqueeze(0)
    
        if not train:
            theta_mask = omega_mask
    
        y_theta  = kspace*theta_mask
        y_lambda = kspace*lambda_mask
    
        output, mu, inter_out_DC, inter_out_RES = unrolled_model(y_theta, coils, theta_mask)
        enc_output = dc.E(output, coils, lambda_mask)
    
        fig, ax = plt.subplots(1,10,figsize = (18,7))
        for i in range(10):
            if DC_or_resnet:
                ax[i].imshow(inter_out_DC[i].squeeze())
            else:
                ax[i].imshow(inter_out_RES[i].squeeze())
            ax[i].set_title(f"Block {i}", fontsize = 14)
        fig.suptitle(f"Network intermediate outputs {l2} resnet using {l1} mask", fontsize = 20)
    
    
        for ax in fig.get_axes():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)
    
        fig.savefig(test_path/f"inter_outputs_{l1}_{l2}.png")
    

'''
x_zerofilled = dc.EH(kspace, coils, omega_mask)

b = 3


fig, ax = plt.subplots(1,5, figsize = (15,4))
ax[0].imshow(x_zerofilled[0,b,:,:].squeeze().abs().detach().cpu(), cmap = "gray")
ax[0].set_title(f"Zerofilled image, R = {conf.acc_rate}", fontsize = 16)
for j, iter in enumerate([3,5,10,30]):
    print(j)
    x_cg = dc(x_zerofilled, coils, omega_mask)
    ax[j+1].imshow(x_cg[0,b,:,:].squeeze().abs().detach().cpu(), cmap = "gray")
    ax[j+1].set_title(f"CG SENSE, {iter} iters.", fontsize = 16)

plt.gray()
for ax in fig.get_axes():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)

fig.tight_layout()
plt.savefig("test_cg.png")

quick_plot(lambda_mask[0,0,:,:].squeeze().cpu().abs(), "lambda_mask.png")
quick_plot(theta_mask[0,0,:,:].squeeze().cpu().abs(), "theta_mask.png")
quick_plot(omega_mask[0,0,:,:].squeeze().cpu().abs(), "omega_mask.png")

'''
