from utils import *
from utils import *


file_list = ['SSDU-MM_test_lr_1e-4',
             'SSDM-MM_test_lr_3e-4',
             'SSDU-MM_test_lr_5e-4',]
legend_list = ['LR = 5e-4',
               'LR = 3e-4',
               'LR = 1e-4' ]

api = wandb.Api()
entity, project, group = "avela019-umn", "parallel_recon", "test_0" 
runs = api.runs(entity + "/" + project)
runs = [run for run in runs if run.group == group and run.name in file_list]

epochs = range(100)

fig, ax = plt.subplots(1,3,figsize = (12,4))
fs = 16

for i, run in enumerate(runs):
    print(run.name, run.name in file_list)
    
    history = run.history
    a = history(keys = ["val/loss"])
    val_loss = history(keys = ["val/loss"])['val/loss'].to_numpy()
    val_PSNR = history(keys = ["val/PSNR"])['val/PSNR'].to_numpy()
    val_SSIM = history(keys = ["val/SSIM"])['val/SSIM'].to_numpy()
    ax[0].plot(epochs, val_loss, label = legend_list[i])
    ax[0].set_ylabel("Loss", fontsize = fs)
    ax[1].plot(epochs, val_SSIM)
    ax[1].set_ylabel("SSIM", fontsize = fs)
    ax[2].plot(epochs, val_PSNR)
    ax[2].set_ylabel("PSNR", fontsize = fs)


for ax1 in ax:
    ax1.grid(True)

ax[0].legend(fontsize = 12)
fig.tight_layout()
fig.savefig("plot1.png")