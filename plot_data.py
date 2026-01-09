from utils import *
from utils import *


file_list = ['SSDU-MM_lr_3e-4_0',
             'SSDU-MM_lr_3e-4_1',
             'SSDU-MM_lr_3e-4_2',
             'SSDU-MM_lr_3e-4_3',]
#legend_list = ['LR = 5e-4',
#               'LR = 3e-4',
#               'LR = 1e-4' ]
legend_list = file_list

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
    ax[0].plot(range(len(val_loss)), val_loss, label = legend_list[i])
    ax[0].set_ylabel("Loss", fontsize = fs)
    ax[1].plot(range(len(val_SSIM)), val_SSIM)
    ax[1].set_ylabel("SSIM", fontsize = fs)
    ax[2].plot(range(len(val_PSNR)), val_PSNR)
    ax[2].set_ylabel("PSNR", fontsize = fs)


ax[0].set_ylim([0.34, 0.7])
ax[1].set_ylim([0.6, 0.775])
ax[2].set_ylim([22, 25.7])
for ax1 in ax:
    ax1.grid(True)

#ax[0].legend(fontsize = 12)
fig.tight_layout()
fig.savefig("plot1.png")