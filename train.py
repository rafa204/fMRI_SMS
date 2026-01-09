import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from Unrolled_Net.UnrolledNet import UnrolledNet
from Unrolled_Net.data_consistency import Data_consistency
from configs import Config
from utils import *
from utils import *
import wandb

# ====================== CONFIGURATION ======================

# Command-line arguments
conf = Config().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = conf.cuda

save_model_each_epoch = True
plot_final_results = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_type = "SSDU-SM" if conf.n_masks == 1 else "SSDU-MM"

#Prepare paths to output results
output_path = Path(
    f"/home/naxos2-raid40/avela019/Dinge/Forschung/Parallel_reconstruction/SMS/"
    f"saved_results/{conf.wandb_group}/{conf.out_path}/{train_type}"
)

results_path = output_path / f"example_slices/" #to save example reconstructions
output_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)

if conf.wandb:
    wandb.init(
    project = "parallel_recon",
    entity = "avela019-umn",
    group = conf.wandb_group,
    job_type = "train",
    name = conf.out_path,
    config = vars(conf)
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

# ====================== DATASET SETUP ======================

# Instantiate datasets
train_dataset = kspace_SMS_dataset("train", device)
val_dataset   = kspace_SMS_dataset("val", device)
test_dataset  = kspace_SMS_dataset("test", device)

#Plot example masks for reference
plot_masks(train_dataset)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 1)

# ====================== MODEL & OPTIMIZER ======================

dc = Data_consistency()
unrolled_model = UnrolledNet().to(device)
optimizer = optim.Adam(unrolled_model.parameters(), lr=conf.lr)

if conf.LR_sch:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.n_epochs)

train_metrics = np.zeros((2, conf.n_epochs))
val_metrics = np.zeros(conf.n_epochs//conf.val_freq + 1)
best_val_loss = np.inf
best_epoch = True
print(f"Starting {train_type} training: {conf.out_path}")

# ====================== MAIN TRAINING LOOP ======================
loss_array = np.zeros((len(train_dataset), conf.n_epochs)) 
plot_examples(unrolled_model, test_dataset, range(conf.n_plot), save_path = results_path, epoch = -1)

for epoch in range(conf.n_epochs):
    unrolled_model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{conf.n_epochs} [Training]")
    avg_loss = 0.0

    for ksp, coils, theta_mask, lambda_mask, idx in train_bar:

        # Get undersampled measurements
        y_theta  = ksp*theta_mask
        y_lambda = ksp*lambda_mask

        # Forward pass
        output, mu = unrolled_model(y_theta, coils, theta_mask)
        enc_output = dc.E(output, coils, lambda_mask)

        # Backpropagate
        loss = L1_L2_norm(enc_output, y_lambda)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=loss.item())
        avg_loss += loss.item()
        loss_array[idx, epoch] = loss.item()
    
    if conf.wandb:
         wandb.log({"epoch": epoch,"train/loss": avg_loss/len(train_bar)})
         wandb.log({"epoch": epoch,"train/mu": mu})

    if conf.LR_sch > 0:
        if conf.wandb:
            wandb.log({"epoch": epoch,"train/lr":scheduler.get_last_lr()[0]})
        scheduler.step()
    
    np.save(results_path/"loss_array.npy", loss_array)

    #====================== VALIDATION & LOGGING ======================
    if (epoch % conf.val_freq == 0):
        val_loss = validate_model(unrolled_model, val_loader).mean()
        test_metrics = test_model(unrolled_model, val_dataset, range(len(val_dataset)))
        avg_metrics = test_metrics.mean(axis=(-1,-2))
        best_epoch = val_loss < best_val_loss
        if best_epoch: best_val_loss = val_loss
        val_metrics[epoch//conf.val_freq] = val_loss
        np.save(output_path / f"val_metrics.npy", val_metrics)
        if conf.wandb:
            wandb.log({"epoch": epoch,"val/loss": val_loss})
            wandb.log({"epoch": epoch,"val/PSNR": avg_metrics[0]})
            wandb.log({"epoch": epoch,"val/SSIM": avg_metrics[1]})

    if (epoch % conf.plot_freq == 0):
        slice_range = np.floor(np.linspace(0, len(val_dataset), conf.n_plot)).astype(int)
        plot_examples(unrolled_model, val_dataset, slice_range, save_path = results_path, epoch = epoch)

    train_metrics[0, epoch] = avg_loss / len(train_bar)
    train_metrics[1, epoch] = mu
    np.save(output_path / f"train_metrics.npy", train_metrics)
    
    if(epoch % conf.save_freq == 0 or epoch == conf.n_epochs-1 and best_epoch):
        torch.save(unrolled_model.state_dict(), output_path / f"unrolled_model.pth")
    plot_metrics(output_path, train_metrics, val_metrics, epoch)

print("---------DONE TRAINING---------")
# ============ TESTING ==============

#plot_examples(unrolled_model, val_dataset, range(0, conf.n_test, 100), save_path = results_path)
'''
test_metrics = test_model(unrolled_model,  test_dataset, range(conf.n_test))
avg_metrics = test_metrics.mean(axis=(-1,-2))

np.save(output_path / "test_metrics.npy", test_metrics)
print(avg_metrics)
if conf.wandb:
    wandb.log({"epoch": epoch,"test/PSNR": avg_metrics[0]})
    wandb.log({"epoch": epoch,"test/SSIM": avg_metrics[1]})
with open(output_path / "params.pkl", 'wb') as file:
    pickle.dump(conf, file)
'''