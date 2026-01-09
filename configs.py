import argparse
import sys

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        # directory for the out directory
        self.parser.add_argument('--out_path', type=str, default='test', help='results file directory')
        self.parser.add_argument('--cuda', type=str, default='0', help='CUDA device to use')
        self.parser.add_argument('--wandb_group', type=str, default='01-09', help='Name of wandb group')
        self.parser.add_argument('--wandb', type=int, default=0, help='use weights and biases to record results')
        
        # hyperparameters for the type of training type
        self.parser.add_argument('--n_masks', type=int, default=1, help='num masks for multi-mask')
        self.parser.add_argument('--gauss', type=int, default=0, help='whether to use gauss. distribution')
        self.parser.add_argument('--lambda_ratio', type=float, default=0.416, help='sampling ratio for lambda mask')
        self.parser.add_argument('--center_size', type=int, default=3, help='sampling ratio for lambda mask')
        self.parser.add_argument('--ordered', type=int, default=1, help='sampling ratio for lambda mask')
        
        # Hyperparameters for leaning
        self.parser.add_argument('--n_train', type=int, default=480, help='number of slices in training')
        self.parser.add_argument('--n_val', type=int, default=64, help='number of slices in validation')
        self.parser.add_argument('--n_test', type=int, default=640, help='number of slices in testing')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
        self.parser.add_argument('--batchSize', type=int, default=1, help='batch size')
        self.parser.add_argument('--LR_sch', type=int, default=0, help='Is there LR schedule or not')
        self.parser.add_argument('--save_freq', type=int, default=1, help='result saving frequency')
        self.parser.add_argument('--val', type=int, default=0, help='Validation or not')
        self.parser.add_argument('--val_freq', type=int, default=1, help='Validation freq')
        self.parser.add_argument('--plot_freq', type=int, default=10, help='Plotting reconstructions freq') 
        self.parser.add_argument('--plot_local', type=int, default=0, help='Plotting reconstructions to file?') 
        self.parser.add_argument('--n_plot', type=int, default=3, help='slices to plot')

        # hyperparameters for the unrolled network
        self.parser.add_argument('--nb_unroll_blocks', type=int, default=10, help='number of unrolled blocks')
        self.parser.add_argument('--nb_res_blocks', type=int, default=15, help="number of residual blocks in ResNet")
        self.parser.add_argument('--CG_Iter', type=int, default=10, help='number of Conjugate Gradient iterations for DC')

    
        # hyperparameters for the dataset
        self.parser.add_argument('--acc_rate', type=int, default=4, help='acceleration rate')
        self.parser.add_argument('--n_groups', type=int, default=16, help='acceleration rate')
        self.parser.add_argument('--n_bands', type=int, default=5, help='acceleration rate')
        self.parser.add_argument('--n_coils', type=int, default=32, help='acceleration rate')
        self.parser.add_argument('--n_rows', type=int, default=110, help='acceleration rate')
        self.parser.add_argument('--n_cols', type=int, default=128, help='acceleration rate')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)

        return self.conf
