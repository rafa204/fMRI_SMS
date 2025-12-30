import torch
import torch.nn as nn
from Unrolled_Net.ResNet import ResNet
from Unrolled_Net.data_consistency import Data_consistency


class UnrolledNet(nn.Module):
    def __init__(self, mu = 0.05, nb_res_blocks = 15, nb_unroll_blocks = 10, test = False):
        super(UnrolledNet, self).__init__()
        
        self.resnet = ResNet(nb_res_blocks, in_channels = 2)
        self.dc = Data_consistency()
        self.nb_unroll_blocks = nb_unroll_blocks
        self.mu = torch.nn.Parameter(torch.tensor(mu), requires_grad = True)
        self.test = test
        
    def forward(self, kspace, coil, mask): 

        zerofilled = self.dc.EH(kspace, coil, mask)
        output = zerofilled.clone() #(N_batches, Nb, Nx, Ny)
        Nb, Nc, Nx, Ny = coil.shape[-4:]
        inter_output_DC = torch.zeros(self.nb_unroll_blocks,Nb*Nx, Ny)
        inter_output_RES = torch.zeros(self.nb_unroll_blocks,Nb*Nx, Ny)
        
        for i in range(self.nb_unroll_blocks):
            
            output = output.contiguous().view(-1, Nb*Nx, Ny)          #(N_batches, Nb * Nx, Ny )
            inter_output_DC[i] = output.detach().clone().abs().cpu()
            output = torch.stack((output.real, output.imag), axis=-3) #(N_batches, 2, Nb * Nx, Ny )
            output = self.resnet(output)                              #(N_batches, 2, Nb * Nx, Ny )
            output = output[..., 0, :, :] + 1j*output[..., 1, :, :]   #(N_batches, Nb * Nx, Ny)
            inter_output_RES[i] = output.detach().clone().abs().cpu()
            output = output.contiguous().view(-1, Nb, Nx, Ny)         #(N_batches, Nb, Nx, Ny)

            output = self.dc(zerofilled, coil, mask, output, self.mu)

        if self.test:
            return output, self.mu.item(), inter_output_DC, inter_output_RES
        else:
            return output, self.mu.item()