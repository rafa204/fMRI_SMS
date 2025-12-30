import torch
from configs import Config

def cus_ifft(img, dims):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(img, dim=dims), dim=dims, norm = 'ortho'), dim=dims)
    
def cus_fft(kspace, dims):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(kspace, dim=dims), dim=dims, norm = 'ortho'), dim=dims)

"""
x = (E^h*E + mu*I)^-1 (E^h*y + mu*z)
"""
class Data_consistency(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = Config().parse()
        

    def repeat_along_neg_dim(self, x, N, negdim):
        x = x.unsqueeze(dim = negdim)
        dim = x.dim() + negdim
        repeats = [1] * x.dim()
        repeats[dim] = N
        return x.repeat(*repeats)

    def E(self, image, coil, mask):
        image = self.repeat_along_neg_dim(image, coil.shape[-3], -3)
        image = torch.sum(image*coil, dim = -4).squeeze()
        return cus_fft(image, [-2,-1]) * mask
        
    def EH(self, kspace, coil, mask):
        image = cus_ifft(kspace*mask, [-2,-1])
        image = self.repeat_along_neg_dim(image, coil.shape[-4], -4)
        return torch.sum(image * torch.conj(coil), dim=-3)

    def EHE(self, image, coil, mask):
        return self.EH(self.E(image, coil, mask), coil, mask)

    def forward(self, zerofilled, coil, mask, denoiser_out = None, mu = 0, CG_iter = 10):

        if denoiser_out is not None:
            r_now = zerofilled + mu*denoiser_out # E^h*y + mu*z = p
        else:
            r_now = zerofilled

        p_now = torch.clone(r_now)
        b_approx = torch.zeros_like(p_now)

        d = (-3,-2,-1) #Last three dimensions
        n_batches = zerofilled.shape[0]

        for _ in range(CG_iter):
            
            q = self.EHE(p_now, coil, mask) + mu * p_now # A * p = (E^h*E + mu*I) * p = E^hE(p) + mu*p
            alpha = torch.sum(r_now*torch.conj(r_now), dim = d) / torch.sum(q*torch.conj(p_now), dim = d)
            b_next = b_approx + alpha.view(n_batches, 1, 1, 1)*p_now
            r_next = r_now - alpha.view(n_batches, 1, 1, 1)*q
            beta = torch.sum(r_next*torch.conj(r_next), dim = d) / torch.sum(r_now*torch.conj(r_now), dim = d)
            p_next = r_next + beta.view(n_batches, 1, 1, 1)*p_now
            b_approx = b_next
    
            p_now = torch.clone(p_next)
            r_now = torch.clone(r_next)
            
        return b_approx