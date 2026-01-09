import numpy as np

def get_grappa_kernel(acs, mask, ks, lmbd = 1e-8, grappa_type = "spsg"):
    ns, nc, cx, cy = acs.shape
    kx, ky = ks
    kn = kx*ky
    kc = kn//2
    k1,k2 = kx//2, ky//2

    acs = np.pad(acs, ((0,0), (0,0), (k1,k1), (k2,k2)))
    mask = np.pad(mask,((k1,k1),(k2,k2)))

    mask_patches = np.lib.stride_tricks.sliding_window_view(mask, ks) #[nx, ny, kx, kx] 
    mask_patches = mask_patches.transpose(0,1,3,2)
    mask_patches = mask_patches.reshape(-1,kn) #[nx*ny, ky*kx] 
    P, patch_idx = np.unique(mask_patches, axis=0, return_inverse=True)
    #Get valid patches (all that have non-zero elements)
    validP = np.argwhere(~np.all(1-P, axis=1)).squeeze()

    acs_patches = np.lib.stride_tricks.sliding_window_view(acs, ks, axis=(-1,-2))
    acs_patches = acs_patches.reshape(ns,nc,cx*cy,kn) #[ns, nc, cy*cx, ky*kx] 
    acs_patches = acs_patches.transpose(0,2,3,1) #[ns, cy*cx, ky*kx, nc] 

    if grappa_type == "sg":
        acs_scr = np.sum(acs_patches, axis = 0) #[cy*cx, ky*kx, nc] 
    elif grappa_type == "spsg":
        acs_scr = acs_patches

    #Get kernels for each slice and each patch
    kernels = {}
    for z in range(ns):
        for i in validP:
            p = P[i]
            nonzero_idx = np.nonzero(p)[0]
            src = acs_scr[...,nonzero_idx,:]
            src = src.reshape(-1,len(nonzero_idx)*nc)
            trg = acs_patches[z,:,kc,:]

            if grappa_type == "spsg":
                zero_slices = np.ones(ns,dtype=int)
                zero_slices[z] = 0
                trg = np.tile(trg[np.newaxis,...],(ns,1,1))
                trg[zero_slices==1,:,:] = 0
                trg = trg.reshape(ns*cx*cy,nc)
            
            AhA = np.conj(src.T) @ src
            Ahb = np.conj(src.T) @ trg

            n = AhA.shape[0]
            x = np.linalg.solve(AhA + np.eye(n)*lmbd, Ahb).T
            kernels[(z,i)] = (x, patch_idx, nonzero_idx)
    
    return kernels

def fill_grappa_kspace(data, kernel, ks, ns):
    nc, nx, ny = data.shape
    kx, ky = ks
    k1, k2 = kx//2, ky//2
    kn = kx*ky

    data_pad = np.pad(data, ((0,0), (k1,k1), (k2,k2)))
    data_patches = np.lib.stride_tricks.sliding_window_view(data_pad, ks, axis=(-1,-2)) #[nc, nx,ny, kx,ky] 
    data_patches = data_patches.reshape(nc, nx*ny, kn) #[nc, nx*ny, ky*kx] 
    data_patches = data_patches.transpose(1,2,0) #[nx*ny, ky*kx, nc] 

    grappa_data = np.zeros((ns,nx*ny,nc), dtype=data_patches.dtype)

    for keys, values in kernel.items():
        z,i = keys
        x, patch_idx, nonzero_idx = values
        n_idx = np.count_nonzero(patch_idx == i)
        y = data_patches[:, nonzero_idx, :]
        y = y[patch_idx == i, :, :]

        y = y.reshape(n_idx,-1)
        y = y[:,np.newaxis,:] * x
        grappa_data[z,patch_idx == i, :] = np.sum(y, axis=-1)
    
    grappa_data = grappa_data.reshape(ns, nx, ny, nc)
    return grappa_data
 



