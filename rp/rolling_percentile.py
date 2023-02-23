import time

import torch


class Rolling_percentile_online:
    """
    Class for doing online stepwise rolling percentile calculation.
    RH 2023
    """
    def __init__(
        self, 
        win_len, 
        n_vals, 
        ptile, 
        device=torch.device('cpu'),
        dtype=torch.float32,
        x_buffer_init=None,
        use_jit=True,
    ):
        self._dtype = dtype
        self._device = device

        self._win_len = win_len
        self.ptile = ptile
        self.idx_ptile = int(round(win_len * (ptile/100)))

        if x_buffer_init is None:
            self.x_buffer = torch.empty((n_vals, win_len), dtype=self._dtype).to(self._device)
            self.x_buffer[:] = torch.inf 
            self.iter = 0
        else:
            self.x_buffer = x_buffer_init.to(self._device)
            self.iter = win_len

        self.arange_arr = torch.stack([torch.arange(win_len)]*n_vals, dim=0).to(self._device)

        self.x_buffer = self.x_buffer.contiguous()
        self.arange_arr = self.arange_arr.contiguous()

        self.x_sorted = torch.sort(self.x_buffer, dim=1)[0].contiguous()

        self._use_jit = use_jit
        self.step_helper = torch.jit.script(_step_helper) if self._use_jit else _step_helper

    def step(self, vals_new):
        vals_new = torch.as_tensor(vals_new, dtype=self._dtype, device=self._device)
        idx_ptile = min(self.idx_ptile, round(self.iter * (self.ptile/100)))
        p, self.x_buffer, self.x_sorted, self.iter = self.step_helper(vals_new, self.x_buffer, self.x_sorted, self.arange_arr, self.iter, self._win_len, idx_ptile)
        return p

def _step_helper(vals_new, x_buffer, x_sorted, arange_arr, iter, win_len, idx_ptile):
    # tic = time.time()
    vals_new[torch.isnan(vals_new)] = torch.inf
    vals_new = vals_new.contiguous()

    vals_old = x_buffer[:, iter % win_len].contiguous()

    idx_ins = torch.searchsorted(
    # idx_tmp = torch.searchsorted(
        x_sorted, 
        vals_new[:,None],
        # torch.cat((vals_new[:,None], vals_old[:,None]), dim=1),
        side='left',
    )
    # idx_ins, idx_del = idx_tmp[:,0][:,None], idx_tmp[:,1][:,None]
    # print(idx_ins)
    idx_del = torch.searchsorted(
        x_sorted, 
        vals_old[:,None],
        side='left',
    )

    s = torch.sign(idx_del - idx_ins)

    ## separate cases for left vs right shifts
    shift_left = (s==-1)
    shift_right = (s==1)

    mask_left  = ((arange_arr >  idx_del) * (arange_arr <= idx_ins)) * shift_left
    mask_right = ((arange_arr >= idx_ins) * (arange_arr <  idx_del)) * shift_right

    mask = mask_left + mask_right
    mask_shifted = torch.roll(mask_left, -1, dims=1) + torch.roll(mask_right, 1, dims=1)
    
    ## shift
    ### RATE LIMITING STEP
    #### Faster for GPU
    x_sorted[mask_shifted] = x_sorted[mask]
    
    #### Sometimes faster for CPU, but not always
    # x_sorted2 = x_sorted.clone()
    # for ii in range(x_sorted.shape[0]):
    #     if shift_left[ii,0]:
    #         x_sorted[ii, idx_del[ii,0]+1:idx_ins[ii,0]+1] = x_sorted2[ii, idx_del[ii,0]:idx_ins[ii,0]]
    #     elif shift_right[ii,0]:
    #         x_sorted[ii, idx_ins[ii,0]:idx_del[ii,0]] = x_sorted2[ii, idx_ins[ii,0]+1:idx_del[ii,0]+1]

    # print(time.time() - tic)

    ## insert new value
    x_sorted[torch.arange(x_sorted.shape[0]), (idx_ins - (s==-1).type(torch.long)).squeeze()] = vals_new

    x_buffer[:, iter % win_len] = vals_new
    iter += 1

    ## return percentile
    return x_sorted[torch.arange(x_sorted.shape[0]), idx_ptile], x_buffer, x_sorted, iter
