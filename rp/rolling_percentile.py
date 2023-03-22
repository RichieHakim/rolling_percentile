import time

import torch


class Rolling_percentile_online:
    """
    Class for doing online stepwise rolling percentile calculation.
    RH 2023
    """
    def __init__(
        self, 
        win_len: int, 
        n_vals: int, 
        ptile: float, 
        device: str='cpu',
        x_buffer_init=None,
        use_jit=True,
        dtype=torch.float32,
    ):
        """
        Initialize rolling percentile object.

        Args:
            win_len (int):
                Window length.
            n_vals (int):
                Number of values to calculate percentile for.
            ptile (float):
                Percentile to calculate.
                0-100.
            device (str):
                Device to use.
                'cpu' or 'cuda'.
            x_buffer_init (torch.Tensor):
                Initial values for x_buffer.
                To use, must be of shape (n_vals, win_len).
                Also make sure to set self.iter to at least win_len
                 so that it will pull from the prepoluated buffer.
                If None, will initialize to torch.inf.
            use_jit (bool):
                Whether to use torch.jit.script to speed up step function.
            dtype (str or torch.dtype):
                Data type to use for self.x_buffer and self.x_sorted.
                If str, must be a valid torch dtype.
        """

        self._dtype = dtype if not isinstance(dtype, str) else getattr(torch, dtype)
        self._device = device

        self._win_len = int(win_len)
        self.ptile = ptile
        self.idx_ptile = int(round(win_len * (ptile/100)))

        if x_buffer_init is None or x_buffer_init=='None':
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

        self._use_jit = use_jit if use_jit==True else use_jit==1
        self.step_helper = torch.jit.script(_step_helper) if self._use_jit else _step_helper

    def step(self, vals_new):
        """
        Stepwise rolling percentile calculation.
        Wrapper for _step_helper so that it can be jit compiled.

        Args:
            vals_new (np.ndarray or torch.Tensor):
                New values to add to rolling percentile calculation.
                Must be of shape (n_vals,).

        Returns:
            p (torch.Tensor):
                Percentile values.
        """
        vals_new = torch.as_tensor(vals_new, dtype=self._dtype, device=self._device)
        idx_ptile = min(self.idx_ptile, round(self.iter * (self.ptile/100)))
        p, self.x_buffer, self.x_sorted, self.iter = self.step_helper(vals_new, self.x_buffer, self.x_sorted, self.arange_arr, self.iter, self._win_len, idx_ptile)
        return p
    
    def multistep(self, array):
        """
        Rolling percentile calculation for array of values.

        Args:
            array (np.ndarray or torch.Tensor):
                Values to calculate rolling percentile for.
                Must be of shape (n_features, n_samples).

        Returns:
            p (torch.Tensor):
                Percentile values.
                Shape (n_features, n_samples).
        """
        array = torch.as_tensor(array, dtype=self._dtype, device=self._device)
        p = torch.empty_like(array)
        for i in range(array.shape[1]):
            p[:,i] = self.step(array[:,i])
        return p

def _step_helper(vals_new, x_buffer, x_sorted, arange_arr, iter: int, win_len: int, idx_ptile: int):
    """
    Stepwise rolling percentile calculation.

    Args:
        vals_new (torch.Tensor):
            New values to add to rolling percentile calculation.
            Must be of shape (n_vals,).
        x_buffer (torch.Tensor):
            Buffer of values.
            Must be of shape (n_vals, win_len).
        x_sorted (torch.Tensor):
            Sorted buffer of values.
            Must be of shape (n_vals, win_len).
        arange_arr (torch.Tensor):
            Array of arange(win_len) for each row.
            Must be of shape (n_vals, win_len).
        iter (int):
            Iteration number.
        win_len (int):
            Window length.
        idx_ptile (int):
            Index of percentile value.

    Returns:
        p (torch.Tensor):
            Percentile values.
    """
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
