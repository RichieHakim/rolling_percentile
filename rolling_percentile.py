import numpy as np
from numba import jit, njit, prange


@njit(parallel=True)
def rolling_percentile(x_in, win_len, ptile):
    '''
    Rolling percentile. Calculated along dimension 1 (rows).
    Window is CENTERED, and BOXCAR.
    Indices near edges are NaN (min_period=win_len/2).
    This algorithm is faster than pandas' df.rolling().quantile()
     when win_len < 1000 for a single row input,
     or   win_len < 10000 when num rows > num CPU threads
    RH 2021

    Args:
        x_in (numpy.ndarray):
            Input array. Calculation performed along dimension 1
             (rows). 
        win_len (int):
            Window length.
        ptile (scalar, 0-100):
            Percentile. Will be converted to an integer index:
                idx_ptile = int(win_len * (ptile/100))
        
    Returns:
        out_ptile (numpy.ndarray):
            Output array. Same shape as x_in, but with NaNs
             near edges. Values are of the rolling percentile.
    '''        
    win_len_half = int(win_len/2)
    win_len_half
    idx_ptile = int(win_len * (ptile/100))

    # initialize output
    out_ptile = np.empty_like(x_in)
    out_ptile[:] = np.nan

    for jj in prange(x_in.shape[0]):
        x_win_sorted = np.sort(x_in[jj, 0:win_len])
        for ii in range(win_len_half, x_in.shape[1] - win_len_half-1):
            # Swap out line below with custom functions
            out_ptile[jj][ii] = x_win_sorted[idx_ptile]

            # centered rolling window
            idx_new = ii + win_len_half + 1
            val_new = x_in[jj, idx_new]

            idx_old = ii-win_len_half
            val_old = x_in[jj, idx_old]

            found, idx_ins = bisect_left(x_win_sorted, val_new)
            found, idx_del = bisect_left(x_win_sorted, val_old)

            idx_l, idx_r, shift = find_shift_params(idx_del, idx_ins, shift=1)

            shift_chunk_inplace(x_win_sorted, idx_l, idx_r, shift=shift)

            if shift==1 or shift==0:
                x_win_sorted[idx_ins] = val_new
            elif shift==-1:
                x_win_sorted[idx_ins-1] = val_new

    return out_ptile


@njit
def bisect_left(arr, val):
    '''
    Array bisection with left alignment.
    If val in array, then returns index of found val.
    If val not in array, then returns index to the right
     of largest value in array smaller than val.
    By using numba's jit. This function is faster than Python 3's
    built in bisect module.
    RH 2021

    Args: (same as in binary_search)
        arr (sorted list or numpy.ndarray): 
            1-D array of numbers that are already sorted.
            To use numba's jit features, arr MUST be a
            typed list. These include:
                - numpy.ndarray (ideally np.ascontiguousarray)
                - numba.typed.List
        val (scalar):
            value being searched for
    
    Returns: (same as in binary_search)
        output (tuple):
            output 1 (int):
                index of val in arr.
                returns index of a found value, 
                and returns the insertion index 
                if value is not present
            output 2 (bool 0/1):
                boolean for whether the value was found
                in the array
    '''
    return binary_search(arr, 0, len(arr)-1, val)
@njit
def binary_search(arr, lb, ub, val):
    '''
    Recursive binary search with outputs for left-aligned index if value not found.
    adapted from https://www.geeksforgeeks.org/python-program-for-binary-search/
    RH 2021
    
    Args:
        arr (sorted list):
            1-D array of numbers that are already sorted.
            To use numba's jit features, arr MUST be a
            typed list. These include:
                - numpy.ndarray (ideally np.ascontiguousarray)
                - numba.typed.List
        lb (int):
            lower bound index.
        ub (int):
            upper bound index.
        val (scalar):
            value being searched for
    
    Returns:
        output (tuple):
            output 1 (int):
                index of val in arr.
                returns index of a found value, 
                and returns the insertion index 
                if value is not present
            output 2 (bool 0/1):
                boolean for whether the value was found
                in the array
            
    Example:
        if arr=[1,2,3] and val=[2.5], return (0,2)
        if arr=[1,2,3] and val=[2], return (1,1)
    Demo:
        # Test array
        arr = np.array([ 2, 3, 4, 10, 40 ])
        x = 100

        # Function call
        result = binary_search(arr, 0, len(arr)-1, x)

        if result != -1:
            print("Element is present at index", str(result))
        else:
            print("Element is not present in array")
    '''
    # Check base case
    if ub >= lb:
 
        mid = (ub + lb) // 2
 
        # If element is present at the middle itself
        if arr[mid] == val:
            return (1, mid)
 
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > val:
            return binary_search(arr, lb, mid - 1, val)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, ub, val)
 
    else:
        # Element is not present in the array
        return (0, lb)


@njit
def find_shift_params(idx_del, idx_ins, shift=1):
    '''
    Finds parameters to use for functions:
        shift_chunk_inplace
        shift_chunk
    This function is used as a step in an deletion+insertion
     into a sorted array.
    It finds the indices and direction to shift a chunk of
     values in a sorted array so that the value at idx_del
     is deleted and the value at idx_ins may be inserted in
     subsequent steps. Doing it this way allows the array to
     remain sorted.
    RH 2021
    
    Args:
        idx_del (int):
            Index to be deleted.
        idx_ins (int):
            Index to be inserted.
        shift (int):
            Number of indices to shift chunk by. Only sign
            of input matters since output is only changed
            in sign or set to 0.
    
    Returns: (same as inputs to shift_chunk_inplace)
        idx_l (int):
            Left index. Elements starting from this position
            will be shifted in subsequent function.
        idx_r (int):
            Right index. Elements starting from this position
            will be shifted in subsequent function.
        shift (int):
            Number of indices to shift the chunk by. If
            negative, then shift is to the left. If positive,
            then shift is to the right. If 0, the x is
            unchanged.
    '''
    if idx_del < idx_ins:
        shift = -shift
        idx_l = idx_del + 1
        idx_r = idx_ins -1
    elif idx_del > idx_ins:
        shift = shift
        idx_l = idx_ins
        idx_r = idx_del -1
    elif idx_del == idx_ins:
        shift = 0
        idx_l = -1 # using -1 as a placeholder since calculation shouldn't occur
        idx_r = -1
    return idx_l, idx_r, shift



@njit
def shift_chunk_inplace(x, idx_l=None, idx_r=None, shift=1):
    '''
    Shifts/rolls a subset of an array from idx_l to idx_r
     (inclusive on both ends) by 'shift'.
    Operation performed in-place, meaning input array will be
     changed in memory (x).
    Useful as one of the steps in a sorted deletion+insertion
    RH 2021

    Args:
        x (list or numpy.ndarray):
            Input array
        idx_l (int):
            Left index. Elements starting from this position
            will be shifted
        idx_r (int):
            Right index. Elements ending from this position
            will be shifted.
        shift (int):
            Number of indices to shift the chunk by. If
            negative, then shift is to the left. If positive,
            then shift is to the right. If 0, the x is
            unchanged.

    Returns: NONE. This is an in-place operation
    '''
    if shift != 0:
        x[idx_l+shift:idx_r+1+shift] = x[idx_l:idx_r+1]

# import copy
# @njit
# def shift_chunk(x, idx_l=None, idx_r=None, shift=1):
#     out = x.copy()
#     if shift != 0:
#         out[idx_l+shift:idx_r+1+shift] = x[idx_l:idx_r+1]
#     return out


#######################################################
#################### MISC #############################
#######################################################

@jit
def sorted_roll_generator(x_in, win_len, x_win_sorted):
    '''
    NEED TO FIX DOCUMENTATION
    This is a generator function for generating sorted values
     taken by rolling along the axis of an array with a defined
     window length.
    This function can't be done in @njit, so its a little slower
    than the above function rolling_percentile.
    RH 2021

    Args:
        x_in (numpy.ndarray):
            Input array. Calculation performed along dimension 1
             (rows). 
        win_len (int):
            Window length.
        ptile (scalar, 0-100):
            Percentile. Will be converted to an integer index:
                idx_ptile = int(win_len * (ptile/100))
        
    Returns:
        sorted window of values (numpy.ndarray):
            Sorted output array of length win_len.
            If x_in.shape[0] > 1, then it goes through each
            row completely before going on to the next trace

    '''
    win_len_half = int(win_len/2)

    for jj in range(x_in.shape[0]):
        for ii in range(win_len_half, x_in.shape[1] - win_len_half-1):
            yield x_win_sorted[jj]
#             out_ptile[jj][ii] = x_win_sorted[jj][idx_ptile]

            # centered rolling window
            idx_new = ii + win_len_half + 1
            val_new = x_in[jj, idx_new]

            idx_old = ii-win_len_half
            val_old = x_in[jj, idx_old]


            found, idx_ins = binary_search(x_win_sorted[jj], 0, win_len-1, val_new)
            found, idx_del = binary_search(x_win_sorted[jj], 0, win_len-1, val_old)
    
            idx_l, idx_r, shift = find_shift_params(idx_del, idx_ins, shift=1)

            shift_chunk_inplace(x_win_sorted[jj], idx_l, idx_r, shift=shift)

            if shift==1 or shift==0:
                x_win_sorted[jj][idx_ins] = val_new
            elif shift==-1:
                x_win_sorted[jj][idx_ins-1] = val_new