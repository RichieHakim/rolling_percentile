import numpy as np
from . import shift_chunk, shift_chunk_inplace, binary_search, find_shift_params

# test shift_chunk

print(f'testing output version')

x_test = np.array([0,1,2,3,4,5])
out_test = shift_chunk(x_test, 1,3, shift=1)
pass_test = np.allclose(out_test, np.array([0, 1, 1, 2, 3, 5]))
print(f'positive shift test pass: {pass_test}')

out_test = shift_chunk(x_test, 1,3, -1)
pass_test = np.allclose(out_test, np.array([1,2,3,3,4,5]))
print(f'negative shift test pass: {pass_test}')

out_test = shift_chunk(x_test, 1,3, 0)
pass_test = np.allclose(out_test, np.array([0,1,2,3,4,5]))
print(f'negative shift test pass: {pass_test}')

print()
print(f'testing inplace version')

x_test = np.array([0,1,2,3,4,5])
shift_chunk_inplace(x_test, 1,3, shift=1)
pass_test = np.allclose(x_test, np.array([0, 1, 1, 2, 3, 5]))
print(f'positive shift test pass: {pass_test}')

x_test = np.array([0,1,2,3,4,5])
shift_chunk_inplace(x_test, 1,3, -1)
pass_test = np.allclose(x_test, np.array([1,2,3,3,4,5]))
print(f'negative shift test pass: {pass_test}')

x_test = np.array([0,1,2,3,4,5])
shift_chunk_inplace(x_test, 1,3, 0)
pass_test = np.allclose(x_test, np.array([0,1,2,3,4,5]))
print(f'negative shift test pass: {pass_test}')


# test binary_search
x_test = np.array([0,1,3,3.5,4,4,5,5,5,50])
vals_toTest = np.array([-np.inf, -1, 0, 0, 0, 0.1, 0.5, 0.99999999, 1, 3, 3.25, 3.5, 3.75, 4, 4, 5, 5, 6, 50, 51, np.inf])
idx = np.ones(len(vals_toTest))*-9
found = np.ones(len(vals_toTest))*-9
for ii,val in enumerate(vals_toTest):
    found[ii], idx[ii] = binary_search(x_test, 0, len(x_test)-1, val)
pass_test = np.allclose(found, np.array([0,0,1,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,1,0,0]))
print(f'found test pass: {pass_test}')
np.allclose(idx, np.array([0,0,0,0,0,1,1,1,1,2,3,3,4,4,4,7,7,9,9,10,10]))
print(f'idx test pass: {pass_test}')


# test find_shift_params
idx_del = 3
idx_ins = 7
shift = 1
idx_l, idx_r, shift = find_shift_params(idx_del, idx_ins, shift=shift)
test_pass = np.allclose( np.array([idx_l, idx_r, shift]) , np.array([4,6,-1]) )
print(f'negative shift test pass: {test_pass}')

idx_del = 5
idx_ins = 2
shift = 1
idx_l, idx_r, shift = find_shift_params(idx_del, idx_ins, shift=shift)
np.array([idx_l, idx_r, shift]) 
test_pass = np.allclose( np.array([idx_l, idx_r, shift]) , np.array([2,4,1]) )
print(f'positive shift test pass: {test_pass}')

idx_del = 2
idx_ins = 2
shift = 1
idx_l, idx_r, shift = find_shift_params(idx_del, idx_ins, shift=shift)
np.array([idx_l, idx_r, shift]) 
test_pass = np.allclose( np.array([idx_l, idx_r, shift]) , np.array([-1,-1,0]) )
print(f'positive shift test pass: {test_pass}')
