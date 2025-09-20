import numpy as np

def four_step_fft_incorrect(x, n1, n2):
    """Forward DFT via Bailey's four-step for N=r*t. x: 1-D complex array."""
    x = np.asarray(x, dtype=np.complex64)
    N = x.size
    assert N == n1*n2 and (n1>0 and n2>0)

    # 1) reshape to r x t (rows contiguous, length-t)
    A = x.reshape(n1, n2)              # A[i, j] corresponds to x[i*t + j]

    # 2) do t-point FFTs across the contiguous axis (axis=1: columns within each row)
    A = np.fft.fft(A, axis=0)

    # 3) multiply by twiddle grid W_N^{i*j} = exp(-2πi * i*j/N), i=0..r-1, j=0..t-1
    i = np.arange(n1)[:, None]
    j = np.arange(n2)[None, :]
    W = np.exp(-2j*np.pi * (i*j) / N)
    A *= W

    # 4) transpose to t x r
    B = A.T.copy()                   # shape (t, r), rows now length-r contiguous

    # 5) do r-point FFTs along the contiguous axis (axis=1 again)
    B = np.fft.fft(B, axis=0)

    # 6) flatten back to 1-D in the standard order
    #    With the mapping above, the natural 1-D output is row-major flatten of B^T.
    y = B.reshape(N)
    # y = B.T.reshape(N)
    return y

def four_step_fft(x, r, t):
    """Forward DFT via Bailey's four-step for N=r*t. x: 1-D complex array."""
    x = np.asarray(x, dtype=np.complex64)
    N = x.size
    assert N == r*t and (r > 0 and t > 0)

    # Correct layout: A[n1, n2] = x[n1 + r*n2]
    # Two equivalent ways; pick one:
    A = x.reshape(t, r).T.copy()                 # rows contiguous, length t
    # A = np.reshape(x, (r, t), order='F').copy()  # also OK; copy makes rows contiguous

    # t-point FFTs along n2 (axis=1)
    A = np.fft.fft(A, axis=1)

    # Twiddle grid W_N^{n1 * k1}
    n1 = np.arange(r)[:, None]
    k1 = np.arange(t)[None, :]
    A *= np.exp(-2j*np.pi * (n1 * k1) / N).astype(np.complex64)

    # Transpose to t x r, then r-point FFTs along the contiguous axis
    B = A.T.copy()
    B = np.fft.fft(B, axis=1)

    # Flatten back: k = k1 + t*k2 corresponds to row-major flatten of B^T
    return B.T.reshape(N)
def six_step_fft(x, n1, n2):
    """
    Forward DFT of length N=n1*n2 via six-step.
    Equivalent to np.fft.fft(x).
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.size
    assert N == n1*n2 and n1 > 0 and n2 > 0

    # view as (n1, n2)
    X = x.reshape(n1, n2)

    # 1) transpose -> (n2, n1), so rows are contiguous length n1
    X = X.T.copy()

    # 2) do n1-FFTs along rows
    X = np.fft.fft(X, axis=1)

    # 3) twiddle: W_N^{j * k}, j=row index (0..n2-1), k=col index (0..n1-1)
    j = np.arange(n2)[:, None]
    k = np.arange(n1)[None, :]
    X *= np.exp(-2j*np.pi * (j * k) / N)

    # 4) transpose back -> (n1, n2), rows contiguous length n2
    X = X.T.copy()

    # 5) do n2-FFTs along rows
    X = np.fft.fft(X, axis=1)

    # 6) final transpose to standard 1-D order k = k1 + n1*k2
    # y = X.reshape(N)
    y = X.T.reshape(N)
    return y

# def four_step_fft(x, r, t):
#     """
#     Forward DFT of length N=r*t via Bailey's four-step.
#     x: 1-D complex array, N=r*t.
#     Returns y in the same frequency order as np.fft.fft(x).
#     """
#     x = np.asarray(x, dtype=np.complex128)
#     N = x.size
#     assert N == r*t and r > 0 and t > 0

#     # 1) reshape to (r, t): rows are contiguous length-t chunks
#     A = x.reshape(r, t)

#     # 2) do t-FFTs along rows (axis=1)
#     A = np.fft.fft(A, axis=1)

#     # 3) multiply by twiddle grid W_N^{i * k2}, where i=row index, k2=frequency index from step 2
#     i = np.arange(r)[:, None]     # shape (r,1)
#     k2 = np.arange(t)[None, :]    # shape (1,t)
#     A *= np.exp(-2j*np.pi * (i * k2) / N)

#     # 4) transpose -> (t, r); now rows are contiguous length-r
#     B = A.T.copy()

#     # 5) do r-FFTs along rows (axis=1)
#     B = np.fft.fft(B, axis=1)

#     # 6) map to 1-D: final index is k = k1 + r*k2 -> exactly B.T row-major flatten
#     y = B.T.reshape(N)
#     return y

# quick check
rng = np.random.default_rng(0)
N, t = 8, 4   # try other factorizations too (e.g., r=4, t=16)
r = N // t
x = rng.standard_normal(N) + 1j*rng.standard_normal(N)
ref = np.fft.fft(x)
# y = four_step_fft_incorrect(x, r, t)
y = six_step_fft(x, r, t)
# y = four_step_fft(x, r, t)
print(ref)
print(y)
