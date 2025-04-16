"""
COMPREHENSIVE NUMPY REFERENCE
Covering all 10 major topics with complete function examples
"""

import numpy as np

# ==============================================
# 1. BASICS OF NUMPY
# ==============================================
print("\n=== 1. Basics of NumPy ===\n")

# Array creation from lists
arr1d = np.array([1, 2, 3, 4])  # 1D array
arr2d = np.array([[1, 2], [3, 4]])  # 2D array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D array

# Special arrays
zeros = np.zeros((2, 3))  # Array of zeros
ones = np.ones((2, 2))  # Array of ones
identity = np.eye(3)  # Identity matrix
empty = np.empty((2, 2))  # Uninitialized array

# Array attributes
print("Array shape:", arr2d.shape)  # (2, 2)
print("Number of dimensions:", arr2d.ndim)  # 2
print("Number of elements:", arr2d.size)  # 4
print("Data type:", arr2d.dtype)  # int64
print("Item size (bytes):", arr2d.itemsize)  # 8

# Constants
print("Pi:", np.pi)
print("Euler's number:", np.e)
print("Infinity:", np.inf)
print("Not a Number:", np.nan)

# ==============================================
# 2. ARRAY CREATION
# ==============================================
print("\n=== 2. Array Creation ===\n")

# Sequence generation
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
lin_arr = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
log_arr = np.logspace(0, 2, 3)  # [1., 10., 100.]

# Special matrices
diag_matrix = np.diag([1, 2, 3])  # Diagonal matrix
triu = np.triu([[1, 2], [3, 4]])  # Upper triangular
tril = np.tril([[1, 2], [3, 4]])  # Lower triangular

# Random arrays
rand_uniform = np.random.rand(2, 2)  # Uniform distribution [0,1)
rand_normal = np.random.randn(2, 2)  # Standard normal
rand_int = np.random.randint(0, 10, (3,))  # Random integers
rand_choice = np.random.choice([1, 2, 3], size=5)  # Random selection

# Copying arrays
arr_copy = np.copy(arr1d)  # Deep copy
arr_view = arr1d.view()  # View (shares memory)

# ==============================================
# 3. ARRAY MANIPULATION
# ==============================================
print("\n=== 3. Array Manipulation ===\n")

# Reshaping
reshaped = np.arange(6).reshape(2, 3)  # 2x3 array
flattened = reshaped.flatten()  # 1D copy
raveled = reshaped.ravel()  # 1D view if possible

# Transposing
transposed = reshaped.T  # Transpose
swapped = np.swapaxes(reshaped, 0, 1)  # Swap axes

# Concatenation
a = np.array([1, 2])
b = np.array([3, 4])
vstack = np.vstack((a, b))  # Vertical stack
hstack = np.hstack((a.reshape(2,1), b.reshape(2,1)))  # Horizontal stack
stacked = np.stack((a, b))  # New axis stack

# Splitting
arr = np.arange(9).reshape(3, 3)
split_arr = np.split(arr, [1])  # Split at index
vsplit = np.vsplit(arr, 3)  # Vertical split
hsplit = np.hsplit(arr, 3)  # Horizontal split

# Adding/removing elements
appended = np.append(arr, [10, 11, 12])  # Append
inserted = np.insert(arr, 1, [99, 99, 99])  # Insert
deleted = np.delete(arr, 1, axis=0)  # Delete row

# Sorting
sorted_arr = np.sort([3, 1, 2])  # [1, 2, 3]
argsorted = np.argsort([3, 1, 2])  # [1, 2, 0] (indices)

# ==============================================
# 4. ARRAY INDEXING & SLICING
# ==============================================
print("\n=== 4. Array Indexing & Slicing ===\n")

arr = np.arange(10, 60, 10)  # [10, 20, 30, 40, 50]

# Basic indexing
print(arr[0])  # 10 (first element)
print(arr[-1])  # 50 (last element)

# Slicing
print(arr[1:4])  # [20, 30, 40]
print(arr[::2])  # [10, 30, 50] (every 2nd)
print(arr[::-1])  # [50, 40, 30, 20, 10] (reverse)

# Boolean indexing
mask = arr > 25
print(arr[mask])  # [30, 40, 50]

# Fancy indexing
indices = [0, 2, 4]
print(arr[indices])  # [10, 30, 50]

# 2D indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[1, 2])  # 6 (row 1, column 2)
print(arr2d[:, 1])  # [2, 5, 8] (all rows, column 1)

# ==============================================
# 5. BROADCASTING
# ==============================================
print("\n=== 5. Broadcasting ===\n")

# Scalar broadcasting
matrix = np.array([[1, 2], [3, 4]])
print(matrix * 10)  # Broadcasts 10 to all elements

# Vector broadcasting
vector = np.array([10, 20])
print(matrix + vector)  # Broadcasts vector to each row

# 3D broadcasting
arr3d = np.ones((3, 1, 5))
arr2d = np.ones((2, 5))
print((arr3d + arr2d).shape)  # (3, 2, 5)

# ==============================================
# 6. UNIVERSAL FUNCTIONS (UFUNCS)
# ==============================================
print("\n=== 6. Universal Functions (ufuncs) ===\n")

# Math operations
arr = np.array([1.0, 4.0, 9.0])
print(np.sqrt(arr))  # [1., 2., 3.]
print(np.exp(arr))  # Exponentials
print(np.log(arr))  # Natural log
print(np.power(arr, 2))  # [1., 16., 81.]

# Trigonometric
angles = np.array([0, np.pi/2, np.pi])
print(np.sin(angles))  # [0., 1., 0.]

# Comparison
print(np.greater(arr, 2))  # [False, True, True]
print(np.equal(arr, [1, 4, 8]))  # [True, True, False]

# Aggregation
print(np.sum(arr))  # 14.0
print(np.mean(arr))  # 4.666...
print(np.max(arr))  # 9.0
print(np.std(arr))  # Standard deviation

# ==============================================
# 7. LINEAR ALGEBRA
# ==============================================
print("\n=== 7. Linear Algebra ===\n")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix operations
print("Matrix multiplication:")
print(np.dot(A, B))  # Traditional dot product
print(A @ B)  # Python 3.5+ matrix operator

# Decompositions
print("\nMatrix inverse:")
print(np.linalg.inv(A))  # Inverse

print("\nEigenvalues/eigenvectors:")
eigenvals, eigenvecs = np.linalg.eig(A)
print("Eigenvalues:", eigenvals)
print("Eigenvectors:", eigenvecs)

print("\nSingular Value Decomposition:")
U, S, Vt = np.linalg.svd(A)
print("U:", U)
print("S:", S)
print("Vt:", Vt)

# Solving linear systems
print("\nSolving Ax = b:")
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print("Solution x:", x)  # [1., 2.]

# Norms and determinants
print("\nMatrix norm:", np.linalg.norm(A))
print("Determinant:", np.linalg.det(A))
print("Trace:", np.trace(A))  # Sum of diagonal

# ==============================================
# 8. INPUT/OUTPUT (I/O)
# ==============================================
print("\n=== 8. Input/Output (I/O) ===\n")

# Binary files
np.save('saved_array.npy', A)
loaded = np.load('saved_array.npy')
print("Loaded from .npy:", loaded)

# Multiple arrays
np.savez('multiple_arrays.npz', arr1=A, arr2=B)
with np.load('multiple_arrays.npz') as data:
    print("Loaded from .npz:", data['arr1'])

# Text files
np.savetxt('array.txt', A, fmt='%d')
loaded_txt = np.loadtxt('array.txt')
print("Loaded from text:", loaded_txt)

# Memory mapping
mmap = np.memmap('mmap.dat', dtype='float32', mode='w+', shape=(2,2))
mmap[:] = A  # Write data
del mmap  # Flush to disk

# ==============================================
# 9. ADVANCED TOPICS
# ==============================================
print("\n=== 9. Advanced Topics ===\n")

# Structured arrays
dtype = [('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
people = np.array([('Alice', 25, 55.5), ('Bob', 30, 85.2)], dtype=dtype)
print("Structured array:", people)
print("Names:", people['name'])

# Masked arrays
data = np.array([1, 2, -999, 4])
masked = np.ma.masked_where(data == -999, data)
print("Masked array:", masked)
print("Mean (ignoring masked):", masked.mean())

# Vectorization
def my_func(x, y):
    return x * 2 + y

vectorized_func = np.vectorize(my_func)
print("Vectorized function:", vectorized_func([1, 2, 3], 1))  # [3, 5, 7]

# Einsum (Einstein summation)
print("Matrix diagonal via einsum:", np.einsum('ii->i', A))  # [1, 4]

# ==============================================
# 10. APPLICATIONS
# ==============================================
print("\n=== 10. Applications ===")

# Signal processing
print("\nSignal Processing Example:")
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t)  # 5Hz sine wave
noise = 0.5 * np.random.randn(1000)
filtered = np.convolve(signal + noise, np.ones(10)/10, mode='same')

# Image processing
print("\nImage Processing Example:")
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening
# filtered_image = scipy.signal.convolve2d(image, kernel)  # Requires scipy

# Machine Learning
print("\nML Preprocessing Example:")
data = np.random.randn(100, 3)
normalized = (data - data.mean(axis=0)) / data.std(axis=0)
one_hot = np.eye(3)[np.array([0, 1, 2, 0])]  # One-hot encoding

# Finance
print("\nFinancial Example:")
returns = np.random.randn(1000) * 0.01  # Daily returns
cumulative = np.cumprod(1 + returns)  # Cumulative product
moving_avg = np.convolve(returns, np.ones(50)/50, mode='valid')

print("\nAll examples completed successfully!")
