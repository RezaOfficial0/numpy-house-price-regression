"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X = np.array(X, dtype=float).copy()
    rows = len(X)
    cols = len (X[0])
    for row in range(rows):
        for col in range(cols):

            if (np.isnan(X[row][col])):
                col_mean = np.nanmean(X[:, col])
                X[row][col] = col_mean
                if(np.isnan(col_mean)):
                    X[row][col] = 0
            

    return X

# Step 2 - compute_iqr_bounds
import numpy as np

def compute_iqr_bounds(X, k=1.5):
    X = np.asarray(X, dtype=float)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# Step 3 - clip_columns
import numpy as np

def clip_columns(X, lower, upper):
    X = np.asarray(X, dtype=float)
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
import numpy as np

def make_ratio_feature(numerator, denominator, eps=1e-8):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return numerator / (denominator + eps)

# Step 5 - append_column
import numpy as np
def append_column(X, col):

    x = X.copy()
    new_rows = []

    for i in range(len(x)):
        updated_row = np.append(x[i], col[i])
        new_rows.append(updated_row)



    
    return np.array(new_rows)

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels):
    unique_categories, unique_inverse = np.unique(labels, return_inverse=True)
    
    N = len(labels)
    C = len(unique_categories)
    one_hot = np.zeros((N, C), dtype=float)
    
    one_hot[np.arange(N), unique_inverse] = 1.0
    
    return one_hot

# Step 7 - fit_standardizer
import numpy as np

def fit_standardizer(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Force columns with 0.0 variance/std to have a std of 1.0
    std[std == 0.0] = 1.0
    
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
import numpy as np

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Prepends a column of ones to a 2-D feature matrix X.
    
    Parameters:
    X (np.ndarray): 2-D feature matrix of shape (N, F)
    
    Returns:
    np.ndarray: Modified feature matrix of shape (N, F+1) with ones in column 0.
    """
    N = X.shape[0]
    
    # Create a column vector of ones with shape (N, 1) matching X's data type
    bias = np.ones((N, 1), dtype=X.dtype)
    
    # Concatenate along the columns axis (axis=1)
    return np.hstack((bias, X))

# Step 10 - make_shuffled_indices
import numpy as np

def make_shuffled_indices(n_samples: int, seed: int) -> np.ndarray:
    """
    Creates a reproducibly shuffled permutation of row indices from 0 through n_samples-1.
    
    Parameters:
    n_samples (int): Total number of samples (length of index array).
    seed (int): Integer random seed for reproducibility.
    
    Returns:
    np.ndarray: 1-D integer array of shape (n_samples,) containing the shuffled indices.
    """
    # Instantiate an isolated random number generator using the provided seed
    rng = np.random.default_rng(seed)
    
    # Generate a shuffled permutation array from 0 to n_samples-1
    return rng.permutation(n_samples)

# Step 11 - partition_indices
import numpy as np

def partition_indices(indices: np.ndarray, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions an array of row indices into contiguous train, validation, and test slices.
    
    Parameters:
    indices (np.ndarray): 1-D array of row indices.
    train_ratio (float): Ratio for training subset (0, 1).
    val_ratio (float): Ratio for validation subset (0, 1).
    
    Returns:
    tuple: (train_idx, val_idx, test_idx) as 1-D integer NumPy arrays.
    """
    N = len(indices)
    
    # Calculate split boundaries using explicit flooring
    train_end = int(np.floor(N * train_ratio))
    val_end = train_end + int(np.floor(N * val_ratio))
    
    # Extract contiguous prefixes/slices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
import numpy as np

def subset_xy(X: np.ndarray, y: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Selects rows of feature matrix X and target vector y at the given indices.
    
    Parameters:
    X (np.ndarray): 2-D array of shape (N, F)
    y (np.ndarray): 1-D array of shape (N,) or 2-D array of shape (N, T)
    indices (np.ndarray): 1-D array of indices to extract
    
    Returns:
    tuple: (X_sub, y_sub) matching the row ordering specified in indices.
    """
    # NumPy fancy indexing extracts the exact rows in the order of indices
    X_sub = X[indices]
    y_sub = y[indices]
    
    return X_sub, y_sub

# Step 13 - ols_fit
import numpy as np

def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the OLS weight vector theta by solving the normal equations.
    
    Parameters:
    X (np.ndarray): Design matrix of shape (N, D) including the bias column.
    y (np.ndarray): Target vector of shape (N,).
    
    Returns:
    np.ndarray: Weight vector theta of shape (D,).
    """
    # Form the components of the normal equations: (X^T * X) * theta = X^T * y
    XT = X.T
    A = XT @ X
    b = XT @ y
    
    # Solve the system of linear equations directly for numerical stability
    theta = np.linalg.solve(A, b)
    
    return theta

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

