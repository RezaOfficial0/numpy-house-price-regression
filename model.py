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

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

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

