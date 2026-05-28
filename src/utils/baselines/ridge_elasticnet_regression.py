"""
Ridge and ElasticNet regression helper functions with pre-extracted features.
Works with features that have already been extracted and scaled if needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def convert_groups_to_numeric(groups):
    """Convert group labels (file paths) to numeric indices"""
    unique_groups = np.unique(groups)
    group_mapping = {g: i for i, g in enumerate(unique_groups)}
    return np.array([group_mapping[g] for g in groups])


def train_ridge_regression(X_train, y_train, groups_train=None, alpha_candidates=None):
    """
    Train Ridge regression with optional GroupKFold CV.
    
    Parameters:
    -----------
    X_train : array-like, shape (n_samples, n_features)
        Training features (should already be scaled if needed)
    y_train : array-like, shape (n_samples,)
        Training labels
    groups_train : array-like or None, shape (n_samples,)
        Group labels for GroupKFold CV. If None, uses standard KFold.
    alpha_candidates : list or None
        Alpha values to search. Default: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    Returns:
    --------
    best_model : Ridge
        Best Ridge model
    best_params : dict
        Best parameters found
    cv_score : float
        Best CV R² score
    """
    if alpha_candidates is None:
        alpha_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    ridge_params = {'alpha': alpha_candidates}
    
    if groups_train is not None:
        # Convert groups to numeric if needed
        if isinstance(groups_train[0], (str, np.str_)):
            groups_numeric = convert_groups_to_numeric(groups_train)
        else:
            groups_numeric = groups_train
        
        cv = GroupKFold(n_splits=5)
        ridge_cv = GridSearchCV(
            Ridge(),
            ridge_params,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        ridge_cv.fit(X_train, y_train, groups=groups_numeric)
    else:
        ridge_cv = GridSearchCV(
            Ridge(),
            ridge_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        ridge_cv.fit(X_train, y_train)
    
    return ridge_cv.best_estimator_, ridge_cv.best_params_, ridge_cv.best_score_


def train_elasticnet_regression(X_train, y_train, groups_train=None, 
                                alpha_candidates=None, l1_ratio_candidates=None):
    """
    Train ElasticNet regression with optional GroupKFold CV.
    
    Parameters:
    -----------
    X_train : array-like, shape (n_samples, n_features)
        Training features (should already be scaled if needed)
    y_train : array-like, shape (n_samples,)
        Training labels
    groups_train : array-like or None, shape (n_samples,)
        Group labels for GroupKFold CV. If None, uses standard KFold.
    alpha_candidates : list or None
        Alpha values to search. Default: [0.0001, 0.001, 0.01, 0.1, 1.0]
    l1_ratio_candidates : list or None
        L1 ratio values to search. Default: [0.1, 0.3, 0.5, 0.7, 0.9]
    
    Returns:
    --------
    best_model : ElasticNet
        Best ElasticNet model
    best_params : dict
        Best parameters found
    cv_score : float
        Best CV R² score
    """
    if alpha_candidates is None:
        alpha_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    if l1_ratio_candidates is None:
        l1_ratio_candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    elasticnet_params = {
        'alpha': alpha_candidates,
        'l1_ratio': l1_ratio_candidates
    }
    
    if groups_train is not None:
        # Convert groups to numeric if needed
        if isinstance(groups_train[0], (str, np.str_)):
            groups_numeric = convert_groups_to_numeric(groups_train)
        else:
            groups_numeric = groups_train
        
        cv = GroupKFold(n_splits=5)
        elasticnet_cv = GridSearchCV(
            ElasticNet(max_iter=10000, random_state=42),
            elasticnet_params,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        elasticnet_cv.fit(X_train, y_train, groups=groups_numeric)
    else:
        elasticnet_cv = GridSearchCV(
            ElasticNet(max_iter=10000, random_state=42),
            elasticnet_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        elasticnet_cv.fit(X_train, y_train)
    
    return elasticnet_cv.best_estimator_, elasticnet_cv.best_params_, elasticnet_cv.best_score_


def evaluate_models(y_true, y_pred_ridge, y_pred_elasticnet):
    """
    Evaluate and compare Ridge and ElasticNet predictions.
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with evaluation metrics for both models
    """
    ridge_r2 = r2_score(y_true, y_pred_ridge)
    ridge_rmse = np.sqrt(mean_squared_error(y_true, y_pred_ridge))
    ridge_mae = mean_absolute_error(y_true, y_pred_ridge)
    
    elasticnet_r2 = r2_score(y_true, y_pred_elasticnet)
    elasticnet_rmse = np.sqrt(mean_squared_error(y_true, y_pred_elasticnet))
    elasticnet_mae = mean_absolute_error(y_true, y_pred_elasticnet)
    
    results_df = pd.DataFrame({
        'Model': ['Ridge', 'ElasticNet'],
        'R2': [ridge_r2, elasticnet_r2],
        'RMSE': [ridge_rmse, elasticnet_rmse],
        'MAE': [ridge_mae, elasticnet_mae]
    })
    
    return results_df


def plot_predictions(y_val, y_pred_ridge, y_test, y_pred_test_ridge,
                     y_pred_elasticnet_val, y_pred_elasticnet_test,
                     figsize=(18, 10)):
    """
    Create comparison plots for Ridge and ElasticNet predictions.
    
    Returns:
    --------
    fig : matplotlib figure
    """
    # Calculate metrics for titles
    ridge_val_r2 = r2_score(y_val, y_pred_ridge)
    ridge_val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
    ridge_test_r2 = r2_score(y_test, y_pred_test_ridge)
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_ridge))
    
    elasticnet_val_r2 = r2_score(y_val, y_pred_elasticnet_val)
    elasticnet_val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_elasticnet_val))
    elasticnet_test_r2 = r2_score(y_test, y_pred_elasticnet_test)
    elasticnet_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_elasticnet_test))
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Ridge validation
    axes[0, 0].scatter(y_val, y_pred_ridge, alpha=0.6, edgecolors='k')
    axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Temperature [°C]')
    axes[0, 0].set_ylabel('Predicted Temperature [°C]')
    axes[0, 0].set_title(f'Ridge - Validation\nR² = {ridge_val_r2:.4f}, RMSE = {ridge_val_rmse:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Ridge test
    axes[0, 1].scatter(y_test, y_pred_test_ridge, alpha=0.6, edgecolors='k', color='orange')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Temperature [°C]')
    axes[0, 1].set_ylabel('Predicted Temperature [°C]')
    axes[0, 1].set_title(f'Ridge - Test\nR² = {ridge_test_r2:.4f}, RMSE = {ridge_test_rmse:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ridge residuals
    residuals_ridge_val = y_val - y_pred_ridge
    axes[0, 2].hist(residuals_ridge_val, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Residuals [°C]')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Ridge - Validation Residuals\nMean = {residuals_ridge_val.mean():.4f}, Std = {residuals_ridge_val.std():.4f}')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # ElasticNet validation
    axes[1, 0].scatter(y_val, y_pred_elasticnet_val, alpha=0.6, edgecolors='k', color='green')
    axes[1, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Temperature [°C]')
    axes[1, 0].set_ylabel('Predicted Temperature [°C]')
    axes[1, 0].set_title(f'ElasticNet - Validation\nR² = {elasticnet_val_r2:.4f}, RMSE = {elasticnet_val_rmse:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ElasticNet test
    axes[1, 1].scatter(y_test, y_pred_elasticnet_test, alpha=0.6, edgecolors='k', color='green')
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('True Temperature [°C]')
    axes[1, 1].set_ylabel('Predicted Temperature [°C]')
    axes[1, 1].set_title(f'ElasticNet - Test\nR² = {elasticnet_test_r2:.4f}, RMSE = {elasticnet_test_rmse:.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # ElasticNet residuals
    residuals_elasticnet_val = y_val - y_pred_elasticnet_val
    axes[1, 2].hist(residuals_elasticnet_val, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 2].set_xlabel('Residuals [°C]')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title(f'ElasticNet - Validation Residuals\nMean = {residuals_elasticnet_val.mean():.4f}, Std = {residuals_elasticnet_val.std():.4f}')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig
