# Run this Python script to get your regression results
import math

# Data with known true relationships: Anxiety = Stress + 0.1 × Time
data = {
    'Stress': [0,0,0,1,1,1,2,2,2,8,8,8,12,12,12],
    'StressSurvey': [0,0,0,3,3,3,6,6,6,9,9,9,12,12,12],
    'Time': [0,1,1,1,1,1,2,2,2,2,2,2.1,2.2,2.2,2.2],
    'Anxiety': [0,0.1,0.1,1.1,1.1,1.1,2.2,2.2,2.2,8.2,8.2,8.21,12.22,12.22,12.22]
}

def simple_linear_regression(x, y):
    """Calculate simple linear regression coefficients"""
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] * x[i] for i in range(n))
    
    # Calculate slope (β1) and intercept (β0)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate R-squared
    y_mean = sum_y / n
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    ss_res = sum((y[i] - (intercept + slope * x[i])) ** 2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot)
    
    return intercept, slope, r_squared

def multiple_regression(x1, x2, y):
    """Calculate multiple regression coefficients using normal equations"""
    n = len(y)
    
    # Create matrices for normal equations: X'X * β = X'y
    # For model: y = β₀ + β₁*x₁ + β₂*x₂
    
    # X matrix (with intercept column)
    X = [[1, x1[i], x2[i]] for i in range(n)]
    
    # Calculate X'X
    XtX = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            XtX[i][j] = sum(X[k][i] * X[k][j] for k in range(n))
    
    # Calculate X'y
    Xty = [0, 0, 0]
    for i in range(3):
        Xty[i] = sum(X[k][i] * y[k] for k in range(n))
    
    # Solve using Gaussian elimination (simplified for 3x3)
    # β = (X'X)^(-1) * X'y
    
    # Calculate determinant
    det = (XtX[0][0] * (XtX[1][1] * XtX[2][2] - XtX[1][2] * XtX[2][1]) -
           XtX[0][1] * (XtX[1][0] * XtX[2][2] - XtX[1][2] * XtX[2][0]) +
           XtX[0][2] * (XtX[1][0] * XtX[2][1] - XtX[1][1] * XtX[2][0]))
    
    if abs(det) < 1e-10:
        return None, None, None, 0
    
    # Calculate inverse matrix (simplified for 3x3)
    inv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    inv[0][0] = (XtX[1][1] * XtX[2][2] - XtX[1][2] * XtX[2][1]) / det
    inv[0][1] = -(XtX[0][1] * XtX[2][2] - XtX[0][2] * XtX[2][1]) / det
    inv[0][2] = (XtX[0][1] * XtX[1][2] - XtX[0][2] * XtX[1][1]) / det
    inv[1][0] = -(XtX[1][0] * XtX[2][2] - XtX[1][2] * XtX[2][0]) / det
    inv[1][1] = (XtX[0][0] * XtX[2][2] - XtX[0][2] * XtX[2][0]) / det
    inv[1][2] = -(XtX[0][0] * XtX[1][2] - XtX[0][2] * XtX[1][0]) / det
    inv[2][0] = (XtX[1][0] * XtX[2][1] - XtX[1][1] * XtX[2][0]) / det
    inv[2][1] = -(XtX[0][0] * XtX[2][1] - XtX[0][1] * XtX[2][0]) / det
    inv[2][2] = (XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0]) / det
    
    # Calculate coefficients
    beta0 = sum(inv[0][i] * Xty[i] for i in range(3))
    beta1 = sum(inv[1][i] * Xty[i] for i in range(3))
    beta2 = sum(inv[2][i] * Xty[i] for i in range(3))
    
    # Calculate R-squared
    y_mean = sum(y) / n
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    ss_res = sum((y[i] - (beta0 + beta1 * x1[i] + beta2 * x2[i])) ** 2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot)
    
    return beta0, beta1, beta2, r_squared

# Run all analyses
stress_survey = data['StressSurvey']
time = data['Time']
stress = data['Stress']
anxiety = data['Anxiety']

print("GARBAGE CAN REGRESSION CHALLENGE RESULTS")
print("=" * 50)

# 1. Bivariate: Anxiety ~ StressSurvey
beta0_survey, beta1_survey, r2_survey = simple_linear_regression(stress_survey, anxiety)
print(f"\n1. BIVARIATE: Anxiety ~ StressSurvey")
print(f"   Beta0 = {beta0_survey:.4f}, Beta1 = {beta1_survey:.4f}, R2 = {r2_survey:.4f}")

# 2. Bivariate: Anxiety ~ Time
beta0_time, beta1_time, r2_time = simple_linear_regression(time, anxiety)
print(f"\n2. BIVARIATE: Anxiety ~ Time")
print(f"   Beta0 = {beta0_time:.4f}, Beta2 = {beta1_time:.4f}, R2 = {r2_time:.4f}")

# 3. Multiple: Anxiety ~ StressSurvey + Time
beta0_mult_survey, beta1_mult_survey, beta2_mult_survey, r2_mult_survey = multiple_regression(stress_survey, time, anxiety)
print(f"\n3. MULTIPLE: Anxiety ~ StressSurvey + Time")
print(f"   Beta0 = {beta0_mult_survey:.4f}, Beta1 = {beta1_mult_survey:.4f}, Beta2 = {beta2_mult_survey:.4f}, R2 = {r2_mult_survey:.4f}")

# 4. Multiple: Anxiety ~ Stress + Time (TRUE MODEL)
beta0_mult_stress, beta1_mult_stress, beta2_mult_stress, r2_mult_stress = multiple_regression(stress, time, anxiety)
print(f"\n4. MULTIPLE: Anxiety ~ Stress + Time (TRUE MODEL)")
print(f"   Beta0 = {beta0_mult_stress:.4f}, Beta1 = {beta1_mult_stress:.4f}, Beta2 = {beta2_mult_stress:.4f}, R2 = {r2_mult_stress:.4f}")

print(f"\nTRUE VALUES: Beta0 = 0, Beta1 = 1, Beta2 = 0.1")
print(f"\nPERFECT MATCH: Model 4 (Stress + Time) gives exact true coefficients!")
