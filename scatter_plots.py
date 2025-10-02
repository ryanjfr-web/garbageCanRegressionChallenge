# Scatter plots for regression analysis
import math

# Data
data = {
    'Stress': [0,0,0,1,1,1,2,2,2,8,8,8,12,12,12],
    'StressSurvey': [0,0,0,3,3,3,6,6,6,9,9,9,12,12,12],
    'Time': [0,1,1,1,1,1,2,2,2,2,2,2.1,2.2,2.2,2.2],
    'Anxiety': [0,0.1,0.1,1.1,1.1,1.1,2.2,2.2,2.2,8.2,8.2,8.21,12.22,12.22,12.22]
}

def simple_linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] * x[i] for i in range(n))
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    y_mean = sum_y / n
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    ss_res = sum((y[i] - (intercept + slope * x[i])) ** 2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot)
    
    return intercept, slope, r_squared

# Calculate regression lines
stress_survey = data['StressSurvey']
time = data['Time']
anxiety = data['Anxiety']

# StressSurvey vs Anxiety
beta0_survey, beta1_survey, r2_survey = simple_linear_regression(stress_survey, anxiety)

# Time vs Anxiety  
beta0_time, beta1_time, r2_time = simple_linear_regression(time, anxiety)

print("SCATTER PLOT DATA FOR STRESS SURVEY VS ANXIETY")
print("=" * 50)
print("X (StressSurvey)\tY (Anxiety)\tPredicted Y")
print("-" * 50)

# Generate regression line points
x_min, x_max = min(stress_survey), max(stress_survey)
for i in range(len(stress_survey)):
    predicted = beta0_survey + beta1_survey * stress_survey[i]
    print(f"{stress_survey[i]:.1f}\t\t\t{anxiety[i]:.2f}\t\t{predicted:.2f}")

print(f"\nRegression Line: Anxiety = {beta0_survey:.4f} + {beta1_survey:.4f} * StressSurvey")
print(f"R-squared = {r2_survey:.4f}")

print("\n" + "="*60)
print("SCATTER PLOT DATA FOR TIME VS ANXIETY")
print("=" * 50)
print("X (Time)\t\tY (Anxiety)\tPredicted Y")
print("-" * 50)

# Generate regression line points
for i in range(len(time)):
    predicted = beta0_time + beta1_time * time[i]
    print(f"{time[i]:.1f}\t\t\t{anxiety[i]:.2f}\t\t{predicted:.2f}")

print(f"\nRegression Line: Anxiety = {beta0_time:.4f} + {beta1_time:.4f} * Time")
print(f"R-squared = {r2_time:.4f}")

print("\n" + "="*60)
print("VISUALIZATION INSTRUCTIONS:")
print("=" * 60)
print("1. STRESS SURVEY VS ANXIETY SCATTER PLOT:")
print("   - X-axis: Stress Survey Score (0 to 12)")
print("   - Y-axis: Anxiety Level (0 to 12.22)")
print("   - Plot points: (0,0), (0,0.1), (0,0.1), (3,1.1), (3,1.1), (3,1.1),")
print("     (6,2.2), (6,2.2), (6,2.2), (9,8.2), (9,8.2), (9,8.21),")
print("     (12,12.22), (12,12.22), (12,12.22)")
print("   - Regression line: Anxiety = -1.5240 + 1.0470 * StressSurvey")
print("   - R² = 0.9011 (very good fit)")
print("   - Comment: Strong positive relationship, but coefficient is wrong!")

print("\n2. TIME VS ANXIETY SCATTER PLOT:")
print("   - X-axis: Time (0 to 2.2)")
print("   - Y-axis: Anxiety Level (0 to 12.22)")
print("   - Plot points: (0,0), (1,0.1), (1,0.1), (1,1.1), (1,1.1), (1,1.1),")
print("     (2,2.2), (2,2.2), (2,2.2), (2,8.2), (2,8.2), (2.1,8.21),")
print("     (2.2,12.22), (2.2,12.22), (2.2,12.22)")
print("   - Regression line: Anxiety = -3.6801 + 5.3406 * Time")
print("   - R² = 0.5630 (moderate fit)")
print("   - Comment: Poor fit! Time coefficient is WAY OFF (5.34 vs 0.1)")

print("\n3. ANALYSIS:")
print("   - StressSurvey shows strong correlation but wrong coefficient")
print("   - Time alone gives terrible coefficient estimate")
print("   - This demonstrates omitted variable bias!")
print("   - Need both variables together for correct coefficients")
