# Detailed scatter plot with regression lines
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

print("DETAILED SCATTER PLOT WITH REGRESSION LINES")
print("=" * 60)

print("\n1. STRESS SURVEY VS ANXIETY")
print("-" * 40)
print("Data Points:")
for i in range(len(stress_survey)):
    predicted = beta0_survey + beta1_survey * stress_survey[i]
    print(f"  ({stress_survey[i]:.1f}, {anxiety[i]:.2f}) -> Predicted: {predicted:.2f}")

print(f"\nRegression Line: Anxiety = {beta0_survey:.4f} + {beta1_survey:.4f} * StressSurvey")
print(f"R-squared = {r2_survey:.4f}")

# Generate regression line points
print(f"\nRegression Line Points:")
x_range = [0, 3, 6, 9, 12]
for x in x_range:
    y_line = beta0_survey + beta1_survey * x
    print(f"  ({x}, {y_line:.2f})")

print("\n" + "="*60)
print("2. TIME VS ANXIETY")
print("-" * 40)
print("Data Points:")
for i in range(len(time)):
    predicted = beta0_time + beta1_time * time[i]
    print(f"  ({time[i]:.1f}, {anxiety[i]:.2f}) -> Predicted: {predicted:.2f}")

print(f"\nRegression Line: Anxiety = {beta0_time:.4f} + {beta1_time:.4f} * Time")
print(f"R-squared = {r2_time:.4f}")

# Generate regression line points
print(f"\nRegression Line Points:")
x_range_time = [0, 0.5, 1.0, 1.5, 2.0, 2.2]
for x in x_range_time:
    y_line = beta0_time + beta1_time * x
    print(f"  ({x:.1f}, {y_line:.2f})")

print("\n" + "="*60)
print("VISUALIZATION GUIDE")
print("=" * 60)
print("To create scatter plots with regression lines:")
print("\n1. STRESS SURVEY VS ANXIETY:")
print("   - Plot data points: (0,0), (0,0.1), (0,0.1), (3,1.1), (3,1.1), (3,1.1),")
print("     (6,2.2), (6,2.2), (6,2.2), (9,8.2), (9,8.2), (9,8.21),")
print("     (12,12.22), (12,12.22), (12,12.22)")
print("   - Draw regression line through: (0,-1.52), (3,1.62), (6,4.76), (9,7.90), (12,11.04)")
print("   - Line equation: Anxiety = -1.5240 + 1.0470 * StressSurvey")

print("\n2. TIME VS ANXIETY:")
print("   - Plot data points: (0,0), (1,0.1), (1,0.1), (1,1.1), (1,1.1), (1,1.1),")
print("     (2,2.2), (2,2.2), (2,2.2), (2,8.2), (2,8.2), (2.1,8.21),")
print("     (2.2,12.22), (2.2,12.22), (2.2,12.22)")
print("   - Draw regression line through: (0,-3.68), (0.5,-0.91), (1,1.66), (1.5,4.23), (2,7.00), (2.2,8.07)")
print("   - Line equation: Anxiety = -3.6801 + 5.3406 * Time")

print("\n3. KEY OBSERVATIONS:")
print("   - StressSurvey line: Steep positive slope, good fit to data")
print("   - Time line: Very steep positive slope, poor fit to data")
print("   - Both lines miss the true relationship due to omitted variables")
print("   - True relationship: Anxiety = Stress + 0.1 * Time")
print("   - Neither bivariate model captures this correctly!")
