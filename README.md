# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
## AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

## ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
## PROGRAM:
```
# Name: JEEVANESH
# Reg No: 21222243002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CarDekho dataset
file_path = "/content/cardekho.csv"   # <-- update file path
data = pd.read_csv(file_path)

# Check dataset columns
print(data.head())

# Assuming dataset has columns: "year" and "selling_price"
# Aggregate yearly sales (sum of selling_price per year)
yearly_sales = data.groupby("year")["selling_price"].sum().reset_index()

# Rename columns to fit trend estimation format
yearly_sales.rename(columns={"year": "Year", "selling_price": "Sales"}, inplace=True)

# Extract values
years = yearly_sales["Year"].tolist()
sales = yearly_sales["Sales"].tolist()

# Prepare values for linear regression
X = [i - (len(years) // 2) for i in range(len(years))]
x2 = [i**2 for i in X]
xy = [i * j for i, j in zip(X, sales)]

n = len(years)
b = (n * sum(xy) - sum(sales) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(sales) - b * sum(X)) / n
linear_trend = [a + b * Xi for Xi in X]

# Polynomial Trend Estimation (Degree 2)
x3 = [i**3 for i in X]
x4 = [i**4 for i in X]
x2y = [i * j for i, j in zip(x2, sales)]

coeff = [[n, sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(sales), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * Xi + c_poly * (Xi**2) for Xi in X]

# Display trend equations
print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Plot results
plt.figure(figsize=(12,6))
plt.plot(years, sales, 'bo-', label="Actual Sales")
plt.plot(years, linear_trend, 'k--', label="Linear Trend")
plt.plot(years, poly_trend, 'r-', label="Polynomial Trend")
plt.title("CarDekho Sales Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Total Sales (Selling Price Sum)")
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
<img width="1546" height="382" alt="image" src="https://github.com/user-attachments/assets/2c0ace21-955c-4370-84e3-5e08119a1462" />

<img width="1012" height="546" alt="image" src="https://github.com/user-attachments/assets/b007d6a7-9764-4543-be70-739072e488d1" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
