import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv('Concrete_Data.csv')

print(df.shape)

test_df = df.iloc[500:630].copy()
train_df = pd.concat([df.iloc[0:500], df.iloc[630:]]).copy()

print (test_df.shape)
print (train_df.shape)

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def deriv_mse(y, y_hat):
    return -2 * (y - y_hat) / len(y)


def r2(y, y_hat):
    return 1 - (
        mse(y, y_hat) / 
        np.mean((y - np.mean(y)) ** 2)
    )

def gradient_descent(X,y, starting_weights, starting_bias,learning_rate, iterations):
    weights = starting_weights.copy()
    b = starting_bias

    losses = []

    for i in range(iterations):
        y_hat = X @ weights + b
        loss = mse(y, y_hat)
        losses.append(loss)

        dL_dyhat = deriv_mse(y, y_hat)

        dw = X.T @ dL_dyhat
        db = np.sum(dL_dyhat)

        weights = weights - learning_rate * dw
        b = b - learning_rate * db
    return weights, b, losses


print(df.columns)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

starting_weights = np.zeros(X_train.shape[1])
starting_bias = 0

print(starting_weights)


weights, b, losses = gradient_descent(
    X_train,
    y_train,
    starting_weights,
    starting_bias,
    learning_rate=0.0000001,
    iterations=5000
)

y_train_hat = X_train @ weights + b

print("Weights:", weights)
print("Bias:", b)
print("Training MSE:", mse(y_train, y_train_hat))
print("Training R2:", r2(y_train, y_train_hat))

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Gradient Descent Loss - Untransformed Model")
plt.show()



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
starting_weights_scaled = np.zeros(X_train_scaled.shape[1])
starting_bias_scaled = 0

weights_scaled, b_scaled, losses_scaled = gradient_descent(
    X_train_scaled,
    y_train,
    starting_weights_scaled,
    starting_bias_scaled,
    learning_rate=0.01,
    iterations=5000
)

y_train_hat_scaled = X_train_scaled @ weights_scaled + b_scaled

print("Scaled Weights:", weights_scaled)
print("Scaled Bias:", b_scaled)
print("Scaled Training MSE:", mse(y_train, y_train_hat_scaled))
print("Scaled Training R2:", r2(y_train, y_train_hat_scaled))

plt.plot(losses, label="Untransformed")
plt.plot(losses_scaled, label="Standardized")

plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Gradient Descent Convergence: Untransformed (alpha = 0.0000001) vs. Standardized Models (alpha = 0.01)")
plt.legend()
plt.show()

#Q4
untransformed_optimal = LinearRegression()
untransformed_optimal.fit(X_train, y_train)

y_train_hat_optimal = untransformed_optimal.predict(X_train)

print("Untransformed Optimal MSE:",
      mse(y_train, y_train_hat_optimal))
print("Untransformed Optimal R2:",
      r2(y_train, y_train_hat_optimal))

scaled_optimal = LinearRegression()
scaled_optimal.fit(X_train_scaled, y_train)

y_train_hat_scaled_optimal = scaled_optimal.predict(X_train_scaled)

print("Scaled Optimal MSE:",
      mse(y_train, y_train_hat_scaled_optimal))
print("Scaled Optimal R2:",
      r2(y_train, y_train_hat_scaled_optimal))


#Q5
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

starting_weights_log = np.zeros(X_train_log.shape[1])
starting_bias_log = 0

weights_log, b_log, losses_log = gradient_descent(
    X_train_log,
    y_train,
    starting_weights_log,
    starting_bias_log,
    learning_rate=0.005,
    iterations=5000
)

y_train_hat_log = X_train_log @ weights_log + b_log

print("Log Weights:", weights_log)
print("Log Bias:", b_log)
print("Log Training MSE:", mse(y_train, y_train_hat_log))
print("Log Training R2:", r2(y_train, y_train_hat_log))

plt.plot(losses, label="Untransformed (α = 0.0000001)")
plt.plot(losses_scaled, label="Standardized (α = 0.01)")
plt.plot(losses_log, label="Log-Transformed (α = 0.005)")

plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Gradient Descent Convergence Across Feature Transformations")
plt.legend()
plt.show()

#Q5.3
log_optimal = LinearRegression()
log_optimal.fit(X_train_log, y_train)

log_train_pred = log_optimal.predict(X_train_log)
log_test_pred = log_optimal.predict(X_test_log)

print("Log Optimal Training MSE:", mse(y_train, log_train_pred))
print("Log Optimal Testing MSE:", mse(y_test, log_test_pred))
print("Log Optimal Training R2:", r2(y_train, log_train_pred))
print("Log Optimal Testing R2:", r2(y_test, log_test_pred))

untransformed_train_pred = untransformed_optimal.predict(X_train)
untransformed_test_pred = untransformed_optimal.predict(X_test)

scaled_train_pred = scaled_optimal.predict(X_train_scaled)
scaled_test_pred = scaled_optimal.predict(X_test_scaled)

print("Untransformed:")
print("Training MSE:", mse(y_train, untransformed_train_pred))
print("Testing MSE:", mse(y_test, untransformed_test_pred))
print("Training R2:", r2(y_train, untransformed_train_pred))
print("Testing R2:", r2(y_test, untransformed_test_pred))

print("\nStandardized:")
print("Training MSE:", mse(y_train, scaled_train_pred))
print("Testing MSE:", mse(y_test, scaled_test_pred))
print("Training R2:", r2(y_train, scaled_train_pred))
print("Testing R2:", r2(y_test, scaled_test_pred))

print("\nLog-Transformed:")
print("Training MSE:", mse(y_train, log_train_pred))
print("Testing MSE:", mse(y_test, log_test_pred))
print("Training R2:", r2(y_train, log_train_pred))
print("Testing R2:", r2(y_test, log_test_pred))

#Q6
print("")
print("Begin Q6")
univariate_results = []

for column in X_train.columns[:1]:
    X_uni = X_train[[column]]

    starting_weight = np.zeros(1)
    starting_bias = 0

    weights_uni, b_uni, losses_uni = gradient_descent(
        X_uni,
        y_train,
        starting_weight,
        starting_bias,
        learning_rate=0.0000001,
        iterations=5000
    )

    y_train_hat_uni = X_uni @ weights_uni + b_uni

    train_mse = mse(y_train, y_train_hat_uni)
    train_r2 = r2(y_train, y_train_hat_uni)

    univariate_results.append({
        "Feature": column,
        "Weight": weights_uni.iloc[0],
        "Bias": b_uni,
        "MSE": train_mse,
        "R2": train_r2
    })

    print(column)
    print("Weight:", weights_uni.iloc[0])
    print("Bias:", b_uni)
    print("MSE:", train_mse)
    print("R2:", train_r2)
    print()

for column in X_train.columns[1:4]:
    X_uni = X_train[[column]]

    starting_weight = np.zeros(1)
    starting_bias = 0

    weights_uni, b_uni, losses_uni = gradient_descent(
        X_uni,
        y_train,
        starting_weight,
        starting_bias,
        learning_rate=0.0000001,
        iterations=5000
    )

    y_train_hat_uni = X_uni @ weights_uni + b_uni

    train_mse = mse(y_train, y_train_hat_uni)
    train_r2 = r2(y_train, y_train_hat_uni)

    univariate_results.append({
        "Feature": column,
        "Weight": weights_uni.iloc[0],
        "Bias": b_uni,
        "MSE": train_mse,
        "R2": train_r2
    })

    print(column)
    print("Weight:", weights_uni.iloc[0])
    print("Bias:", b_uni)
    print("MSE:", train_mse)
    print("R2:", train_r2)
    print()

#ONE I CARE ABOUT
for column in X_train.columns[4:5]:
    X_uni = X_train[[column]]

    starting_weight = np.zeros(1)
    starting_bias = 0

    weights_uni, b_uni, losses_uni = gradient_descent(
        X_uni,
        y_train,
        starting_weight,
        starting_bias,
        learning_rate=0.0002,
        iterations=10000
    )

    y_train_hat_uni = X_uni @ weights_uni + b_uni

    train_mse = mse(y_train, y_train_hat_uni)
    train_r2 = r2(y_train, y_train_hat_uni)

    univariate_results.append({
        "Feature": column,
        "Weight": weights_uni.iloc[0],
        "Bias": b_uni,
        "MSE": train_mse,
        "R2": train_r2
    })

    print(column)
    print("Weight:", weights_uni.iloc[0])
    print("Bias:", b_uni)
    print("MSE:", train_mse)
    print("R2:", train_r2)
    print()

for column in X_train.columns[5:]:
    X_uni = X_train[[column]]

    starting_weight = np.zeros(1)
    starting_bias = 0

    weights_uni, b_uni, losses_uni = gradient_descent(
        X_uni,
        y_train,
        starting_weight,
        starting_bias,
        learning_rate=0.0000001,
        iterations=5000
    )

    y_train_hat_uni = X_uni @ weights_uni + b_uni

    train_mse = mse(y_train, y_train_hat_uni)
    train_r2 = r2(y_train, y_train_hat_uni)

    univariate_results.append({
        "Feature": column,
        "Weight": weights_uni.iloc[0],
        "Bias": b_uni,
        "MSE": train_mse,
        "R2": train_r2
    })

    print(column)
    print("Weight:", weights_uni.iloc[0])
    print("Bias:", b_uni)
    print("MSE:", train_mse)
    print("R2:", train_r2)
    print()


# Q6.2
optimal_univariate_results = []

for column in X_train.columns:
    X_uni = X_train[[column]]

    uni_optimal = LinearRegression()
    uni_optimal.fit(X_uni, y_train)

    univariate_coefficient = uni_optimal.coef_[0]

    optimal_univariate_results.append({
        "Feature": column,
        "Univariate Coefficient": univariate_coefficient
    })

optimal_univariate_df = pd.DataFrame(optimal_univariate_results)

optimal_univariate_df["Multivariate Coefficient"] = untransformed_optimal.coef_

print(optimal_univariate_df)


#Q7
print("")
print("Q7")
# Q7.1 - Untransformed model

X_train_sm = sm.add_constant(X_train)

untransformed_ols = sm.OLS(y_train, X_train_sm).fit()

print(untransformed_ols.pvalues)

# Standardized model

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_train_scaled_sm = sm.add_constant(X_train_scaled_df)

scaled_ols = sm.OLS(y_train, X_train_scaled_sm).fit()

print(scaled_ols.pvalues)

# Log-transformed model

X_train_log_sm = sm.add_constant(X_train_log)

log_ols = sm.OLS(y_train, X_train_log_sm).fit()

print(log_ols.pvalues)

pvalue_table = pd.DataFrame({
    "Feature": X_train.columns,
    "Untransformed p-value": untransformed_ols.pvalues.iloc[1:].values,
    "Standardized p-value": scaled_ols.pvalues.iloc[1:].values,
    "Log-Transformed p-value": log_ols.pvalues.iloc[1:].values
})

print(pvalue_table)

pvalue_table.to_excel("DrewRosenberg-DataMining-Project1-Part1-Q7.1.xlsx", index=False)



#Part 2
#Summary Stats
summary_stats = train_df.describe().loc[['mean', 'std', 'min', 'max']].T
print(summary_stats)
summary_stats.round(3).to_excel("Concrete_Descriptive_Statistics.xlsx")

#Age vs Strength Scatter Plot
plt.scatter(train_df['Age (day)'], train_df['Concrete compressive strength(MPa, megapascals) '])
plt.xlabel('Age (days)')
plt.ylabel('Concrete Compressive Strength (MPa)')
plt.title('Age vs Concrete Compressive Strength')
plt.show()
plt.savefig('Age_vs_Strength_Scatter_Plot.png') 
plt.close()

#Water vs Strength Scatter Plot
plt.scatter(train_df['Water  (component 4)(kg in a m^3 mixture)'], train_df['Concrete compressive strength(MPa, megapascals) '])
plt.xlabel('Water (kg/m^3)')
plt.ylabel('Concrete Compressive Strength (MPa)')
plt.title('Water vs Concrete Compressive Strength')
plt.savefig('Water_vs_Strength_Scatter_Plot.png') 
plt.show()
