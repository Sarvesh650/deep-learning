code:
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Perceptron 
X = np.array([[0,0],[0,1],[1,0],[1,1]]) 
y = np.array([0,1,1,0]) 
clf = Perceptron(tol=1e-3, random_state=0) 
clf.fit(X, y) 
print("Predictions:", clf.predict(X)) 
# Plotting 
for i in range(len(X)): 
if y[i] == 0: 
plt.scatter(X[i][0], X[i][1], color='red') 
else: 
plt.scatter(X[i][0], X[i][1], color='blue') 
# Decision boundary 
x_values = [0, 1] 
y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1] 
plt.plot(x_values, y_values) 
plt.title('Perceptron Decision Boundary for XOR') 
plt.show() 
output:
<img width="733" height="590" alt="image" src="https://github.com/user-attachments/assets/7b555fd2-216d-4133-bca3-3872e147a6bd" />
