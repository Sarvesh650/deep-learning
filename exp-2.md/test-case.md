from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd

# Fashion-MNIST class names
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=5, batch_size=32,
          validation_data=(X_test, y_test_cat), verbose=2)

# Pick some sample test indices (just for demonstration)
sample_indices = [0, 1, 2, 3]  # You can change these
X_sample = X_test[sample_indices]
y_sample_true = y_test[sample_indices]

# Predict
predictions = model.predict(X_sample)
y_sample_pred = np.argmax(predictions, axis=1)

# Create results table
data = {
    "Input Image": [class_names[i] for i in y_sample_true],
    "True Label": [class_names[i] for i in y_sample_true],
    "Predicted Label": [class_names[i] for i in y_sample_pred],
    "Correct (Y/N)": ["Y" if t == p else "N" for t, p in zip(y_sample_true, y_sample_pred)]
}

df = pd.DataFrame(data)

# Display results
print("\nSample Predictions:\n")
print(df)

# Calculate accuracy for sample
sample_acc = (df["Correct (Y/N)"] == "Y").mean() * 100
print(f"\nSample Accuracy: {sample_acc:.2f}%")
output:
<img width="924" height="382" alt="image" src="https://github.com/user-attachments/assets/bc851234-7498-496e-acf9-707e29b7bcc7" />
