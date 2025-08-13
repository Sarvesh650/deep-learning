from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=3, batch_size=32,
          validation_data=(X_test, y_test_cat), verbose=2)

# Select 4 sample test indices
sample_indices = [0, 1, 2, 3]
X_sample = X_test[sample_indices]
y_sample_true = y_test[sample_indices]

# Predict
predictions = model.predict(X_sample)
y_sample_pred = np.argmax(predictions, axis=1)

# Create results table
data = {
    "Input Digit Image": [f"Image of {t}" for t in y_sample_true],
    "Expected Label": y_sample_true,
    "Model Output": y_sample_pred,
    "Correct (Y/N)": ["Y" if t == p else "N" for t, p in zip(y_sample_true, y_sample_pred)]
}

df = pd.DataFrame(data)

# Display results
print("\nSample Predictions:\n")
print(df)

# Sample accuracy
sample_acc = (df["Correct (Y/N)"] == "Y").mean() * 100
print(f"\nSample Accuracy: {sample_acc:.2f}%")
output:
<img width="825" height="290" alt="image" src="https://github.com/user-attachments/assets/2e2230b2-fd26-4175-b933-ec7d6907e2db" />
