import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
import tensorflow as tf
import os
import joblib
tf.config.set_visible_devices([], 'GPU')

# Check and make dirs
dirs = ["images", "npy_data"]
[os.makedirs(d, exist_ok=True) for d in dirs]


# Load the features and labels data
# X = np.load('features.npy')
X = np.load('features_hist.npy')
y = np.load('labels.npy')
ss = np.load('ssdata.npy')

idorg = np.arange(len(X))

# # Split the data into training, validation and testing sets
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Split the data into training, validation and testing sets
X_train, X_temp, y_train, y_temp, ss_train, ss_temp, idorg_train, idorg_temp = train_test_split(X, y, ss, idorg, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, ss_val, ss_test, idorg_val, idorg_test = train_test_split(X_temp, y_temp, ss_temp, idorg_temp, test_size=0.5, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test) 
# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

# Build the MLP model
model = Sequential()
model.add(Dense(1024, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

# Train the model and track training history
history = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_val, y_val))

# Plot training and validation curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./images/loss_nm300.png', dpi=300)
#plt.show()
plt.close()
# Make predictions on the test set
predictions = model.predict(X_test)

# For regression tasks (e.g., predicting a continuous value)
# Evaluate using a relevant metric (e.g., Mean Squared Error)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error on Test Set: {mse}')

# plt.plot(y_test, predictions, 'o')
plt.scatter(ss_test[:,0], ss_test[:,1],color='r',alpha=0.1,label='$S_{filtered}$')
plt.scatter(y_test, predictions,alpha=0.1,label='MLP')
plt.xlim([y_test.min(), y_test.max()])
plt.ylim([y_test.min(), y_test.max()])
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'orange')
plt.xlabel('Ground Truth, $S_{eff}$',weight='bold')
plt.ylabel('Prediction',weight='bold')
plt.legend()
plt.tight_layout()
# plt.savefig('plot_abstract.png', dpi=300)
plt.savefig('./images/plot_nm300.png', dpi=300)
#plt.show()


mape = np.mean(np.abs((y_test.squeeze() - predictions.squeeze()) / np.abs(y_test.squeeze()))) * 100
smape = np.mean(np.abs(2 * (y_test - predictions.squeeze()) / (np.abs(y_test) + np.abs(predictions.squeeze()))) * 100)
r2 = r2_score(y_test,predictions.squeeze())
r2_org = r2_score(ss_test[:,0],ss_test[:,1])
model.save('./output/model_ml_1.keras')

# Open a file in write mode
with open('./output/results.txt', 'w') as file:
    # Write the variables and their values to the file
    file.write(f'mape = {mape}\n')
    file.write(f'smape = {smape}\n')
    file.write(f'r2 = {r2}\n')
    file.write(f'r2_org = {r2_org}\n')

# Save input and predictions
np.save('./npy_data/y_test.npy',y_test)
np.save('./npy_data/predictions.npy',predictions)
np.save('./npy_data/idorg_train.npy',idorg_train)
np.save('./npy_data/idorg_val.npy',idorg_val)
np.save('./npy_data/idorg_test.npy',idorg_test)
np.save('./npy_data/ss_train.npy',ss_train)
np.save('./npy_data/ss_val.npy',ss_val)
np.save('./npy_data/ss_test.npy',ss_test)
