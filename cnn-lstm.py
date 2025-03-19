import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# Load the data
data = pd.read_excel('Final Road Traffic Incidents.xlsx')

# Preprocessing
data['starttime'] = pd.to_datetime(data['starttime'])
data['lastupdated'] = pd.to_datetime(data['lastupdated'])
data['duration'] = (data['lastupdated'] - data['starttime']).dt.total_seconds() / 60  # Duration in minutes

# Encode categorical features
label_encoders = {}
categorical_features = ['maincategory', 'subCategoryA', 'subCategoryB', 'attendinggroups', 'displayname', 'direction', 'mainstreet', 'suburb']
for column in categorical_features:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Scale numerical features
scaler = MinMaxScaler()
numerical_features = ['longitude', 'latitude', 'duration', 'lanes']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define features and labels
X = data[['longitude', 'latitude', 'duration', 'maincategory', 'subCategoryA', 'subCategoryB', 'attendinggroups', 'direction', 'mainstreet', 'lanes', 'suburb']].values
y_severity = data['isMajor'].values
y_time = data['duration'].values

# Split the data
X_train, X_test, y_severity_train, y_severity_test, y_time_train, y_time_test = train_test_split(X, y_severity, y_time, test_size=0.2, random_state=42)

# Reshape data for CNN-LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the CNN-LSTM model for severity prediction
severity_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

severity_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the severity model
severity_history = severity_model.fit(X_train, y_severity_train, epochs=20, batch_size=64, validation_split=0.2)

# Predict severity
y_severity_pred = severity_model.predict(X_test)
y_severity_pred = (y_severity_pred > 0.5).astype(int)

# Evaluate the severity model
print("Severity Model Evaluation")
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_severity_test, y_severity_pred)
print(conf_matrix)

print("Classification Report:")
class_report = classification_report(y_severity_test, y_severity_pred)
print(class_report)

print("Accuracy Score:", accuracy_score(y_severity_test, y_severity_pred))
print("Recall Score:", recall_score(y_severity_test, y_severity_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Severity Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot accuracy and loss for severity model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(severity_history.history['accuracy'], label='Accuracy')
plt.plot(severity_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Severity Model Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(severity_history.history['loss'], label='Loss')
plt.plot(severity_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Severity Model Loss over Epochs')

plt.show()

# Define the CNN-LSTM model for duration prediction
duration_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

duration_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the duration model
duration_history = duration_model.fit(X_train, y_time_train, epochs=20, batch_size=64, validation_split=0.2)

# Predict duration
y_time_pred = duration_model.predict(X_test)

# Evaluate the duration model
mae = np.mean(np.abs(y_time_test - y_time_pred))
print("Mean Absolute Error for Duration Prediction:", mae)

# Plot accuracy and loss for duration model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(duration_history.history['mae'], label='MAE')
plt.plot(duration_history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Duration Model MAE over Epochs')

plt.subplot(1, 2, 2)
plt.plot(duration_history.history['loss'], label='Loss')
plt.plot(duration_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Duration Model Loss over Epochs')

plt.show()
