import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Function to load images and angles
def load_data(image_directory, csv_file, image_extension='.png', image_size=(64, 64)):
    df = pd.read_csv(csv_file)
    images, angles = [], []
    for _, row in df.iterrows():
        img_filename = row['filename'] + image_extension
        img_path = os.path.join(image_directory, img_filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('L').resize(image_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            angles.append(row['angle'])
    return np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)

# Load the data
image_dir = 'C:/Users/Asus/OneDrive/Desktop/NNMHA/generated_images'
csv_file = 'C:/Users/Asus/OneDrive/Desktop/NNMHA/rectangle_angles.csv'
images, angles = load_data(image_dir, csv_file)

# Convert data to PyTorch tensors
#images = torch.tensor(images.transpose(0, 3, 1, 2))  # NCHW format
images = torch.tensor(np.expand_dims(images, axis=1))  # Add channel dimension

angles = torch.tensor(angles)

# Define a CNN model for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        #correct_input_size = self._get_conv_output_shape((64, 64), self.conv1, self.pool, self.conv2, self.pool, self.conv3)
        #print(correct_input_size)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    '''def _get_conv_output_shape(self, input_shape, *layers):
        with torch.no_grad():
            input_tensor = torch.zeros(1, 3, *input_shape)  # Adjust to 3 channels
            output_tensor = input_tensor
            for layer in layers:
                output_tensor = layer(output_tensor)
            return int(np.prod(output_tensor.size()))'''


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)

feature_extractor = FeatureExtractor()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(feature_extractor.parameters(), lr=0.001)

# Split data into training, validation, and test sets
train_images, test_images, train_angles, test_angles = train_test_split(images, angles, test_size=0.2, random_state=42)
train_images, val_images, train_angles, val_angles = train_test_split(train_images, train_angles, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

print(f'Training Data Size:{len(train_angles)}')
print(f'Test Data Size:{len(test_angles)}')
print(f'Validation Data Size:{len(val_angles)}')


# Convert to tensor dataset and dataloader for training and validation
train_dataset = torch.utils.data.TensorDataset(train_images, train_angles)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_images, val_angles)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize lists to monitor loss
train_losses, val_losses = [], []

# Inside your training loop
for epoch in range(200):
    # Training phase
    feature_extractor.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = feature_extractor(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation phase
    feature_extractor.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = feature_extractor(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    val_losses.append(running_loss / len(val_loader))

    print(f'Epoch [{epoch+1}/200], Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')


# Switch to evaluation mode for feature extraction
feature_extractor.eval()
with torch.no_grad():
    train_features = feature_extractor(train_images).numpy().reshape(-1, 1)
    test_features = feature_extractor(test_images).numpy().reshape(-1, 1)

# SVR for regression
svr = SVR(C=1, epsilon=0.5)
svr.fit(train_features, train_angles.numpy())
predicted_angles = svr.predict(test_features)

# Print the actual vs predicted values
print("Actual vs Predicted Angles:")
for actual, predicted in zip(test_angles.numpy(), predicted_angles):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Calculate and print the mean squared error on the test set
mse = mean_squared_error(test_angles.numpy(), predicted_angles)
print(f"Mean Squared Error on Test Set: {mse}")

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to calculate MAE
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Calculate and print the RMSE on the test set
rmse_value = rmse(test_angles.numpy(), predicted_angles)
print(f"Root Mean Squared Error on Test Set: {rmse_value}")

# Calculate and print the MAE on the test set
mae_value = mae(test_angles.numpy(), predicted_angles)
print(f"Mean Absolute Error on Test Set: {mae_value}")

# Calculate and print the MAPE on the test set
mape_value = mape(test_angles.numpy(), predicted_angles)
print(f"Mean Absolute Percentage Error on Test Set: {mape_value:.2f}%")


# Plotting the training and validation losses
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


"""# Collect actual vs predicted values in a DataFrame
results_df = pd.DataFrame({
    'Actual Angles': test_angles.numpy(),
    'Predicted Angles': predicted_angles
})

# Collect error metrics in a DataFrame
metrics_df = pd.DataFrame({
    'Mean Squared Error': [mse],
    'Root Mean Squared Error': [rmse_value],
    'Mean Absolute Error': [mae_value],
    'Mean Absolute Percentage Error': [mape_value]
})

# Display the tables
print("Actual vs Predicted Angles:")
print(results_df)
print("\nError Metrics:")
print(metrics_df)"""
