import os
import cv2
import keras
import pandas as pd
from keras import regularizers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Dropout, BatchNormalization


# Section 1: Importing the Excel file and filtering the DataFrame

xlsx = pd.ExcelFile('Thesisstyles.xlsx')
df = pd.read_excel(xlsx, sheet_name='Thesisstyles')

# Keep only the rows where masterCategory is "Apparel"
df = df[df['subCategory'] == 'Topwear']

# Get the list of valid photo IDs
valid_photo_ids = [int(os.path.splitext(filename)[0]) for filename in os.listdir('images')]

# Filter the DataFrame based on valid photo IDs
df = df[df['id'].isin(valid_photo_ids)]

# Reset the DataFrame index
df.reset_index(drop=True, inplace=True)

# Print the updated DataFrame shape
print("Updated DataFrame shape:", df.shape)

# Section 2: Data preprocessing

# Keep only the necessary columns
columns_to_keep = ['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', ]
df = df[columns_to_keep]

# Preprocess the columns
df['gender'] = df['gender'].str.lower()
df['articleType'] = df['articleType'].str.lower()
df['baseColour'] = df['baseColour'].str.lower()
df['season'] = df['season'].str.lower()
df['usage'] = df['usage'].str.lower()

# Remove spaces and special characters
df['gender'] = df['gender'].str.replace('\s+', '')
df['articleType'] = df['articleType'].str.replace('[^\w\s]', '')
df['baseColour'] = df['baseColour'].str.replace('[^\w\s]', '')
df['season'] = df['season'].str.replace('[^\w\s]', '')
df['usage'] = df['usage'].str.replace('[^\w\s]', '')

# Perform label encoding for categorical columns
label_encoder = LabelEncoder()
num_classes_dict = {}  # To store the number of classes for each category
categorical_columns = ['gender', 'articleType', 'baseColour', 'season', 'usage']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
    num_classes_dict[column] = len(label_encoder.classes_)

# Calculate the total number of unique classes across all categorical columns
total_classes = sum(num_classes_dict.values())

print("Preprocessed data shape:", df.shape)

# Save the preprocessed DataFrame as a Feather file
df.to_feather('preprocessed_data.feather')

# Section 3: Preprocess the images and connect them to the DataFrame

# Folder path containing the photos
photos_folder = 'images'

# Load photos from the folder using data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

photos = []
valid_rows = []

for filename in os.listdir(photos_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        photo_path = os.path.join(photos_folder, filename)
        photo_id = int(os.path.splitext(filename)[0])
        if photo_id in df['id'].values:
            photo = cv2.imread(photo_path)
            if photo is not None:
                photos.append(photo)
                valid_rows.append(photo_id)

# Filter the DataFrame based on valid rows
df = df[df['id'].isin(valid_rows)]

# Reset the DataFrame index
df.reset_index(drop=True, inplace=True)

# Print the updated DataFrame shape
print("Updated DataFrame shape:", df.shape)

# Preprocess the photos
processed_photos = []
valid_rows = []

for idx, row in df.iterrows():
    photo_id = row['id']
    photo_filename = str(photo_id) + '.jpg'
    photo_path = os.path.join(photos_folder, photo_filename)
    photo = cv2.imread(photo_path)
    if photo is not None:
        # Resize the image to 64x64
        resized_photo = cv2.resize(photo, (64, 64))
        # Normalize the pixel values to be between 0 and 1
        normalized_photo = resized_photo.astype(np.float32) / 255.0
        # Reshape the photo to 3D
        reshaped_photo = normalized_photo.reshape(64, 64, 3)
        processed_photos.append(reshaped_photo)
        valid_rows.append(photo_id)

        # Apply data augmentation
        augmented_photos = []
        for _ in range(5):  # You can adjust the number of augmented samples
            augmented_photo = datagen.random_transform(reshaped_photo)
            augmented_photos.append(augmented_photo)

        photos.extend(augmented_photos)
        valid_rows.extend([photo_id] * len(augmented_photos))

# Convert the processed photos to a numpy array
processed_photos = np.array(processed_photos)

# Reshape the processed photos to 2D
num_samples = processed_photos.shape[0]
processed_photos_2d = processed_photos.reshape(num_samples, -1)

# Create a list of column names for the DataFrame
column_names = [f"pixel_{i}" for i in range(processed_photos_2d.shape[1])]

# Create a DataFrame with processed photos and column names
processed_df = pd.DataFrame(data=processed_photos_2d, columns=column_names)

# Write the processed photos to Feather file
processed_df.to_feather('processed_photos.feather')

# Filter the DataFrame based on valid rows
df = df[df['id'].isin(valid_rows)]
df.reset_index(drop=True, inplace=True)

# Print the shape of the processed photos and the updated DataFrame
print("Processed photos shape:", processed_photos.shape)
print("Updated DataFrame shape:", df.shape)

# Split the data into training and testing sets
train_df, test_df, train_images, test_images = train_test_split(df, processed_photos_2d, test_size=0.1, random_state=42)

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

# Convert categorical columns to one-hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
train_categorical_encoded = encoder.fit_transform(train_df[categorical_columns])
test_categorical_encoded = encoder.transform(test_df[categorical_columns])

# Reshape the train_images and test_images arrays
train_images_reshaped = train_images.reshape(train_images.shape[0], 64, 64, 3)
test_images_reshaped = test_images.reshape(test_images.shape[0], 64, 64, 3)

# Prepare the input data for the model
input_shapes = (64, 64, 3 + len(categorical_columns))
train_inputs = np.concatenate(
    (train_images_reshaped, np.tile(train_categorical_encoded[:, np.newaxis, np.newaxis, :len(categorical_columns)], (1, 64, 64, 1))),
    axis=-1)
test_inputs = np.concatenate(
    (test_images_reshaped, np.tile(test_categorical_encoded[:, np.newaxis, np.newaxis, :len(categorical_columns)], (1, 64, 64, 1))),
    axis=-1)

print("Train inputs shape:", train_inputs.shape)
print("Test inputs shape:", test_inputs.shape)

# Section 4: Build and train the model

# Define model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shapes, padding='same'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu' ))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(total_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', ])

# Train the model with data augmentation
datagen.fit(train_images_reshaped)

history = model.fit(
    datagen.flow(train_inputs, train_categorical_encoded, batch_size=64),
    steps_per_epoch=len(train_inputs) / 64,
    epochs=10,
    validation_data=(test_inputs, test_categorical_encoded)
)

# Section 5: Model evaluation

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_inputs, test_categorical_encoded, verbose=2)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions on the test data
predictions = model.predict(test_inputs)

# Save the trained model
model.save('ImageCNN.h5')

# Convert predictions from one-hot encoded format to labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_categorical_encoded, axis=1)

# Get the class labels
class_labels = label_encoder.classes_

# Compute the confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# Compute classification report
report = classification_report(true_labels, predicted_labels, labels=np.unique(true_labels))

# Print the classification report
print(report)


# Plot loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



# Print the class labels along with the number of samples
for i, (label, count) in enumerate(zip(class_labels, np.sum(test_categorical_encoded, axis=0))):
    print(f"Label {i} ({label}): {count} samples")

#In general if we had a better PC we would add more layers on the CNN (as shown in the code) and more epochs at the training section
#This would lead to a much more accurate CNN in the fraction of time . Sadly my pc is not capable of doing
#such demanding projects so i had to adapt to my pc's capabilities .
