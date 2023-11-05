import os
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential , Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten , Dropout
from keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import math
from keras.layers import Dropout, BatchNormalization
from keras import regularizers

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
columns_to_keep = ['id', 'gender', 'articleType', 'baseColour', 'season', 'year',
                   'usage', 'productDisplayName']
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
categorical_columns = ['gender', 'articleType', 'baseColour', 'season', 'usage']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("Preprocessed data shape:", df.shape)

# Save the preprocessed DataFrame as a Feather file
df.to_feather('preprocessed_data.feather')

# Section 3: Preprocess the images and connect them to the DataFrame

# Folder path containing the photos
photos_folder = 'images'

# Load photos from the folder
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

# Create a shared backbone network (for feature extraction)
shared_backbone = Sequential()
shared_backbone.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same' , kernel_regularizer=l1(0.2))) ,
shared_backbone.add(keras.layers.MaxPooling2D(2,2)) ,
#shared_backbone.add(keras.layers.Conv2D(32,(3,3),activation = "relu") ) ,
shared_backbone.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same' , kernel_regularizer=l1(0.2))) ,
#shared_backbone.add(keras.layers.MaxPooling2D(2,2)) ,
#shared_backbone.add(keras.layers.Conv2D(64,(3,3),activation = "relu")) ,
shared_backbone.add(keras.layers.BatchNormalization())
#shared_backbone.add(keras.layers.MaxPooling2D(2,2)) ,
#shared_backbone.add(keras.layers.BatchNormalization())
#shared_backbone.add(keras.layers.Conv2D(128,(3,3),activation = "relu")) ,
#shared_backbone.add(keras.layers.MaxPooling2D(2,2)) ,
shared_backbone.add(keras.layers.Flatten()) ,
shared_backbone.add(keras.layers.Dense(550,activation="relu")) ,
#shared_backbone.add(keras.layers.Dropout(0.1,seed = 2019)) ,
#shared_backbone.add(keras.layers.Dense(400,activation ="relu")) ,
#shared_backbone.add(keras.layers.Dropout(0.3,seed = 2019)) ,
#shared_backbone.add(keras.layers.Dense(300,activation="relu")) ,
shared_backbone.add(keras.layers.Dropout(0.4,seed = 2019)),
#shared_backbone.add(keras.layers.Dense(200,activation ="relu" ,kernel_regularizer=l1(0.2))),
#shared_backbone.add(keras.layers.Dropout(0.2,seed = 2019)),
#shared_backbone.add(keras.layers.BatchNormalization())
shared_backbone.add(Dense(77, activation='softmax', )) ,

# Create a separate head for gender prediction
gender_head = Sequential()
gender_head.add(Dense(128, activation='relu'))
gender_head.add(Dense(1, activation='sigmoid'))  # Binary classification for gender

# Connect the shared backbone and gender head
shared_backbone_output = shared_backbone(shared_backbone.input)
gender_output = gender_head(shared_backbone_output)

# Create a model for gender prediction
gender_model = Model(inputs=shared_backbone.input, outputs=gender_output)

# Compile the gender model
gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the gender model architecture
gender_model.summary()

# Train the gender model
history = gender_model.fit(train_images_reshaped, train_df['gender'], epochs=10, batch_size=16, )

# Section 5: Model evaluation

# Evaluate the gender model
test_loss, test_accuracy = gender_model.evaluate(test_images_reshaped, test_df['gender'])
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions on the test data
gender_predictions = gender_model.predict(test_images_reshaped)

# Convert predictions to binary labels
gender_predicted_labels = (gender_predictions > 0.2).astype(int)
true_gender_labels = test_df['gender']

# Compute the confusion matrix
gender_confusion = confusion_matrix(true_gender_labels, gender_predicted_labels)

# Plot the confusion matrix for gender
plt.figure(figsize=(8, 6))
sns.heatmap(gender_confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Gender Prediction - Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Visualization as a diagram
pd.DataFrame(history.history).plot()
plt.show()

