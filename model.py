#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Dense, MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[4]:


data_dir = r"C:\Users\LOKESH\Downloads\archive\chest_xray\train"




sub_folders = os.listdir(data_dir)
print(len(sub_folders))


# In[6]:


images = []
labels = []


# In[7]:


for sub_folder in sub_folders:
    label = sub_folder
    
    # Constructing the path to the current sub_folder
    path = os.path.join(data_dir, sub_folder)
    
    # Listing all the images in the sub_folder
    sub_folder_images = os.listdir(path)
    
    for image_name in sub_folder_images:
        
        # Constructing the path of current image
        image_path = os.path.join(path, image_name)
        
        # Loading the image using OpenCV
        img = cv2.imread(image_path)
        
        img = cv2.resize(img, (256, 256))
        
        # Append the images to the image list 
        images.append(img)
        
        # Append the labels
        labels.append(label)


# In[8]:


# list of images and labels to the numpy array
images = np.array(images)
labels = np.array(labels)


# In[9]:


# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)


# In[10]:


# Image Preprocessing
def preprocessing(img):
    img = img / 255.0
    return img


# In[11]:


# Applying the preprocessing to the entire dataset
X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))


# In[12]:


# Data Augmentation
data_gen = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 10,
    shear_range = 0.1
    )


# In[13]:


# Label Encoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)


# In[14]:


# Encode the labels
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)


# In[15]:


# Number of Classes
num_classes = len(label_encoder.classes_)


# In[16]:


# Converting the labels into One-Hot encoding
y_train = to_categorical(y_train, num_classes = num_classes)
y_val = to_categorical(y_val, num_classes = num_classes)
y_test = to_categorical(y_test, num_classes = num_classes)


# In[17]:


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found, using CPU instead")


# In[18]:


from tensorflow.keras.layers import Input

def build_model():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(32, (5,5), strides=(1,1), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



# In[19]:


model = build_model()

print(model.summary())


# In[20]:


history = model.fit(data_gen.flow(X_train, y_train, batch_size = 32),
                   validation_data = (X_val, y_val),
                   epochs = 30,
                   verbose = 2)


# In[ ]:





# In[27]:


import matplotlib.pyplot as plt

# In[22]:


model.save('Y00.h5')
#tf.keras.models.load_model('model.h5')


# In[23]:


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Converting one-hot encoded test labels back to categorical labels
y_true = np.argmax(y_test, axis=1)

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[24]:


accuracy = accuracy_score(y_true, y_pred_classes)

# Precision
precision = precision_score(y_true, y_pred_classes)

# Recall
recall = recall_score(y_true, y_pred_classes)

# F1 Score
f1 = f1_score(y_true, y_pred_classes)

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_classes)
roc_auc = auc(fpr, tpr)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC: {roc_auc:.2f}")

