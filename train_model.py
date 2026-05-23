# --- Imports ---
import numpy as np                 # numerical arrays (used for class weights)
import matplotlib.pyplot as plt    # plotting the accuracy/loss graphs
import os                          # reading folder names and counting files

# Keras tools for loading images, building, and training the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight  # fixes class imbalance

# Path to your dataset (folders containing one sub-folder per emotion)
train_dir = 'train'
test_dir = 'test'

# Emotion labels = the names of the sub-folders inside 'train'
# Sorting keeps the order consistent every time you run the script.
emotions = os.listdir(train_dir)
emotions.sort()
print("Emotions:", emotions)

# Count how many images are in each emotion folder (just for a sanity check)
print("\n--- Training Data ---")
for emotion in emotions:
    count = len(os.listdir(os.path.join(train_dir, emotion)))
    print(f"{emotion}: {count} images")

print("\n--- Test Data ---")
for emotion in emotions:
    count = len(os.listdir(os.path.join(test_dir, emotion)))
    print(f"{emotion}: {count} images")

# --- Config ---
# Constants used throughout the script, kept in one place so they're easy to change.
IMG_SIZE = 48        # images are resized to 48x48 pixels
BATCH_SIZE = 64      # how many images the model sees before updating its weights
NUM_CLASSES = 7      # 7 emotions = 7 output neurons
EPOCHS = 50          # max number of full passes over the training data

# --- Data augmentation / preprocessing ---
# The training generator creates slightly altered copies of each image on the fly
# (rotated, shifted, zoomed, flipped). This shows the model more variety so it
# generalizes better instead of memorizing the exact training pictures.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # convert pixel values from 0-255 down to 0-1
    rotation_range=15,        # randomly rotate images up to 15 degrees
    width_shift_range=0.1,    # randomly shift left/right by up to 10%
    height_shift_range=0.1,   # randomly shift up/down by up to 10%
    zoom_range=0.1,           # randomly zoom in/out by up to 10%
    horizontal_flip=True,     # randomly mirror images left-to-right
)

# Test images are only rescaled, NOT augmented, because we want to evaluate
# the model on clean, unmodified data.
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# flow_from_directory reads images straight from the folders. It uses each
# sub-folder name as the label automatically.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),  # resize every image to 48x48
    color_mode='grayscale',            # 1 color channel (black & white)
    batch_size=BATCH_SIZE,
    class_mode='categorical',          # labels as one-hot vectors (for softmax)
    shuffle=True,                      # shuffle so batches aren't all one emotion
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,                     # keep order fixed for consistent evaluation
)

# --- Handle class imbalance ---
# Some emotions have far more images than others (e.g. happy ~7000 vs disgust ~400).
# Without help, the model would just learn to predict the common classes.
# compute_class_weight gives rarer classes a higher weight so the model pays
# more attention to them during training.
class_labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels,
)
# fit() expects a {class_index: weight} dictionary, so we build that here.
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:", class_weight_dict)

# --- Build CNN ---
# A Sequential model is a simple stack of layers, one after another.
# The 3 "conv blocks" each detect more complex patterns than the last:
# edges -> shapes -> facial features.
model = Sequential([
    # Conv block 1: 64 filters scan the image for basic patterns like edges
    Conv2D(64, (3, 3), padding='same', activation='relu',
           input_shape=(IMG_SIZE, IMG_SIZE, 1)),  # 1 = grayscale channel
    BatchNormalization(),     # stabilizes and speeds up training
    MaxPooling2D((2, 2)),     # halves the image size, keeping the strongest signals
    Dropout(0.25),            # randomly switches off 25% of neurons to reduce overfitting

    # Conv block 2: 128 filters learn more detailed patterns
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Conv block 3: 256 filters learn high-level features
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Classifier: turn the learned features into a prediction
    Flatten(),                          # flatten the 2D feature maps into a 1D list
    Dense(256, activation='relu'),      # fully-connected layer that combines features
    BatchNormalization(),
    Dropout(0.5),                       # heavier dropout here since this layer is large
    Dense(NUM_CLASSES, activation='softmax'),  # 7 outputs = probability per emotion
])

# Compile = set up how the model learns.
model.compile(
    optimizer=Adam(learning_rate=0.0001),   # Adam adjusts weights; small LR = careful steps
    loss='categorical_crossentropy',        # standard loss for multi-class classification
    metrics=['accuracy'],                    # track accuracy while training
)

model.summary()  # print a table of all layers and their parameter counts

# --- Callbacks ---
# Callbacks run automatically during training to make it smarter.

# Stop early if validation loss hasn't improved for 10 epochs, and roll back
# to the best weights seen. This saves time and prevents overfitting.
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
)

# If validation loss plateaus for 5 epochs, cut the learning rate in half so
# the model can fine-tune with smaller steps.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)

# --- Train ---
# This is where the actual learning happens.
history = model.fit(
    train_generator,
    validation_data=test_generator,     # check performance on unseen data each epoch
    epochs=EPOCHS,
    class_weight=class_weight_dict,     # apply the imbalance fix from earlier
    callbacks=[early_stop, reduce_lr],
)

# --- Save model ---
# Save the trained model to a file so you can reuse it later without retraining.
model.save('emotion_model.h5')
print("\nModel saved as emotion_model.h5")

# --- Plot accuracy and loss ---
# Draw two graphs side by side so you can see how training went.
# Comparing train vs validation lines tells you if the model is overfitting.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left graph: accuracy over time (higher is better)
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Right graph: loss over time (lower is better)
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()                      # keep the graphs from overlapping
plt.savefig('training_results.png')     # save the figure to a file
print("Training plots saved as training_results.png")
