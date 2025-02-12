"""
Using transfer learning on a pre-trained CNN (MobileNet) to build an Alpaca/Not Alpaca classifier.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense, RandomFlip, RandomRotation, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# CREATE DATASET AND SPLIT DATA INTO TRAIN, VALIDATION, AND TEST DATASETS
# Set the training set to subset='training' and the validation set to subset='validation.' Same seed to avoid overlap
SEED = 42
tf.random.set_seed(SEED)
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
directory = os.path.join(os.getcwd(), 'dataset')

train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             labels='inferred',  # Auto infer labels based on subdirectory names
                                             seed=SEED)

validation_dataset = image_dataset_from_directory(directory,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  labels='inferred',
                                                  seed=SEED)


# Further split validation set into validation and test datasets if needed
test_dataset = validation_dataset.take(1)
print("Training batches:", train_dataset.cardinality().numpy())
print("Validation batches:", validation_dataset.cardinality().numpy())
print("Test batches:", test_dataset.cardinality().numpy())

# Class labels
class_names = train_dataset.class_names
print("The classes: {}".format(class_names))
print(train_dataset.element_spec)

# Output (images, labels): (TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32, name=None),
# TensorSpec(shape=(None,), dtype=tf.int32, name=None))


def class_count(dataset):
    """
    Count the number of elements per class in the dataset.

    Arguments:
        dataset -- tf.data.Dataset, the dataset to count elements from

    Returns:
        Counter object with counts of elements per class
    """
    # Initialize a Counter to keep track of the number of elements per class
    counter = Counter()

    # Iterate over the dataset
    for img, label in dataset:
        counter.update(label.numpy())

    return counter


# Count the elements per class
train_count = class_count(train_dataset)
validation_count = class_count(validation_dataset)
test_count = class_count(test_dataset)

# Print the counts in a readable format
for count_name, dataset_count in [("Training", train_count), ("Validation", validation_count), ("Test", test_count)]:
    for class_idx, count in dataset_count.items():
        print(f"{count_name} dataset: {count} images for class '{class_names[class_idx]}'")

# Visualize some images
fig = plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):  # Display the first 9 images of the mini batch
        ax = plt.subplot(3, 3, i + 1)  # Create subplots
        plt.imshow(images[i].numpy().astype('uint8'))  # Convert image to Unsigned Integer (8-bit) format
        plt.title(class_names[labels[i]])
        plt.tight_layout()
        plt.axis('off')

    plt.show()  # Display the images


# BUILDING THE MODEL USING MobileNet v2
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                 include_top=True,
                                                 weights='imagenet')

mobilenet_v2.summary()


# Note: MobileNet v2 was pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes.
# But we only need 2 classes for our classifier. We will have to customize the model to build an alpaca classifier

def data_augmenter():
    """
    Creates a 2-layer sequential model for data augmentation.
    This model applies random horizontal flips and random rotations to input images,
    helping to improve the robustness and generalization of the neural network during training.

    Returns:
        tf.keras.Sequential: A Keras Sequential model for data augmentation.
    """
    augmenter = Sequential([
        RandomFlip('horizontal'),   # Randomly flip input images horizontally
        RandomRotation(0.2)])       # Randomly rotate input images by up to 20%

    return augmenter


# Testing the data augmenter
data_augmentation = data_augmenter()
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.tight_layout()
        plt.axis('off')
    plt.show()


def classifier(input_shape=(160, 160, 3), augmenter=data_augmenter(), dropout_rate=0.2):
    """
    Builds a binary classifier on top of the MobileNetV2 model.

    Arguments:
        input_shape -- tuple, the shape of the input images (width, height, channels)
        data_augmentation -- tf.keras.Sequential, the data augmentation model
        dropout_rate -- float, the dropout rate (default 0.2)

    Returns:
        model -- tf.keras.Model, the compiled binary classification model
    """

    # Initialize MobileNetV2 as the base model with pre-trained ImageNet weights
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')

    # Freeze the base model to prevent its weights from being updated during training
    base.trainable = False

    # Create the input layer (same as the ImageNet v2 input size)
    inputs = Input(shape=input_shape)

    # Apply data augmentation to the inputs
    X = augmenter(inputs)

    # Data preprocessing: essential for ensuring that the input images are in the same format that the MobileNetV2 model
    # was trained on. This is important for consistency with training and is crucial for achieving optimal performance
    # and ensuring that the  classifier leverages the pre-trained weights effectively.
    X = preprocess_input(X)

    # Set training=False to avoid updating batch normalization statistics during training
    X = base(X, training=False)

    # Add new binary classification layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(rate=dropout_rate)(X)
    outputs = Dense(units=1)(X)

    # Define the model
    classifier_model = Model(inputs, outputs)

    return classifier_model


model = classifier(IMG_SHAPE, data_augmentation)
base_learning_rate = 0.001
loss_function = BinaryCrossentropy(from_logits=True)
model.compile(optimizer=Adam(learning_rate=base_learning_rate), loss=loss_function, metrics=['accuracy'])

initial_epochs = 5
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)


# Plots
# Extract accuracy and validation accuracy, prepending 0 for the initial epoch
def plot_history(hist):
    """
    Plots the training and validation accuracy and loss history.

    Arguments:
        history -- keras.callbacks.History object that contains training history
    """

    # Extract training and validation accuracy from history
    accuracy = hist.history['accuracy']
    val_accuracy = hist.history['val_accuracy']

    # Extract training and validation loss from history
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    # Set the figure size for the plots
    plt.figure(figsize=(8, 8))

    # Plot Training and Validation Loss
    plt.subplot(2, 1, 1)
    plt.plot(loss, '-o', label='Training Loss')
    plt.plot(val_loss, '-o', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    # plt.grid(True)  # Uncomment to add grid for better readability

    # Plot Training and Validation Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(accuracy, '-o', label='Training Accuracy')
    plt.plot(val_accuracy, '-o', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    # plt.ylim([0, 1])  # Set y-axis limit to [0, 1] for accuracy
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    # plt.grid(True)  # Uncomment to add grid for better readability

    # Display the plots
    plt.show()


plot_history(history)

# FINE-TUNING THE MODEL
# Allow the network to detect features more related to our data
for i, layer in enumerate(model.layers):
    print(i, ":", layer.name)

base_model = model.layers[4]  # Select the 5th layer from classifier (i.e., MobileNet v2)
base_model.trainable = True
print(f"Number of layers in the base model: {len(base_model.layers)}")

# Fine-tune from this layer onwards
fine_tune_at = 120

# Freeze all layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# Compiling the model
model.compile(optimizer=Adam(learning_rate=base_learning_rate * 0.1), loss=loss_function, metrics=['accuracy'])

# Define callbacks to be used during training
model_path = os.path.join(os.getcwd(), 'models', 'best_model.h5')
check_point_path = os.path.join(os.getcwd(), 'checkpoints', 'model_epoch_{epoch:02d}.h5')  # Save with epoch number
check_point = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
check_point2 = ModelCheckpoint(filepath=check_point_path, save_weights_only=False, save_freq='epoch')

fine_tune_epochs = 15
total_epochs = initial_epochs + fine_tune_epochs
callbacks = [check_point, check_point2]
history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
                         callbacks=callbacks)

plot_history(history_fine)

# Load the best model and set it for inference
best_model_path = model_path
# best_model_path = os.path.join(os.getcwd(), 'models', 'best_alpaca_model.h5')
pre_trained_model = load_model(best_model_path)
# tf.keras.backend.set_learning_phase(False)
pre_trained_model.trainable = False

train_info = pre_trained_model.evaluate(train_dataset)
validation_info = pre_trained_model.evaluate(validation_dataset)
test_info = pre_trained_model.evaluate(test_dataset)

for phase, phase_info in [("Training", train_info), ("Validation", validation_info), ("Test", test_info)]:
    print(f"{phase} dataset: loss is {round(phase_info[0], 4)} and accuracy is {round(phase_info[1], 4)}")

