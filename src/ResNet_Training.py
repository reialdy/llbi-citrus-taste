# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:27:35 2024
@author: Reinaldy
Training
ResNet50
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import itertools
import os
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import BinaryCrossentropy     
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

data_train_directory = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Red\Splitted (Processed Mask) - Undersampling\Train"
data_validation_directory = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Red\Splitted (Processed Mask) - Undersampling\Valid"

#Parameters
img_height = 256
img_width = 256
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_train_directory,
                                                               labels='inferred',
                                                               label_mode='binary',
                                                               color_mode='rgb',
                                                               seed=1337,
                                                               image_size=(img_height,img_width),
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               verbose=True)
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(data_validation_directory,
                                                             labels='inferred',
                                                             label_mode='binary',
                                                             color_mode='rgb',
                                                             seed=1337,
                                                             image_size=(img_height,img_width),
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             verbose=True)
train_ds = train_ds.prefetch(buffer_size=32)
validation_ds = validation_ds.prefetch(buffer_size=32)

#visualizing the data
plt.figure(figsize=(10, 10))
label_map = ['Asam', 'Manis']  # map 0 to 'Asam', 1 to 'Manis'
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(label_map[int(labels[i])])  # use label_map to get the string label
        plt.axis("off")
        
# Model
epochs=50
base_model = ResNet50(include_top=False,
                      input_shape=(img_height,img_width, 3),
                      pooling='max',
                      weights='imagenet')

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

## If you want to load the model, uncomment this
# model = tf.keras.models.load_model(
#     r"D:/College/Skripsi/Code/CNN/ResNet/save_models/red/raw_red_augmented_model.h5")

# model.load_weights(r"D:/College/Skripsi/Code/CNN/ResNet/save_models/red/weights-improvement.epochs20-val_accuracy0.96.weights.h5")

#train and evaluate model ResNet_50
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
filepath = 'D:\\College\\Skripsi\\Code\\CNN\\ResNet\\save_models\\red_mask\\weights-improvement.epochs{epoch:02d}-val_accuracy{val_accuracy:.2f}.weights.h5'
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
class stopTraining(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') >= 0.99):
            print('Stop training because validation accuracy has reached 99%!')
            self.model.stop_training = True
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=10) # Seharusnya monitor val_loss ya!
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max',
                             save_weights_only=True)
                
#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('/ResNet/save_models/red_mask/red_mask_resnet_model_logs.csv', separator=',', append=False)

callback_list = [reduce_lr, log_csv, checkpoint, stopTraining()]

history = model.fit(train_ds,
                    validation_data=validation_ds,
                    verbose=1, batch_size=64,
                    epochs=epochs, callbacks=callback_list)

model.save('/ResNet/save_models/red_mask/red_mask_model.h5')

#Evaluate the ResNet-50 model after training the model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# Combination of plot graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate training and validation dataset
model.evaluate(validation_ds)
print('\nTraining and Validation Evaluation : \n')
print('Train acc: {0:0.2f}'.format(np.round((history.history['accuracy'][-1])*100, 2)))
print('validation acc: {0:0.2f}'.format(np.round((history.history['val_accuracy'][-1])*100, 2)))
print('Train loss: {0:0.2f}'.format(np.round((history.history['loss'][-1]), 2)))
print('validation loss: {0:0.2f}'.format(np.round((history.history['val_loss'][-1]), 2)))


####################################################################################################################33

train_ds_metrics = tf.keras.preprocessing.image_dataset_from_directory(data_train_directory, labels ='inferred',
                                                             color_mode='rgb',
                                                             image_size=(img_height,img_width),
                                                             batch_size=None,
                                                             shuffle=True)

valid_ds_metrics = tf.keras.preprocessing.image_dataset_from_directory(data_validation_directory, labels ='inferred',
                                                             color_mode='rgb',
                                                             image_size=(img_height,img_width),
                                                             batch_size=None,
                                                             shuffle=True)

# Load Model
from tensorflow.keras.models import load_model
model = load_model(r'/ResNet/save_models/red_mask/red_mask_model.h5')

# Get x and y train
inp_train = []
labels_train = []
for x, y in train_ds_metrics.as_numpy_iterator():
    inp_train.append(x)
    labels_train.append(y)

inp_train = np.array(inp_train)
print(inp_train.shape)
labels_train = np.array(labels_train)
print(labels_train)

predicted_train = model.predict(inp_train)
print(predicted_train[:,0])

# Get x and y valid
inp_valid = []
labels_valid = []
for x, y in valid_ds_metrics.as_numpy_iterator():
    inp_valid.append(x)
    labels_valid.append(y)

inp_valid = np.array(inp_valid)
print(inp_train.shape)
labels_valid = np.array(labels_valid)
print(labels_valid)

predicted_valid = model.predict(inp_valid)
print(predicted_valid[:,0])


# Plot the confussion matrix
def plot_confusion_matrix(cm, classes,
                        normalize=None,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm_plot_labels = ['Asam', 'Manis']

# Confussion Matrix for train and valid dataset
threshold = 0.5

cm_train= confusion_matrix(labels_train, predicted_train > threshold)
print(cm_train)
plot_confusion_matrix(cm=cm_train, classes=cm_plot_labels, title='Confussion Matrix Train Datasets Mask \nWith Laser 648 nm')

cm_valid= confusion_matrix(labels_valid, predicted_valid > threshold)
print(cm_valid)
plot_confusion_matrix(cm=cm_valid, classes=cm_plot_labels, title='Confussion Matrix Valid Datasets Mask \nWith Laser 648 nm')

# ROC - AUC
# Calculate the ROC curve of train data
fpr, tpr, _ = roc_curve(labels_train, predicted_train)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Train\n Laser 648 nm')
plt.legend(loc="lower right")
plt.show()

# Calculate the ROC curve of valid data
fpr, tpr, _ = roc_curve(labels_valid, predicted_valid)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Valid Raw Dataset\n Laser 648 nm')
plt.legend(loc="lower right")
plt.show()

############## Testing the Mode on Test Set #####################

data_test_directory = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Red\Splitted (Processed Mask) - Undersampling\Test"
test_ds_evaluation = tf.keras.preprocessing.image_dataset_from_directory(data_test_directory,
                                                             labels='inferred',
                                                             label_mode='binary',
                                                             color_mode='rgb',
                                                             seed=1337,
                                                             image_size=(img_height,img_width),
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             verbose=True)

test_ds_metrics = tf.keras.preprocessing.image_dataset_from_directory(data_test_directory, labels ='inferred',
                                                             color_mode='rgb',
                                                             image_size=(img_height,img_width),
                                                             batch_size=None,
                                                             shuffle=True)


# Load Model
from tensorflow.keras.models import load_model
model_test = load_model(r'/ResNet/save_models/red_mask/red_mask_model.h5')

# Evaluation
test_loss, test_accuracy = model_test.evaluate(test_ds_evaluation, batch_size=64)
print("result : \n")
print(f"Test Loss:     {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

inp = []
labels = []
for x, y in test_ds_metrics.as_numpy_iterator():
    inp.append(x)
    labels.append(y)

inp = np.array(inp)
print(inp.shape)
labels = np.array(labels)
print(labels)

predicted = model_test.predict(inp)
print(predicted[:,0])

# Save to dataframe and excel
import pandas as pd

labels_str = ['Asam' if label == 0 else 'Manis' for label in labels]
predicted_str = ['Manis' if pred >= 0.5 else 'Asam' for pred in predicted[:,0]]
keterangan = ['Sesuai' if orig == hasil else 'Tidak Sesuai' for orig, hasil in zip(labels_str, predicted_str)]
tested_class = {'Original Data :' : labels, 
                'Original data (String) : ' : labels_str, 
                'Hasil Testing :' : predicted[:,0], 
                'Hasil Testing (String) :' : predicted_str, 
                'Keterangan :' : keterangan}
tested_class_df = pd.DataFrame(tested_class)
tested_class_df.to_excel("/ResNet/hasil_test_resnet_red_mask.xlsx", sheet_name='red', index=False)


# Confussion Matrix for testing dataset
threshold = 0.5
cm_test= confusion_matrix(labels, predicted > threshold)
print(cm_test)
plot_confusion_matrix(cm=cm_test, classes=cm_plot_labels, title='Confussion Matrix Testing Raw Dataset\nLaser 648 nm')

#################################################################################################################

# ROC - AUC
# Calculate the ROC curve
fpr, tpr, _ = roc_curve(labels, predicted)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Testing Mask Dataset\nLaser 648 nm')
plt.legend(loc="lower right")
plt.show()

# AUC Value
auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)
