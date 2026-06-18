# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:27:35 2024
@author: Reinaldy
Testing
ResNet50
"""

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

img_height = 150
img_width = 150
batch_size = 32

data_test_directory = r"D:\College\Skripsi\Image Acquisition\Jeruk Siam Garut\LLBI\Labeled - 2 Class\Red\Splitted - Undersampling\Augmented\Test"
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
model_test = load_model(r'D:/College/Skripsi/Code/CNN/ResNet/save_models/red/red_augmented_model.h5')

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

## If you want to save the result to dataframe and excel, please uncomment this below code.
# import pandas as pd

# labels_str = ['Asam' if label == 0 else 'Manis' for label in labels]
# predicted_str = ['Manis' if pred >= 0.5 else 'Asam' for pred in predicted[:,0]]
# keterangan = ['Sesuai' if orig == hasil else 'Tidak Sesuai' for orig, hasil in zip(labels_str, predicted_str)]
# tested_class = {'Original Data :' : labels, 
#                 'Original data (String) : ' : labels_str, 
#                 'Hasil Testing :' : predicted[:,0], 
#                 'Hasil Testing (String) :' : predicted_str, 
#                 'Keterangan :' : keterangan}
# tested_class_df = pd.DataFrame(tested_class)
# tested_class_df.to_excel("hasil_test_resnet_blue.xlsx", sheet_name='blue', index=False)


# Confussion Matrix for testing dataset
threshold = 0.5
cm_test= confusion_matrix(labels, predicted > threshold)
print(cm_test)

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
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Hasil Prediksi')
    
cm_plot_labels = ['Asam', 'Manis']
plot_confusion_matrix(cm=cm_test, classes=cm_plot_labels, title='Confussion Matrix Laser 648 nm\nSubset Testing')

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
plt.title('Kurva ROC\nLaser 648 nm Subset Training')
plt.legend(loc="lower right")
plt.show()

# AUC Value
auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)
