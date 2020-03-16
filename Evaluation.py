import numpy as np
from PIL import Image

y_true = np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\ChangeDetection(Label)\Prediction_Label\Prediction_Label.tif'))
y_pred = np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Predict_2004To2014\ModelWithBatch_epochs_05\Predict2004_ModelWithBatch_epochs_05_Th_mean.tif'))

Confusion_Matrix = np.zeros(shape=y_true.shape)
for i in range(0, y_true.shape[0]):
    for j in range(0, y_true.shape[1]):
        if y_true[i, j] == 1 and y_pred[i, j] == 1:  # Urban-->Urban
            Confusion_Matrix[i, j] = 0  # Hits
        elif y_true[i, j] == 0 and y_pred[i, j] == 1:  # NonUrban-->Urban
            Confusion_Matrix[i, j] = 1  # False Alarms
        elif y_true[i, j] == 1 and y_pred[i, j] == 0:  # NotUrban-->Urban
            Confusion_Matrix[i, j] = 2  # Misses
        elif y_true[i, j] == 0 and y_pred[i, j] == 0:
            Confusion_Matrix[i, j] = 3  # Correct Rejection

unique, counts = np.unique(Confusion_Matrix, return_counts=True)
freq = np.asarray((unique, counts)).T
Hits = freq[0, 1]
False_Alarms = freq[1, 1]
Misses = freq[2, 1]
Correct_Rejection = freq[3, 1]

print('Hits:', Hits, 'False_Alarms:', False_Alarms, 'Misses:', Misses, 'Correct_Rejection:', Correct_Rejection)

PA = Hits / (Hits + Misses)
OA = (Hits + Correct_Rejection) / (Hits + Misses + False_Alarms + Correct_Rejection)
FOM = Hits / (Hits + False_Alarms + Misses)
print('PA =', PA, 'OA =', OA, 'FOM:', FOM)
