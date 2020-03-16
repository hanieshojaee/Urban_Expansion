import numpy as np
from PIL import Image

y_true = np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\ChangeDetection(Label)\Prediction_Label.tif'))
y_pred = np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\input_Image\results_input_9depth\Predicted_2004.tif'))

Confusion_Matrix = np.zeros(shape=y_true.shape)
for i in range (0,y_true.shape[0]):
    for j in range(0, y_true.shape[1]):
        if y_true[i,j] == 0 and y_pred[i,j] == 0:    #NotUrban-->NotUrban
            Confusion_Matrix[i,j] = 0  #TP
        elif y_true[i,j] == 1 and y_pred[i,j] == 1:  #Urban-->Urban
            Confusion_Matrix[i,j] = 1  #TN
        elif y_true[i,j] == 0 and y_pred[i,j] == 1:  #NotUrban-->Urban
            Confusion_Matrix[i,j] = 2 #FN
        elif y_true[i,j] == 1 and y_pred[i,j] == 0:
            Confusion_Matrix[i,j] = 3  #FP
            
unique, counts = np.unique(Confusion_Matrix, return_counts=True)
freq = np.asarray((unique, counts)).T
TP = freq[0,1]
TN = freq[1,1]
FN = freq[2,1]
FP = freq[3,1]

print('TP:', TP, 'TN:', TN, 'FN:', FN, 'FP:', FP)
PA = TP/(TP + FN)
OA = (TP + TN)/(y_true.shape[0] * y_true.shape[1])
FOM = TN / ((y_true.shape[0] * y_true.shape[1]) - TP)
print('PA = ', PA , 'OA = ', OA)

