import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pr_path = 'C:\\Users\Gnegnel\Desktop\FMI\Master\Anul1\Machine Learning\Proiect\Chest_xray\dataset'

category = ['Normal', 'Pneumonia']

data = []

for folder in category:
    path = os.path.join(pr_path, folder)
    label = category.index(folder)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        xray = cv2.imread(imgpath, 0)
        try:
            xray = cv2.resize(xray, (64, 64))
            image = np.array(xray).flatten()

            data.append([image, label])
        except Exception as e:
            pass

random.shuffle(data)
x = []
y = []

for feature, label in data:
    x.append(feature)
    y.append(label)
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

model = RandomForestClassifier(n_estimators=300)
model.fit(x_train, y_train)



prediction = model.predict(x_test)
accuracy = model.score(x_test, y_test)

print('Accuracy: ', accuracy)
print('Prediction is: ', category[prediction[0]])
# Matricea de Confuzie
print('Matricea de Confuzie:')
print(confusion_matrix(y_test, prediction))
# Precizie, recall, F1-score
print(classification_report(y_test, prediction))

poza = x_test[0].reshape(64,64)
plt.imshow(poza, cmap = 'gray')
plt.show
