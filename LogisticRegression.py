import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cale = 'C:\\Users\Gnegnel\Desktop\FMI\Master\Anul1\Machine Learning\Proiect\Chest_xray\dataset'

categorie = ['Normal', 'Pneumonia']

#data = []
x = []
y = []

for folder in categorie:
    path = os.path.join(cale, folder)
    label = categorie.index(folder)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        xray = cv2.imread(imgpath, 0)
        try:
            xray = cv2.resize(xray, (64, 64))
            image = np.array(xray).flatten()

            #data.append([image, label])
            x.append(image) 
            y.append(label) 
        except Exception as e:
            pass
        
x = np.array(x)


x = x/255.0

#print(x)
#print(y)


# Impartim datele in date de antrenare si date de testare (Am mers pe 80% training si 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Punem modelul la invatat...
model = LogisticRegression(max_iter= 1000)
model.fit(x_train, y_train) 


# Testam modelul
predictions = model.predict(x_test)


# Matricea de Confuzie
print('Matricea de Confuzie:')
print(confusion_matrix(y_test, predictions))


# Precision, recall, F1-score
print(classification_report(y_test, predictions))
print('Accuracy = ', accuracy_score(y_test, predictions))


#coeff
print(model.coef_, model.intercept_)

poza = x_test[10].reshape(64, 64)
plt.imshow(poza, cmap = 'gray')
plt.show
    

