#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.svm import LinearSVC


# Citim datele din csv
X_ds = pd.read_csv('brain_x.csv', low_memory=False)
y_ds = pd.read_csv('brain_y.csv', low_memory=False)

# target- ul (depresiv sau normal)
#y_ds['is_depressed'] = np.where(y_ds['is_depressed'] == 1, 'depresiv', 'normal') #schimbam din R in binar. MCC inbalanced classes


# Pregatim setul de antrenare
X = X_ds.iloc[:, :] # valorile atributelor

y = y_ds.iloc[:, -1] # target value

#print(X)
#print(y)

# Impartim datele in date de antrenare si date de testare (Am mers pe 80% training si 20% testing)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

kfd = model_selection.KFold(n_splits=5, shuffle=True, random_state=47)

classifier = LinearSVC(
    dual=True,
    max_iter=2000,
    random_state=47
)

cv_results = model_selection.cross_val_score(
    classifier,
    x_train,
    y_train,
    cv = kfd,
    scoring = 'f1'
)

print("cross validation result", cv_results.mean())
print()


print("SVC, poly kernel")
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(x_train, y_train)

predict_depression = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, predict_depression))
print()


print("SVC, rbf kernel")
model = SVC(C=1, kernel='rbf', gamma='auto')
model.fit(x_train, y_train)

predict_depression = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, predict_depression))