import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



# Citim datele din csv
X_ds = pd.read_csv('brain_x.csv', low_memory=False)
y_ds = pd.read_csv('brain_y.csv', low_memory=False)

# target- ul (depresiv sau normal)
y_ds['is_depressed'] = np.where(y_ds['is_depressed'] == 1, 'depresiv', 'normal') #schimbam din R in binar. MCC inbalanced classes


# Pregatim setul de antrenare
X = X_ds.iloc[:, :] # valorile atributelor

y = y_ds.iloc[:, -1] # target value

#print(X)
#print(y)


# Plotam relatia dintre fiecare variabila si y (depresiv/normal)
plt.xlabel('Criteriu')
plt.ylabel('Stare')

pltX = X_ds.loc[:, 'pericallosal']
pltY = y_ds.loc[:, 'is_depressed']
plt.scatter(pltX, pltY, color='blue', label='pericallosal')

# pltX = X_ds.loc[:, 'occipital_sup']
# pltY = y_ds.loc[:, 'is_depressed']
# plt.scatter(pltX, pltY, color='red', label='occipital')

# pltX = X_ds.loc[:, 'oc-temp_med-Parahip']
# pltY = y_ds.loc[:, 'is_depressed']
# plt.scatter(pltX, pltY, color='green', label='temporal')

# pltX = X_ds.loc[:, 'circular_insula_ant']
# pltY = y_ds.loc[:, 'is_depressed']
# plt.scatter(pltX, pltY, color='yellow', label='circular')

plt.legend()
plt.show()


# Impartim datele in date de antrenare si date de testare (Am mers pe 80% training si 20% testing)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Punem modelul la invatat...
model = LogisticRegression()
model.fit(x_train, y_train) 


# Testam modelul
predictions = model.predict(x_test)


# Matricea de Confuzie
print('Matricea de Confuzie:')
print(confusion_matrix(y_test, predictions))


# Precizie, recall, F1-score
print(classification_report(y_test, predictions))
print('accuracy = ', accuracy_score(y_test, predictions))


#coeff
print(model.coef_, model.intercept_)