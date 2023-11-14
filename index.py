
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        a = os.path.join(dirname, filename)
         
path = pd.read_csv(a)




X_train, X_test, Y_train, Y_test = train_test_split(path[['age', 'goout']], path[['Dalc']], test_size=0.33, random_state=0)
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)


kmeans = KMeans(n_cluster = 3, random_state = 0, n_init = 'auto')
kmeans.fit(X_train_norm)


sns.scatterplot(data = path, x = 'Dalc', y= 'age', hue = kmeans.labels_)