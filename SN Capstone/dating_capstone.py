import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Can religion be predicted by drinks, drugs, and orientation?
# Can income be predicted by drinks, drugs, religion, and orientation?

df = pd.read_csv("profiles.csv")

##########################################################################################################
# Preprocessing of data and defining new columns

df['religion'].fillna('unknown', inplace=True)
religions = df['religion'].tolist()

for i in range(len(religions)):
    if 'agnosticism' in religions[i]:
        religions[i] = 'agnosticism'
    elif 'christianity' in religions[i]:
        religions[i] = 'christian_other'
    elif 'catholicism' in religions[i]:
        religions[i] = 'catholicism'
    elif 'atheism' in religions[i]:
        religions[i] = 'atheism'
    elif 'buddhism' in religions[i]:
        religions[i] = 'buddhism'
    elif 'judaism' in religions[i]:
        religions[i] = 'judaism'
    elif 'other' in religions[i]:
        religions[i] = 'other'
    elif 'hinduism' in religions[i]:
        religions[i] = 'hinduism'
    elif 'islam' in religions[i]:
        religions[i] = 'islam'

df['religions_simple'] = religions

rel_bar_names = df.religions_simple.value_counts().index
rel_bar_values = df.religions_simple.value_counts().tolist()
drink_bar_names = df.drinks.value_counts().index
drink_bar_values = df.drinks.value_counts().tolist()

plt.bar(rel_bar_names, height=rel_bar_values)
plt.xlabel("Religions")
plt.ylabel("Number of People")
plt.title("Number of people in each religious group")
plt.show()
plt.bar(drink_bar_names, height=drink_bar_values, color="orange")
plt.xlabel("Drinking Habit")
plt.ylabel("Number of People")
plt.title("Number of people catergorised into drinking habits")
plt.show()

religion_mapping = {"unknown": 0, "agnosticism": 1, "other": 2, "atheism": 3, "christian_other": 4, "catholicism": 5,
                    "judaism": 6, "buddhism": 7, "hinduism": 8, "islam": 9}
df["religions_simple_code"] = df.religions_simple.map(religion_mapping)

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)

drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping)

orientation_mapping = {"straight": 0, "gay": 1, "bisexual": 2}
df["orientation_code"] = df.orientation.map(orientation_mapping)

##########################################################################################################
# can religion be predicted by drinking, drug use and orientation?


religion_features = df[['drinks_code', 'drugs_code', 'orientation_code', "religions_simple_code"]]
religion_features.dropna(how='any', inplace=True)
labels = religion_features["religions_simple_code"]
religion_features = religion_features[['drinks_code', 'drugs_code', 'orientation_code']]
rel_feat_values = religion_features.values
min_max_scaler = preprocessing.MinMaxScaler()
rel_feat_val_scaled = min_max_scaler.fit_transform(rel_feat_values)
religion_features = pd.DataFrame(rel_feat_val_scaled, columns=religion_features.columns)

rel_training_labels = labels.values.tolist()
rel_training_data = religion_features
rel_X_train, rel_X_test, rel_y_train, rel_y_test = train_test_split(rel_training_labels, rel_training_data,
                                                                    test_size=0.2, shuffle=True)

for k in range(1, 101, 10):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(rel_y_train, rel_X_train)
    rel_predicted = classifier.predict(rel_y_test)
    rel_predicted = rel_predicted.tolist()
    print(k)
    print(classifier.score(rel_y_test, rel_X_test))
    print(f1_score(rel_X_test, rel_predicted, average='macro'))

for gamma in range(1, 101, 20):
    for C in range(1, 10):
        classifier = SVC(gamma=gamma, C=C)
        classifier.fit(rel_y_train, rel_X_train)
        rel_predicted = classifier.predict(rel_y_test)
        rel_predicted = rel_predicted.tolist()
        print(gamma)
        print(C)
        print(classifier.score(rel_y_test, rel_X_test))
        print(f1_score(rel_X_test, rel_predicted, average='macro'))

classifier = KNeighborsClassifier(n_neighbors=71)
classifier.fit(rel_y_train, rel_X_train)
rel_predicted = classifier.predict(rel_y_test)
rel_predicted = rel_predicted.tolist()
print(classifier.score(rel_y_test, rel_X_test))
print(f1_score(rel_X_test, rel_predicted, average='macro'))

classifier = SVC()
classifier.fit(rel_y_train, rel_X_train)
rel_predicted = classifier.predict(rel_y_test)
rel_predicted = rel_predicted.tolist()
print(classifier.score(rel_y_test, rel_X_test))
print(f1_score(rel_X_test, rel_predicted, average='macro'))

################################################################################################################
# Can income be predicted by drinks, drugs, religion, and orientation?

income_features = df[['drinks_code', 'drugs_code', 'orientation_code', "religions_simple_code", "income"]]
income_features.dropna(how='any', inplace=True)
inc_labels = income_features["income"]
income_features = income_features[['drinks_code', 'drugs_code', 'orientation_code', "religions_simple_code"]]
inc_feat_values = income_features.values
inc_feat_val_scaled = min_max_scaler.fit_transform(inc_feat_values)
income_features = pd.DataFrame(inc_feat_val_scaled, columns=income_features.columns)

inc_training_y = inc_labels.values.tolist()
inc_training_data = income_features

inc_X_train, inc_X_test, inc_y_train, inc_y_test = train_test_split(inc_training_data, inc_training_y, test_size=0.2,
                                                                    shuffle=True)

mlr = LinearRegression()
mlr.fit(inc_X_train, inc_y_train)
inc_predicted = mlr.predict(inc_X_test)
print("MLR Test score:")
print(mlr.score(inc_X_test, inc_y_test))

for k in range(1, 10):
    knr = KNeighborsRegressor(weights="distance", n_neighbors=k)
    knr.fit(inc_X_train, inc_y_train)
    inc_predicted_2 = knr.predict(inc_X_test)
    print(k)

knr = KNeighborsRegressor(weights="distance", n_neighbors=2)
knr.fit(inc_X_train, inc_y_train)
inc_predicted_2 = knr.predict(inc_X_test)
print("KNR Test score:")
print(knr.score(inc_X_test, inc_y_test))
