import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


df = pd.read_csv('C:/Users/Aju/Desktop/ML_project/HR_data.csv', index_col=None)
df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales': 'department',
                        'left': 'turnover'
                        })

front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)
turnover_rate = df.turnover.value_counts() / 14999
print(turnover_rate)

# Create train and test splits
target_name = 'turnover'
#dropping string variables and target variable
X = df.drop(['turnover','department','salary'], axis=1)
y=df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

# Applying PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


#dtree classifier
dtree = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
dtree.fit(X_train, y_train)
print ("\n\n ---Decision Tree Model---")
dt_roc_auc =accuracy_score(y_test, dtree.predict(X_test))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(y_test, dtree.predict(X_test)))


#bayesian classfier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print ("\n\n ---Bayesian Classifier Model---")
gnb_roc_auc = accuracy_score(y_test, gnb.predict(X_test))
print ("Bayesian Model AUC = %2.2f" % gnb_roc_auc)
print(classification_report(y_test, gnb.predict(X_test)))

