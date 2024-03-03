import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.svm import SVC
data = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")
print(data)
print(data.columns)
print(data.dtypes)
x = data.drop(columns='Class')
y = data['Class']
y_labelled = {'Çerçevelik': 0, 'Ürgüp Sivrisi' : 1}
y = y.map(y_labelled)
print(data)
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.2,random_state=42)

scaler = StandardScaler().fit(x_train)
x_scaled = scaler.transform(x_train)
y_scaled = scaler.transform(x_test)
C = [100, 10, 1.0, 0.1, 0.01]
multi_class = ('auto', 'ovr', 'multinomial')
fit_intercept= [True,False]
results=[]
for i in C:
    for f in fit_intercept:
        for max_iter in range(100, 200):
                 model = LogisticRegression(C=i,fit_intercept=f,max_iter=max_iter,solver='liblinear').fit(x_scaled, y_train)
                 score = model.score(x_scaled, y_train)
                 test = model.score(y_scaled, y_test)
                 # print(score)
                 results.append((i, f,max_iter, score, test))
results2 = pd.DataFrame(results, columns=['C', 'fit_intercept','max_iter ','score', 'test'])
#print(results2)
results2.to_csv('seed3.csv')
criterion=['gini', 'entropy', 'log_loss']
splitter=['best', 'random']
max_features=['auto', 'sqrt', 'log2']
results=[]
for i in criterion:
    for s in splitter:
            for max_depth in range(1, 20):
                for max_leaf_nodes in range(2, 10):
                    for min_samples_split in range(2, 80):
                        model = DecisionTreeClassifier(criterion=i,splitter=s,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,min_samples_leaf=min_samples_split).fit(x_scaled,y_train)
                        score = model.score(x_scaled,y_train)
                        test = model.score(y_scaled,y_test)
                        results.append((i,s,max_depth,max_leaf_nodes,min_samples_split,score,test))
results2=pd.DataFrame(results, columns=['criterion','splitter','max_depth','max_leaf_nodes','min_sample_split','score','test'])
results2.to_csv('seed4.csv')
from sklearn.ensemble import RandomForestClassifier
min_samples_leaf = [10,20,30,40,50,60,70,80]
results = []
n_estimators=[50,100,150,200,250,300]

for i in min_samples_leaf:
    for n in n_estimators:
        for max_depth in range(1, 10):
            for max_leaf_nodes in range(2, 10):
               model = RandomForestClassifier(min_samples_leaf=i,n_estimators=n,max_leaf_nodes=max_leaf_nodes,max_depth=max_depth).fit(x_scaled,y_train)
               score = model.score(x_scaled,y_train)
               test = model.score(y_scaled,y_test)
               # print(score)
               results.append((i,n,max_leaf_nodes,max_depth,score,test))
results2 = pd.DataFrame(results,columns=['i','n','min_leaf_nodes','max_depth','score','test'])
#print(results2)
results2.to_csv('seed5.csv')

c_range = [0.1, 1, 10, 100]
kernel_range = ['linear', 'rbf', 'poly', 'sigmoid']
gamma_range = ['scale', 'auto']
class_weight =['dict','balanced']

results = []
from sklearn.svm import SVC

for c in c_range:
    for kernel in kernel_range:
        for gamma in gamma_range:
            for degree in range(1, 10):
                for class_weight in class_weight:
                    svm = SVC(C=c, kernel=kernel, gamma=gamma,degree=degree,class_weight=class_weight).fit(x_scaled,y_train)
                    train = svm.score(x_scaled,y_train)
                    test = svm.score(y_scaled,y_test)
                    results.append((c, kernel, gamma,degree,class_weight, train, test))

results_df = pd.DataFrame(results, columns=['C', 'Kernel', 'Gamma', 'degree','class weight','train score', 'test score'])
#print(results_df.to_string(index=False))
results_df.to_csv('drybean6.csv', index=False)

from sklearn.neighbors import KNeighborsClassifier
n_neighbours = [1,2,3,4,5]
weights=['uniform','distance']
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
leaf_size = [20,30,35,40,45,50]
results = []
for n in n_neighbours:
    for w in weights:
       for a in algorithms:
           for leaf_size in leaf_size:
            model = KNeighborsClassifier(n_neighbors=n,weights=w,algorithm=a,leaf_size=leaf_size).fit(x_scaled,y_train)
            score1 = model.score(x_scaled, y_train)
            score2 = model.score(y_scaled, y_test)
            results.append((n, a, w,leaf_size,score1,score2))
results_df = pd.DataFrame(results, columns=['n_neighbors', 'algorithms','weights', 'leaf size','train score', 'test score'])
#print(results_df.to_string(index=False))
results_df.to_csv('drybean7.csv', index=False)