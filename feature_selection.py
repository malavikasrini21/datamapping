import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#import sklearn 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
import mlxtend
from mlxtend.plotting import plot_decision_regions

#from sklearn.model_selection import train_test_split

df=pd.read_csv('TargetDataBasecsv.csv')
#print(df)
ym=df.columns.values.tolist()
#print(ym)
da=df['Acct_FName']
le=[]
for j in ym:
    for i in range(len(da)):
        le.append(j)
    dq_final = pd.DataFrame(le,columns=['Targetname'])

#print(dq_final)
le=[]
for k in ym:
    
    for l in df[k]:
        le.append(l)
    ds= pd.DataFrame(le,columns=['X'])  
#print(ds)

df_index = pd.merge(ds, dq_final, right_index=True, left_index=True)

print(df_index)

jk=[]
for i in df_index["X"]:
    try:
        jk.append(len(i))
        #kl=pd.DataFrame(jk,columns=['leng'])
        
    except:
        jk.append(0)
        #kl=pd.DataFrame(jk,columns=['leng'])
import numpy as np
kl = pd.DataFrame(np.array([jk]).T)
kl.columns =['leng']

#print(type(jk))

#print(len(jk))
#print(kl)

jk=[]
for i in df_index["X"]:
    try:
        c=0
        for j in i:
            if j==' ':
                c=c+1
        jk.append(c)
        
        
    except:
        jk.append(0)

ty = pd.DataFrame(np.array([jk]).T)
ty.columns =['Spaces']
#print(ty)

df_fin = pd.merge(ds,kl, right_index=True, left_index=True)
#print(df_fin)

df_final = pd.merge(df_fin,ty, right_index=True, left_index=True)
#print(df_final)

df_feature = pd.merge(df_final,dq_final, right_index=True, left_index=True)
#print(df_feature)

Le=LabelEncoder()
df_feature['Targetname']=Le.fit_transform(df_feature['Targetname'])

ds_fina=df_feature.drop('Targetname',axis=1)
ds_final=ds_fina.drop('X',axis=1)

print(ds_final)

le=[]
for i in df_feature['Targetname']:
    le.append(i)
da_final = pd.DataFrame(le,columns=['Target'])
#print(da_final)
refftab = pd.merge(dq_final,da_final, right_index=True, left_index=True)

print(refftab)

fs=[]
a=[]
for i in df_index["X"]:
    try:
        c=0
    
        for j in i:
            
            
            if j.isdigit() or j=='-':
                a.append(j)
                c=1
        fs.append(c)
        
        
    except:
        fs.append(0)
#print(a)

newfs = pd.DataFrame(np.array([fs]).T)
newfs.columns =['Only digits']
#print(newfs)
feature=pd.merge(ds_final,newfs, right_index=True, left_index=True)
#print(feature)


x=feature
y=da_final['Target']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#decission tree

x=ds_final

y=da_final['Target']


clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

score = accuracy_score(clf.predict(x), y)
print(score)


'''
def knn_comparison(data, k):
 x = data[['x_test','y_test']].values
 y = data['class'].astype(int).values
 clf = neighbors.KNeighborsClassifier(n_neighbors=k)
 clf.fit(x, y)# Plotting decision region
 plot_decision_regions(x, y, clf=clf, legend=2)# Adding axes annotations
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.title('Knn with K='+ str(k))
 plt.show()

data1 = pd.read_csv('TargetDataBasecsv.csv')
for i in [1,5,20,30,40,80]:
    knn_comparison(data1, i)

'''
ch=int(input("enter your choice"))
if ch==1:
    plt.plot(y_test,y_pred,'*')
    plt.show()
if ch==2:
    plt.plot(clf.predict(x),y)
    plt.show

#check expected and predicted value

kk=[]
expect=y_test.index
for i in y_pred:
    kk.append(refftab.loc[refftab['Target']==i,'Targetname'].iloc[0])
predicted=pd.DataFrame(kk,columns=['Predicted'])

kk=[]
for i in expect:
    kk.append(dq_final._get_value(i,'Targetname'))
expected=pd.DataFrame(kk,columns=['Expected'])
show=pd.merge(expected,predicted,right_index=True,left_index=True)
print(show)






