from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


global X, Y, dataset, X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, algorithm_name, sensitivity, specificity, classifier

le1 = LabelEncoder()
le2 = LabelEncoder()

np.set_printoptions(suppress=True)

def ProcessData(request):
    if request.method == 'GET':
        global X, Y, dataset, X_train, X_test, y_train, y_test
        dataset = pd.read_csv("Dataset/Flood.csv")
        dataset.fillna(0, inplace = True)
        dataset['SUBDIVISION'] = pd.Series(le1.fit_transform(dataset['SUBDIVISION'].astype(str)))
        dataset['FLOODS'] = pd.Series(le2.fit_transform(dataset['FLOODS'].astype(str)))
        label = dataset.groupby('FLOODS').size()
        columns = dataset.columns
        temp = dataset.values
        dataset = dataset.values
        X = dataset[:,0:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        X = normalize(X)
        Y = Y.astype(int)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(columns)):
            output += "<th>"+font+columns[i]+"</th>"            
        output += "</tr>"
        for i in range(len(temp)):
            output += "<tr>"
            for j in range(0,temp.shape[1]):
                output += '<td><font size="" color="black">'+str(temp[i,j])+'</td>'
            output += "</tr>"    
        context= {'data': output}
        label.plot(kind="bar")
        plt.title("Flood Graph, 0 (No Flood) & 1 (Flood)")
        plt.show()
        return render(request, 'UserScreen.html', context)
        
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    cm = confusion_matrix(y_test, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1])
    sp = cm[1,1]/(cm[1,0]+cm[1,1])
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    algorithm_name.append(algorithm)
    sensitivity.append(se)
    specificity.append(sp)

def TrainML(request):
    global accuracy, precision, recall, fscore, algorithm_name, sensitivity, specificity, classifier
    accuracy = []
    precision = []
    recall = []
    fscore = []
    sensitivity = []
    specificity = []
    algorithm_name = []
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        predict = lr.predict(X_test)
        calculateMetrics("Logistic Regression", predict, y_test)

        svm_cls = SVC(kernel="rbf")
        svm_cls.fit(X_train, y_train)
        predict = svm_cls.predict(X_test)
        calculateMetrics("SVM", predict, y_test)

        knn = KNeighborsClassifier(n_neighbors = 2)
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        calculateMetrics("KNN", predict, y_test)

        mlp = MLPClassifier(max_iter=2000)
        mlp.fit(X, Y)
        predict = mlp.predict(X_test)
        classifier = mlp
        calculateMetrics("MLP", predict, y_test)
        
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Sensitivity', 'Specificity']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithm_name)):
            output +="<tr><td>"+font+str(algorithm_name[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td><td>"+font+str(sensitivity[i])+"</td><td>"+font+str(specificity[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def Forecast(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global classifier
        testFile = request.POST.get('t1', False)
        test = pd.read_csv("Dataset/testData.csv")
        test.fillna(0, inplace = True)
        test['SUBDIVISION'] = pd.Series(le1.transform(test['SUBDIVISION'].astype(str)))
        test = test.values
        test = normalize(test)
        predict = classifier.predict(test)
        print(predict)
        arr = ['Test Data', 'Flood Forecast']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        labels = ['No Flood Occur', 'Flood May Occur'] 
        for i in range(len(predict)):
            if predict[i] == 0:
                font1 = '<font size="" color="green">'
                output +="<tr><td>"+font+str(test[i])+"</td><td>"+font1+str(labels[predict[i]])+"</td></tr>"
            if predict[i] == 1:
                font1 = '<font size="" color="red">'
                output +="<tr><td>"+font+str(test[i])+"</td><td>"+font1+str(labels[predict[i]])+"</td></tr>"    
        context= {'data': output}    
        return render(request, 'UserScreen.html', context) 


def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})


def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Flood',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+uname}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed. Please retry'}
            return render(request, 'UserLogin.html', context)        

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Flood',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Flood',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Signup.html', context)
      


