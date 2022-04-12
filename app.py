from os import name
from statistics import mean
from types import MethodType
from typing import List
from flask import Flask , render_template , request , jsonify
from flask.wrappers import Response
import joblib
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
import pandas as pd 
import numpy as np
from csv import reader
import requests
import io
import jinja2
import json
import plotly
import plotly.express as px
import math
import pickle



app = Flask(__name__)

global E_data 
global G_data

@app.route("/")
def fun():
    return render_template("allset.html")

@app.route("/Registration" , methods =['POST'])
def reg():
    return render_template("/Registration.html")

@app.route("/developer" , methods =['POST'])
def reg1():
    return render_template("/developer.html")


@app.route("/test" , methods =['POST'])
def feed():
    return render_template("/test.html")


@app.route("/home" , methods = ['POST'])
def back():
    return render_template("/allset.html")

@app.route("/allset" , methods = ['GET'])
def predict_again():
    return render_template("/allset.html") 

@app.route("/home" , methods = ['GET'])
def feed_back():
    return render_template("/allset.html")

@app.route("/demo", methods = ['POST'])
def upload():
    return render_template("/demo.html")

@app.route("/demo", methods = ['GET'])
def upload1():
    return render_template("/demo.html")

#PCP prediction

@app.route("/index" , methods = ["POST"])
def pc():
    return render_template('/index.html')

pcp_m = 'C:/Users/Cabinet/Documents/visual studio codes/Copy of main project/Model deployment/PCP_model.pkl'

model = joblib.load(pcp_m)

@app.route('/prediction1',methods=['POST'])
def prediction1():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]  
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    output = round(prediction[0], 2)
    y_prob = round(y_prob_success[0], 3)
    y_prob = y_prob * 100
    p_cnt = " % "

    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE {:.2f}{}'.format(y_prob, p_cnt), x ='{:.1f}{}'.format(y_prob, p_cnt),y =y_prob)
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE {:.2f}{}'.format(y_prob, p_cnt), x ='{:.1f}{}'.format(y_prob, p_cnt),y =y_prob)
    
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


@app.route('/index', methods=['GET'])
def dd():
    return render_template('/index.html')


@app.route('/pcpaccuracy', methods=['GET'])
def dd1():
    Table_algo = ['Logistic Regression','Support Vector Machine','Decision Tree','Random Forest','Naive Bayes','K-Nearest Neighbour']
    Table_cell = ['90%','80%','80%','80%','80%','75%']  
    return render_template('/pcpaccuracy.html', C1 =Table_algo,C2=Table_cell)

# BCP module
model1 = pickle.load(open('bcpmodel.pkl', 'rb'))
scaler = pickle.load(open('features.pkl', 'rb'))

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model1.predict(final_features)
    y_probabilities_test = model1.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    output = round(prediction[0], 2)
    y_prob = round(y_prob_success[0], 3) * 100
    p_cnt = " % "

    if output == 0:
        return render_template('bcpindex.html', prediction_text='THE PATIENT HAVE BENIGN CANCER {:.2f}{}'.format(y_prob, p_cnt), x ='{:.1f}{}'.format(y_prob, p_cnt),y =y_prob )
    else:
         return render_template('bcpindex.html', prediction_text='THE PATIENT HAVE A MALIGNANT CANCER {:.2f}{}'.format(y_prob, p_cnt), x ='{:.1f}{}'.format(y_prob, p_cnt),y =y_prob)
        
@app.route('/predict_api',methods=['POST'])
def predict_api1():

    data = request.get_json(force=True)
    prediction = model1.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/bcpaccuracy', methods=['GET'])
def pcpacc():
    Table_algo = ['Logistic Regression','Support Vector Machine','K-Nearest Neighbour','Random Forest']
    Table_cell = ['98%','97%','97%','95%']
    return render_template('/bcpaccuracy.html', C1 =Table_algo,C2=Table_cell)

@app.route('/bcpindex', methods=['GET'])
def pcpacc1():
    return render_template('/bcpindex.html')

@app.route('/bcpindex', methods=['POST'])
def pcpacc2():
    return render_template('/bcpindex.html')

# lung cancer prediction 
@app.route("/lcpaccuracy" ,methods = ['POST'])
def chart():
    Table_algo = ['Simple Logistic Regression','Logistic Regression','Support Vector Machine','K-Nearest Neighbour','Random Forest','Decision Tree']
    Table_cells = ['96%','91%','90%','89%','89%','87%']
    return render_template("lcpaccuracy.html",C1=Table_algo,C2=Table_cells)

@app.route("/lcpindex" ,methods = ['POST'])
def lcp():
    return render_template("lcpindex.html")

@app.route("/lcpindex" ,methods = ['GET'])
def chart12():
    return render_template("lcpindex.html")


lcp_m = 'C:/Users/Cabinet/Documents/visual studio codes/Copy of main project/Model deployment/LCP_lr_model.pkl'
model3 = joblib.load(lcp_m)

@app.route('/lcppredict',methods=['POST'])
def lcppredict():
    

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]  
    prediction = model3.predict(final_features)
    y_probabilities_test = model3.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    output = round(prediction[0], 2)
    y_prob=round(y_prob_success[0], 3)
    y_prob= y_prob * 100
    p_cnt= " % "

    if output == 0:
        return render_template('lcpindex.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE {:.2f}{}'.format(y_prob, p_cnt), x ='{:.1f}{}'.format(y_prob, p_cnt),y =y_prob)
    else:
         return render_template('lcpindex.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE {:.2f}{}'.format(y_prob, p_cnt), x ='{:.1f}{}'.format(y_prob, p_cnt),y =y_prob)
    
@app.route('/predict_api',methods=['POST'])
def predict_api2():
    data = request.get_json(force=True)
    prediction = model3.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)





# Analsysis module

@app.route("/analysis" , methods = ['POST'])
def uploaded():
    if request.method == "POST":
        url = request.form["link"]
        if url is not None:
            s=requests.get(url).content
            fileData1 =pd.read_csv(io.StringIO(s.decode('utf-8')))

            #removing unnamed attributes
            fileData1 = fileData1.loc[:, ~fileData1.columns.str.contains('^Unnamed')]
            fileData12 = fileData1.loc[:, ~fileData1.columns.str.contains('^id')]


            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            # collecting column names
            colname=[] 
            for col in fileData1.columns:
                colname.append(col)

            # table dataset size
            dim = fileData1.shape
            dim_list_key = ['Rows','Columns']
            dim_list_val = [dim[0],dim[1]]

            # Before handling missing values
            n_case_key = list(fileData1)
            n_case_value = list(fileData1.isnull().sum(axis = 0))

           
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            newdf = fileData1.select_dtypes(include=numerics)
            newdf1 = fileData1.select_dtypes(include=['object','bool'])
            


            # data preparation for statistical evalution 
            newdf1 = fileData12.select_dtypes(include=numerics)


            # finding mean 
            mean_key = list(newdf1)
            mean_value = list(round(newdf1.mean(axis=0),2))
            


            #finding max
            max_key = list(newdf1)
            max_value = list((round(newdf1.max(axis=0),2)))
            
            
            #finding min
            min_key = list(newdf1)
            min_value = list(round(newdf1.min(axis=0),2))
            


            fileData1 = fileData1.fillna(newdf.mean())
            for col in newdf1:
                fileData1[col].fillna(str(fileData1[col].mode()),inplace = True)
            

            # removing attribute with 80% null cases.
            limitPer = len(fileData1) * .80
            fileData1 = fileData1.dropna(thresh=limitPer, axis=1)


            # After Handling null cases
            global E_data 
            E_data = fileData1
            n_case_key1 = list(fileData1)
            n_case_value1 = list(fileData1.isnull().sum(axis = 0))
            

            
            in_f_cols = list(newdf)
            newdf1 = fileData1.select_dtypes(object)

            # 1st 10 rows & last 10 rows
            x = list(fileData1)
            data = fileData1.head(10)
            l_data = fileData1.tail(10)
            tuple_rows = [tuple(row) for row in data.values]
            l_tuple_rows = [tuple(row) for row in l_data.values]

            # table for unique values
            l_uni_val = []
            y=list(newdf1)
            for col in y:
                l_uni_val.append(list(newdf1[col].unique()))
            tuple_rows1 = [tuple(row) for row in l_uni_val]

            

            return render_template("analysis.html", A = colname ,D2 = dim_list_key, D3 =dim_list_val,C1 = n_case_key,C2 = n_case_value, C3 = n_case_key1, C4= n_case_value1,C5=mean_key,C6=mean_value,C7 = max_key,C8=max_value,C9=min_key,C10=min_value,D= in_f_cols, E = list(newdf1), headings = x , data = tuple_rows, l_data = l_tuple_rows ,G = y, F =tuple_rows1  )

    return render_template("demo.html")


@app.route("/analysis1" , methods = ['POST'])
def uploaded1():
    if request.method == "POST" :
        G_url = request.form["link1"]
        if G_url is not None:
            #G_url != "" or url != " "
            file_id=G_url.split('/')[-2]
            dwn_url='https://drive.google.com/uc?id=' + file_id

            
            fileData2 = pd.read_csv(dwn_url)
            
            #removing unnamed attributes
            fileData2 = fileData2.loc[:, ~fileData2.columns.str.contains('^Unnamed')]

            fileData21 = fileData2.loc[:, ~fileData2.columns.str.contains('^id')]
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            colname=[] 
            for col in fileData2.columns:
                colname.append(col)

            # dataset dimesion
            # table dataset size
            dim = fileData2.shape
            dim_list_key = ['Rows','Columns']
            dim_list_val = [dim[0],dim[1]]
            
            #before handling
            n_case_key = list(fileData2)
            n_case_value = list(fileData2.isnull().sum(axis = 0))
            
            
            # After handling missing values
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            newdf = fileData2.select_dtypes(include=numerics)
            newdf1 = fileData2.select_dtypes(include=['object','bool'])

            fileData2 = fileData2.fillna(newdf.mean())
            
            for col in newdf1:
                fileData2[col].fillna(str(fileData2[col].mode().values[0]),inplace = True)

            
            
            # data preparation for statistical evalution 
            newdf1 = fileData21.select_dtypes(include=numerics)


            # mean
            mean_key = list(newdf1)
            mean_value = list(round(newdf1.mean(axis=0),2))
            

            #max
            max_key = list(newdf1)
            max_value = list((round(newdf1.max(axis=0),2)))
            
            
            #min
            min_key = list(newdf1)
            min_value = list(round(newdf1.min(axis=0),2))

            #after handling

            limitPer = len(fileData2) * .80
            fileData2 = fileData2.dropna(thresh=limitPer, axis=1)
            
            global G_data
            G_data = fileData2
            n_case_key1 = list(fileData2)
            n_case_value1 = list(fileData2.isnull().sum(axis = 0))
            
            
            


            in_f_cols = list(newdf)
            newdf1 = fileData2.select_dtypes(object)

            x = list(fileData2)
            data = fileData2.head(10)
            l_data = fileData2.tail(10)
            tuple_rows = [tuple(row) for row in data.values]
            l_tuple_rows = [tuple(row) for row in l_data.values]

             
            l_uni_val = []

            y=list(newdf1)
            for col in y:
                l_uni_val.append(list(newdf1[col].unique()))
            tuple_rows1 = [tuple(row) for row in l_uni_val]

            
            
            

            return render_template("analysis1.html", A = colname ,D2 = dim_list_key, D3 =dim_list_val,C1 = n_case_key,C2 = n_case_value, C3 = n_case_key1, C4= n_case_value1,C5=mean_key,C6=mean_value,C7 = max_key,C8=max_value,C9=min_key,C10=min_value,D= in_f_cols, E = list(newdf1), headings = x , data = tuple_rows, l_data = l_tuple_rows ,G = y, F =tuple_rows1  )
        
    return render_template("demo.html")

@app.route("/visualize" , methods = ['POST'])
def visualize():
    if E_data is not None:
        str_newdf = E_data.select_dtypes(include=['object'])
        str_newdf = str_newdf.loc[:, ~str_newdf.columns.str.contains('^Unnamed')]
        srt_head = list(str_newdf)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        num_newdf = E_data.select_dtypes(include=numerics)
        num_newdf = num_newdf.loc[:, ~num_newdf.columns.str.contains('^Unnamed')]
        num_head = list(num_newdf)
        print(num_newdf)

        fig1 = px.histogram(num_newdf , title= "Histogram")
        fig2 = px.density_heatmap(num_newdf , title= " Density-HeatMap")
        fig3 = px.bar(num_newdf , title= " Bar Graph")
        fig4 = px.box(num_newdf, title="Box Plot")
        fig5 = px.scatter(num_newdf , title="Scatter plot")
        fig6 = px.violin(num_newdf, title= "Violin Plot")
        fig7 = px.funnel(num_newdf, title= " Funnel chart")


        graph1JSON = json.dumps(fig1, cls= plotly.utils.PlotlyJSONEncoder)
        graph2JSON = json.dumps(fig2, cls= plotly.utils.PlotlyJSONEncoder)
        graph3JSON = json.dumps(fig3, cls= plotly.utils.PlotlyJSONEncoder)
        graph4JSON = json.dumps(fig4, cls= plotly.utils.PlotlyJSONEncoder)
        graph5JSON = json.dumps(fig5, cls= plotly.utils.PlotlyJSONEncoder)
        graph6JSON = json.dumps(fig6, cls= plotly.utils.PlotlyJSONEncoder)
        graph7JSON = json.dumps(fig7, cls= plotly.utils.PlotlyJSONEncoder)
        
    

        return render_template("visualize.html",graph1 = graph1JSON ,graph2 = graph2JSON, graph3 = graph3JSON, graph4 = graph4JSON ,graph5 = graph5JSON , graph6 = graph6JSON, graph7 = graph7JSON )

@app.route("/visualize1" , methods = ['POST'])
def visualize1():   
    if G_data is not None:
        str_newdf = G_data.select_dtypes(include=['object'])
        str_newdf=str_newdf.loc[:, ~str_newdf.columns.str.contains('^Unnamed')]
        srt_head = list(str_newdf)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        num_newdf = G_data.select_dtypes(include=numerics)
        num_newdf = num_newdf.loc[:, ~num_newdf.columns.str.contains('^Unnamed')]
        num_head = list(num_newdf)
        print(num_newdf)

        fig1 = px.histogram(num_newdf , title= "Histogram")
        fig2 = px.density_heatmap(num_newdf , title= " Density-HeatMap")
        fig3 = px.bar(num_newdf , title= " Bar Graph")
        fig4 = px.box(num_newdf, title="Box Plot")
        fig5 = px.scatter(num_newdf , title="Scatter plot")
        fig6 = px.violin(num_newdf, title= "Violin Plot")
        fig7 = px.funnel(num_newdf, title= " Funnel Chart")
        



        graph1JSON = json.dumps(fig1, cls= plotly.utils.PlotlyJSONEncoder)
        graph2JSON = json.dumps(fig2, cls= plotly.utils.PlotlyJSONEncoder)
        graph3JSON = json.dumps(fig3, cls= plotly.utils.PlotlyJSONEncoder)
        graph4JSON = json.dumps(fig4, cls= plotly.utils.PlotlyJSONEncoder)
        graph5JSON = json.dumps(fig5, cls= plotly.utils.PlotlyJSONEncoder)
        graph6JSON = json.dumps(fig6, cls= plotly.utils.PlotlyJSONEncoder)
        graph7JSON = json.dumps(fig7, cls= plotly.utils.PlotlyJSONEncoder)
        
    

        return render_template("visualize1.html",graph1 = graph1JSON ,graph2 = graph2JSON, graph3 = graph3JSON, graph4 = graph4JSON ,graph5 = graph5JSON, graph6 = graph6JSON, graph7 = graph7JSON)


@app.route("/analysis" , methods = ['GET'])
def get_back():
    return render_template("analysis.html")

@app.route("/analysis1" , methods = ['GET'])
def get_back1():
    return render_template("analysis1.html")


if __name__ == "__main__":
    app.run(debug=True)  

