# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:24:52 2020

@author: Nikhitha Gururaj
"""

import numpy as np
from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import LabelEncoder
import feedparser

app = Flask(__name__)

#load the model
dtc = joblib.load('ckd_dtc.sav')

url = "https://news.google.com/rss/search?q=chronic+kidney+disease&hl=en-IN&gl=IN&num=10&ceid=IN:en"

class ParseFeed():

    def __init__(self, url):
        self.feed_url = url

    def parse(self):
        '''
        Parse the URL, and print all the details of the news 
        '''
        feeds = feedparser.parse(self.feed_url).entries
        feeds_list = {}
        counter = 0
        for f in feeds:
            if(counter == 10):
                break
            feeds_list[counter] = {
                'Published Date': f.get("published", ""),
                'Title': f.get("title", ""),
                'Url': f.get("link", "")
            }
            counter += 1
        return feeds_list

@app.route('/')
def home():
    return render_template('ckd.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/test')
def test():
    return render_template('form.html')

@app.route('/news')
def news_feed():
    feed = ParseFeed(url)
    feeds_list = feed.parse()
    return render_template('news.html', results = feeds_list)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    #x_test = [[80,"female","other",80,1.02,0.0,0.0,26,20,15.8,49,6600,5.4,'yes']]
    print(x_test)
    
    le = LabelEncoder()
    le.classes_ = np.load('le_for_htn.npy', allow_pickle=True)

    htn = le.transform([x_test[0][11]])
    
    age = float(x_test[0][0])
    bp = float(x_test[0][3]) 
    sg = float(x_test[0][4])
    al = float(x_test[0][5])
    su = float(x_test[0][6])
    bu = float(x_test[0][7])
    sc = float(x_test[0][8])
    hemo = float(x_test[0][9])
    pcv = int(x_test[0][10])
    wcc = int(x_test[0][12])
    rcc = float(x_test[0][13])
    
    gender = x_test[0][1]
    ethn = x_test[0][2]
    
    prediction = dtc.predict([[age, bp, sg, al, su, bu, sc, hemo, pcv, wcc, rcc, htn]])
    print(prediction)
    op_class = ['ckd','notckd']
    output = op_class[prediction[0]]
    stage_val =''
    if(output=="ckd"):
        stage = 175*(sc**(-1.154))*(age**(-0.203))
        if(gender == "female"):
            stage = stage*0.742
        if(ethn == "black"):
            stage = stage*1.212
        print('stage:',stage)
    
        if(stage>90): stage_val = "I"
        elif(stage>=60): stage_val = "II"
        elif(stage>=30): stage_val = "III"
        elif(stage>=15): stage_val = "IV"
        else: stage_val = "V"
    return render_template('form.html', prediction = output, stage = stage_val)
    


if __name__ == "__main__":
    app.run(debug=True)
