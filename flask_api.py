from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
from flask_cors import CORS
from flask import Flask, redirect, url_for, render_template, request

from flask import *
import json
from googletrans import Translator
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
chatbot=ChatBot('health bot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english.greetings",
			"chatterbot.corpus.english.conversations" )

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings(action = 'ignore')
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()

#initializing app and api
app = Flask(__name__)
app.secret_key='chatbot'
CORS(app)
api = Api(app)
app.static_folder = 'static'

def prediction(file):    
    clf = joblib.load('BestModel.pkl')
    new_pred = clf.predict([file])
    print(new_pred)
    new_prob = clf.predict_proba([file])
    test_confidance = 100*np.max(new_prob)
    acc = "{:.0f}".format(test_confidance)
    accuracy = "{:2f}%".format(test_confidance)
    print("Test_Accuracy:",accuracy)
    cs = 'confidance score:',accuracy
    
    if new_pred==[1]:
        print('The corona test patient is: COVID19')
        if int(acc) >=90:
            out = 'COVID19 High '
            print(out)
            return out,cs
        elif int(acc) <=75:
            out = 'COVID19 Low'
            print(out)
            return out,cs
        else:
            out = 'COVID19 Medium'
            print(out)
    else:
        out = 'Normal'
        print(out)
    res = out,cs
    return res
translator = Translator()
result = translator.translate('Hi everyone', src='en', dest='kn')
print(result)
detection = translator.detect("ಎಲ್ಲರಿಗೂ ನಮಸ್ಕಾರ")
print("Language code:", detection.lang)
result = translator.translate('ಎಲ್ಲರಿಗೂ ನಮಸ್ಕಾರ', src='kn', dest='en')
print(result)


d={
    'hello':'hello',
    'how are you':'I am good',
    'I am good':'how can I help you?',
    'I am fine':'how can I help you?'
}

@app.route('/')
def index():
   return render_template('index.html')

l1= [28.0, 0.0, 0.0, 1.0, 12.0, 0.0, 92.0, 1.0, 13.0, 97.0]
rr=prediction(l1)
print('test',rr[0])
covid_flag=False
covid_qus={
    'Age':0,
    'Sex':0,
    'neutrophil':0,
    'serumLevelsOfWhiteBloodCell':0,
    'lymphocytes':0,
    'Diarrhea':0,
    'Fever':0,
    'Coughing':0,
    'SoreThroat':0,
    'Temperature':0
}        
covid_count=0
l=[]
lang=''
df=pd.read_csv('drugs.csv')
lcon=df['condition']#.unique()
my_lang='en'
class process(Resource):
    def get(self):
        global covid_flag,covid_count,covid_qus,l,lcon,my_lang,lang
        covid=[]
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('task', required=True)  # add args    
        args = parser.parse_args()  # parse arguments to dictionary
        print('type',type(args))
        print('task',args['task'])
        task=str(args['task'])
        print('covid_flag',covid_flag)
        text=''
        res=''
        text=task
        if covid_flag==False:
            detection = translator.detect(task)
            
            print("Language code:", detection.lang)
            if detection.lang=='kn':
                lang='k'
                result = translator.translate(task, src='kn', dest='en')
                text=result.text
                print('user',text)
                response = chatbot.get_response(text)
                print('bot',response.text)
                result = translator.translate(response.text, src='en', dest='kn')
                res=lang+str(response.text)
                print(res)
                #res=str(response)
                my_lang='kn'
                if 'fever' in text or 'covid' in text:
                    covid_flag=True
            else:
                lang='e'
                text=task
                print('user',text)
                response = chatbot.get_response(text)
                #print('bot',response,type(response))
                print('bot',response)
                print('>>',response.text)
                #res='hello'
                #print('text',text)
                res=lang+str(response.text)
                if 'fever' in text or 'covid' in text:
                    covid_flag=True
        if 'recommend' in text.lower():
            ll=text.split()
            disease=ll[-1]
            print('disease',disease)
            print('conditions ',lcon)
            ind=[]
            for i in range(len(lcon)):
                try:
                    if disease in lcon[i].lower():
                        
                        ind.append(i)
                except:
                    pass
            drugs=[]
            for i in ind:
                drug_name=df['drugName'][i]
                review=df['review'][i]
                rating=df['rating'][i]
                score = analyzer.polarity_scores(review)
                p=float(score['pos'])
                ne=float(score['neu'])
                n=float(score['neg'])
                performance = [p,n,ne]
                index=performance.index(max(performance))
                if (index==0):
                    result='positive'
                elif (index==1):
                    result='negative'
                elif (index==2):
                    result='neutral'
                print('result',result)
                if result=='positive' or result=='neutral':
                    if int(rating)>5:
                        drugs.append(drug_name)
            djs=''
            for j in drugs:
                djs=djs+' '+j
            res='Recommended Medicine for  '+disease+' are '+djs
            print(res)
            print('l=',l)


        if 'fever' in text or 'covid' in text or  covid_flag==True:
            print('Ask Covid parameter',covid_count)
            covid_flag=True

            if covid_count==0:
                res=lang+'Enter your Age'
                covid_count=covid_count+1
                
            elif covid_count==1:
                res=lang+'Enter your Sex(F:0,M:1)'
                covid_count=covid_count+1
                covid_qus['Age']=text
                l.append(float(text))
            elif covid_count==2:
                res=lang+'Enter neutrophil value'
                covid_count=covid_count+1
                covid_qus['Sex']=text
                l.append(float(text))
            elif covid_count==3:
                res=lang+'Enter serumLevelsOfWhiteBloodCell value'
                covid_count=covid_count+1
                covid_qus['neutrophil']=text
                l.append(float(text))
            elif covid_count==4:
                res=lang+'Enter lymphocytes value'
                covid_count=covid_count+1
                covid_qus['serumLevelsOfWhiteBloodCell']=text
                l.append(float(text))
            elif covid_count==5:
                res=lang+'Enter Diarrhea value'
                covid_count=covid_count+1
                covid_qus['lymphocytes']=text
                l.append(float(text))
            elif covid_count==6:
                res=lang+'Enter Fever value (Y:1,N:0)'
                covid_count=covid_count+1
                covid_qus['Diarrhea']=text
                l.append(float(text))
            elif covid_count==7:
                res=lang+'Enter Coughing value(Y:1,N:0)'
                covid_count=covid_count+1
                covid_qus['Fever']=text
                l.append(float(text))
            elif covid_count==8:
                res=lang+'Enter SoreThroat value(Y:1,N:0)'
                covid_count=covid_count+1
                covid_qus['Coughing']=text
                l.append(float(text))
            elif covid_count==9:
                res=lang+'Enter Temperature value'
                covid_count=covid_count+1
                covid_qus['SoreThroat']=text
                l.append(float(text))
            elif covid_count==10:
                covid_count=covid_count+1
                covid_qus['Temperature']=text
                l.append(float(text))
                print('final covid_qus',covid_qus)
                print('final l=',l)
                covid_pred=prediction(l)    #[float(covid_qus['Age']), float(covid_qus['Sex']), float(covid_qus['neutrophil']), float(covid_qus['serumLevelsOfWhiteBloodCell']), float(covid_qus['lymphocytes']), float(covid_qus['Diarrhea']), float(covid_qus['Fever']), float(covid_qus['Coughing']),float( covid_qus['SoreThroat']),float( covid_qus['Temperature'])])
                print(covid_pred[0])
                res=covid_pred[0]

                print(covid_qus)
                print('conditions ',lcon)
                ind=[]
                for i in range(len(lcon)):
                    try:
                        if 'fever' in lcon[i].lower() or 'covid' in llcon[i].lower():
                            print('Fever or covid')
                            ind.append(i)
                    except:
                        pass
                drugs=[]
                for i in ind:
                    drug_name=df['drugName'][i]
                    review=df['review'][i]
                    rating=df['rating'][i]
                    score = analyzer.polarity_scores(review)
                    p=float(score['pos'])
                    ne=float(score['neu'])
                    n=float(score['neg'])
                    performance = [p,n,ne]
                    index=performance.index(max(performance))
                    if (index==0):
                        result='positive'
                    elif (index==1):
                        result='negative'
                    elif (index==2):
                        result='neutral'
                    print('result',result)
                    if result=='positive' or result=='neutral':
                        if int(rating)>5:
                            drugs.append(drug_name)
                djs=''
                for j in drugs:
                    djs=djs+' '+j
                if my_lang=='en':
                    res='eDetected Disease is '+res+'. Recommended Medicine are '+djs
                else:
                    res='kDetected Disease is '+res+'. Recommended Medicine are '+djs
                print(res)
                print('l=',l)

        return res
       
           
        



                    
#To switch on a particular URL, we use the add resource method and route it to the default slash. 


api.add_resource(process, '/process')

if __name__ == '__main__':
    app.run()  # run our Flask app
