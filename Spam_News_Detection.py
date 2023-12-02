import numpy as np
import pandas as pd
True_news = pd.read_csv("True.csv")
Fake_news = pd.read_csv("Fake.csv")
True_news["label"]=0
Fake_news["label"]=1
dataset1= True_new[["text","label"]]
dataset2= Fake_news[["text","label"]]
dataset=pd.concat([dataset1,dataset2])
dataset.shape
#check for null values
dataset.isnull().sum()
dataset["label"].value_counts()
dataset=dataset.sample(frac=1)

#NLP
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import wordNetlemmatizer

ps=WordNetLemmantizer()
stopwords=stopwords.words("english")
nlkt.download("wordnet")

def clean_row(row):
    row=row.lower()
    row=re.sub("[a-zA-z]"," ",row)
    token=row.split()
    news=[ps.lemmatize(word) for word in token if not word in stopwords]
    cleanned_news= " ".join(news)
    return cleanned_news
    

dataset["text"]=dataset["text"].apply(lambda x : clean_row(x))

from sklearn.feature_extraction.text import IfidfVectorizer
vectorizer=IfidfVectorizer(max_features 50000, lowercase=False, ngram_range(1,2))
x=dataset.iloc[:35000,0]
y=dataset.iloc[:35000,1]

x.head()

from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label= train_test_split(x,y,test_size=0.2,random_state 0)
vec_train_data=vectorizer.fit_transform(train_data)
vec_train_data=vec_train_data.toarray()
vec_test_data=vectorizer.fit_transform(test_data)
vec_train_data=vec_train_data.toarray()
vec_train_data.shape,vec_train_data.shape

train_data=pd.Dataframe(vec_train,columns=vectorizer.get_feature_name())
testing_data=pd.ataframe(vec_test_data,columns=vectorizer.get_feature_name())

#Model
from sklearn.naive_bayes import MultinomilaNB
clf+multinomilaNB()
clf.fit(train_data,train_label)
y_pred=clf.predict(testing_data)

from sklearn.metrics import accuracy_score
y_pred_train=clf.predict(training_data)

#add any text from true excel file and retrive its other information
txr="the following statement were posted"
news=clean_row(txt)
pred=clf.predict(vectorizer.transform([news]).toarray())
txt=input("Enter News: ")
news=clean_row(str(txt))
if pred==0:
    print("News is correct")
else:
    print("News is Fake")
    
