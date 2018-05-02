from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from app.DataPrep import *
from app.FeatureSelection import *
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import urllib.request
import urllib.parse
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import pickle
 
import csv
# Create your views here.
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
#url='http://www.firstpost.com/world/facebook-data-scandal-uk-regulators-search-cambridge-analytica-offices-ruling-on-case-tuesday-4404233.html'
def index(request):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    template='app/index.html'
    if request.method=='POST':
        url=request.POST['url']
        
        #taking summary of the news from url
        values={'s':'basics','submit':'search'}
        data=urllib.parse.urlencode(values)
        data=data.encode('utf-8')
        req=urllib.request.Request(url,data)
        resp=urllib.request.urlopen(req)
        resp_data=resp.read()
        #print(resp_data)

        paragraphs=re.findall(r'<p>(.*?)</p>',str(resp_data))
        for p in paragraphs:
            clean=cleanhtml(p)
            #clean=remove_html_markup(p) 
            print(clean)
        print("--------------SUMMARY----------------")

        LANGUAGE = "english"
        SENTENCES_COUNT = 15
        parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        summary=""
        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            summary+=str(sentence)
        
        # pasing summary to model for prediction 
        with open('app/static/sav/final_model_6.sav', 'rb') as filename:
            load_model = pickle.load(filename)
        prediction = load_model.predict([summary])
        prob = load_model.predict_proba([summary])
        result=prediction[0]
        score=prob[0][1]

        context={
            'url_text':"Entered URL:",
            'url':url,
            'static_text0':"Summary of the news from URL",
            'static_text':"Result:",
            'static_text2':"The given summary is: ",
            #'static_text3':"The truth probability score is: ",
            'result':result,
            'summary':summary,
            #'score':score
        }
        return render(request,template,context)
    else:
        return render(request,template)

    def classifier():
        
        #building classifier using naive bayes 
        #nb_pipeline = Pipeline([
        #        ('NBCV',FeatureSelection.countV),
        #        ('nb_clf',MultinomialNB())])
        #
        #nb_pipeline.fit(DataPrep.test_news['Statement'],DataPrep.test_news['Label'])
        #predicted_nb = nb_pipeline.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_nb == DataPrep.test_news['Label'])
        #
        #
        ##building classifier using logistic regression
        #logR_pipeline = Pipeline([
        #        ('LogRCV',FeatureSelection.countV),
        #        ('LogR_clf',LogisticRegression())
        #        ])
        #
        #logR_pipeline.fit(DataPrep.test_news['Statement'],DataPrep.test_news['Label'])
        #predicted_LogR = logR_pipeline.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_LogR == DataPrep.test_news['Label'])
        #
        #
        ##building Linear SVM classfier
        #svm_pipeline = Pipeline([
        #        ('svmCV',FeatureSelection.countV),
        #        ('svm_clf',svm.LinearSVC())
        #        ])
        #
        #svm_pipeline.fit(DataPrep.test_news['Statement'],DataPrep.test_news['Label'])
        #predicted_svm = svm_pipeline.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_svm == DataPrep.test_news['Label'])
        #
        #
        ##using SVM Stochastic Gradient Descent on hinge loss
        #sgd_pipeline = Pipeline([
        #        ('svm2CV',FeatureSelection.countV),
        #        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
        #        ])
        #
        #sgd_pipeline.fit(DataPrep.test_news['Statement'],DataPrep.test_news['Label'])
        #predicted_sgd = sgd_pipeline.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_sgd == DataPrep.test_news['Label'])
        #
        #
        ##random forest
        #random_forest = Pipeline([
        #        ('rfCV',FeatureSelection.countV),
        #        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        #        ])
        #    
        #random_forest.fit(DataPrep.test_news['Statement'],DataPrep.test_news['Label'])
        #predicted_rf = random_forest.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_rf == DataPrep.test_news['Label'])


        #User defined functon for K-Fold cross validatoin
        def build_confusion_matrix(classifier):
            
            k_fold = KFold(n=len(DataPrep.test_news), n_folds=5)
            scores = []
            confusion = np.array([[0,0],[0,0]])

            for train_ind, test_ind in k_fold:
                train_text = DataPrep.train_news.iloc[train_ind]['Statement'] 
                train_y = DataPrep.train_news.iloc[train_ind]['Label']
            
                test_text = DataPrep.train_news.iloc[test_ind]['Statement']
                test_y = DataPrep.train_news.iloc[test_ind]['Label']
                
                classifier.fit(train_text,train_y)
                predictions = classifier.predict(test_text)
                
                confusion += confusion_matrix(test_y,predictions)
                score = f1_score(test_y,predictions)
                scores.append(score)
            
            return (print('Total statements classified:', len(DataPrep.test_news)),
            print('Score:', sum(scores)/len(scores)),
            print('score length', len(scores)),
            print('Confusion matrix:'),
            print(confusion))
            
        #K-fold cross validation for all classifiers
        #build_confusion_matrix(nb_pipeline)
        #build_confusion_matrix(logR_pipeline)
        #build_confusion_matrix(svm_pipeline)
        #build_confusion_matrix(sgd_pipeline)
        #build_confusion_matrix(random_forest)

        #========================================================================================
        #Bag of words confusion matrix and F1 scores

        #Naive bayes
        # [2118 2370]
        # [1664 4088]
        # f1-Score: 0.669611539651

        #Logistic regression
        # [2252 2236]
        # [1933 3819]
        # f1-Score: 0.646909097798

        #svm
        # [2260 2228]
        # [2246 3506]
        #f1-score: 0.610468748792

        #sgdclassifier
        # [2414 2074]
        # [2042 3710]
        # f1-Score: 0.640874558778

        #random forest classifier
        # [1821 2667]
        # [1192 4560]
        # f1-Score: 0.702651511011
        #=========================================================================================


        """So far we have used bag of words technique to extract the features and passed those featuers into classifiers. We have also seen the
        f1 scores of these classifiers. now lets enhance these features using term frequency weights with various n-grams
        """

        #Now using n-grams
        #naive-bayes classifier
        # nb_pipeline_ngram = Pipeline([
        #         ('nb_tfidf',FeatureSelection.tfidf_ngram),
        #         ('nb_clf',MultinomialNB())])

        # nb_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
        # predicted_nb_ngram = nb_pipeline_ngram.predict(DataPrep.test_news['Statement'])
        # np.mean(predicted_nb_ngram == DataPrep.test_news['Label'])
        #
        #
        #
        ##logistic regression classifier
        logR_pipeline_ngram = Pipeline([
                ('LogR_tfidf',FeatureSelection.tfidf_ngram),
                ('LogR_clf',LogisticRegression(penalty="l2",C=1))
                ])
        logR_pipeline_ngram.fit(app.DataPrep.train_news['Statement'],app.DataPrep.train_news['Label'])
        predicted_LogR_ngram = logR_pipeline_ngram.predict(app.DataPrep.test_news['Statement'])
        np.mean(predicted_LogR_ngram == app.DataPrep.test_news['Label'])
        #build_confusion_matrix(logR_pipeline_ngram)
        print("reached end of logR n gram")
        #
        ##linear SVM classifier
        #svm_pipeline_ngram = Pipeline([
        #        ('svm_tfidf',FeatureSelection.tfidf_ngram),
        #        ('svm_clf',svm.LinearSVC())
        #        ])
        #svm_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
        #predicted_svm_ngram = svm_pipeline_ngram.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_svm_ngram == DataPrep.test_news['Label'])
        #
        #
        ##sgd classifier
        #sgd_pipeline_ngram = Pipeline([
        #         ('sgd_tfidf',FeatureSelection.tfidf_ngram),
        #         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
        #         ])
        #sgd_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
        #predicted_sgd_ngram = sgd_pipeline_ngram.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_sgd_ngram == DataPrep.test_news['Label'])

        #
        ##random forest classifier
        #random_forest_ngram = Pipeline([
        #       ('rf_tfidf',FeatureSelection.tfidf_ngram),
        #       ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        #       ])
        #random_forest_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
        #predicted_rf_ngram = random_forest_ngram.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_rf_ngram == DataPrep.test_news['Label'])
        #
        #
        ##K-fold cross validation for all classifiers
        # build_confusion_matrix(nb_pipeline_ngram)
        # build_confusion_matrix(logR_pipeline_ngram)
        #build_confusion_matrix(svm_pipeline_ngram)
        #build_confusion_matrix(sgd_pipeline_ngram)
        #build_confusion_matrix(random_forest_ngram)
        ##
        ##========================================================================================
        ##n-grams & tfidf confusion matrix and F1 scores

        ##Naive bayes
        ## [841 3647]
        ## [427 5325]
        ## f1-Score: 0.723262051071

        ##Logistic regression
        ## [1617 2871]
        ## [1097 4655]
        ## f1-Score: 0.70113000531

        ##svm
        ## [2016 2472]
        ## [1524 4228]
        ## f1-Score: 0.67909201429

        ##sgdclassifier
        ## [  10 4478]
        ## [  13 5739]
        ## f1-Score: 0.718731637053

        ##random forest
        ## [1979 2509]
        ## [1630 4122]
        ## f1-Score: 0.665720333284
        ##=========================================================================================

        #print(classification_report(DataPrep.test_news['Label'], predicted_nb_ngram))
        # print(classification_report(DataPrep.test_news['Label'], predicted_LogR_ngram))
        #print(classification_report(DataPrep.test_news['Label'], predicted_svm_ngram))
        #print(classification_report(DataPrep.test_news['Label'], predicted_sgd_ngram))
        #print(classification_report(DataPrep.test_news['Label'], predicted_rf_ngram))
        ##
        ##DataPrep.test_news['Label'].shape
        ##
        ##"""
        ##Out of all the models fitted, we would take 2 best performing model. we would call them candidate models
        ##from the confusion matrix, we can see that random forest and logistic regression are best performing 
        ##in terms of precision and recall (take a look into false positive and true negative counts which appeares
        ##to be low compared to rest of the models)
        ##"""
        ##
        ###grid-search parameter optimization
        ###random forest classifier parameters
        #parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
        #               'rf_tfidf__use_idf': (True, False),
        #               'rf_clf__max_depth': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
        #}
        #
        #gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1)
        #gs_clf = gs_clf.fit(DataPrep.train_news['Statement'][:10000],DataPrep.train_news['Label'][:10000])
        #
        #gs_clf.best_score_
        #gs_clf.best_params_
        #gs_clf.cv_results_
        print("Start of Log parameters")
        ###logistic regression parameters
        #parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
        #               'LogR_tfidf__use_idf': (True, False),
        #               'LogR_tfidf__smooth_idf': (True, False)
        #}
        #
        #gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1)
        #gs_clf = gs_clf.fit(DataPrep.test_news['Statement'][:10000],DataPrep.test_news['Label'][:10000])
        #
        #gs_clf.best_score_
        #gs_clf.best_params_
        #gs_clf.cv_results_
        #print("end of Log Parameters")
        #
        ###Linear SVM 
        #parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
        #               'svm_tfidf__use_idf': (True, False),
        #               'svm_tfidf__smooth_idf': (True, False),
        #               'svm_clf__penalty': ('l1','l2'),
        #}
        #
        #gs_clf = GridSearchCV(svm_pipeline_ngram, parameters, n_jobs=-1)
        #gs_clf = gs_clf.fit(DataPrep.test_news['Statement'][:10000],DataPrep.test_news['Label'][:10000])
        #
        #gs_clf.best_score_
        #gs_clf.best_params_
        #gs_clf.cv_results_

        ###by running above commands we can find the model with best performing parameters
        ##
        ##
        ###running both random forest and logistic regression models again with best parameter found with GridSearch method
        #random_forest_final = Pipeline([
        #        ('rf_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)),
        #        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3,max_depth=10))
        #        ])
        #    
        #random_forest_final.fit(DataPrep.test_news['Statement'],DataPrep.test_news['Label'])
        #predicted_rf_final = random_forest_final.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_rf_final == DataPrep.test_news['Label'])
        #build_confusion_matrix(random_forest_final)
        #print("final done")
        ##print(metrics.classification_report(DataPrep.test_news['Label'], predicted_rf_final))
        ##
        #sgd_pipeline_final = Pipeline([
        #         ('sgd_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)),
        #         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
        #         ])
        #sgd_pipeline_final.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
        #predicted_sgd_final = sgd_pipeline_ngram.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_sgd_final == DataPrep.test_news['Label'])
        #build_confusion_matrix(sgd_pipeline_final)
        #print("start of Log best performing parameters")
        #logR_pipeline_final = Pipeline([
        #        #('LogRCV',countV_ngram),
        #        ('LogR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
        #        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        #        ])
        #
        #logR_pipeline_final.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
        #predicted_LogR_final = logR_pipeline_final.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_LogR_final == DataPrep.test_news['Label'])
        #build_confusion_matrix(logR_pipeline_final)

        ##accuracy = 0.62
        #print(metrics.classification_report(DataPrep.test_news['Label'], predicted_LogR_final))

        #sgd_pipeline_final = Pipeline([
        #         ('sgd_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
        #         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
        #         ])
        #
        #sgd_pipeline_final.fit(DataPrep.test_news['Statement'],DataPrep.test_news['Label'])
        #predicted_sgd_final = sgd_pipeline_final.predict(DataPrep.test_news['Statement'])
        #np.mean(predicted_sgd_final == DataPrep.test_news['Label'])
        #build_confusion_matrix(sgd_pipeline_final)
        #print("End of best performing parameters")
        print("End of log_R")

        