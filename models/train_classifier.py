import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt','stopwords','wordnet'])
import re

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
import pickle

def load_data(database_filepath):
    '''
    Function to load database file 

    Parameters:
    database_filepath: location of database file

    Returns:
    X: message portion of data
    Y: target variable portion of data
    category_names: target variable category names

    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Messages', engine)
    
    X = df['message']
    Y = df.iloc[:,4:] 
    
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Function to process and clean text data

    Parameters:
    text: text string data

    Returns:
    tokens: cleaned text tokens

    '''
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    Function to build model pipeline

    Parameters:
    n/a

    Returns:
    cv: gridsearchcv model

    '''
    
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {'clf__estimator__n_estimators': [100],
                  'clf__estimator__learning_rate': [0.1, 0.5, 1]
                 }
    
    # 2-fold cv
    cv = GridSearchCV(pipeline, cv=2, param_grid=parameters, n_jobs=-1, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate model

    Parameters:
    model: model to evaluate
    X_test: X_test dataset
    Y_test: Y_test dataset
    category_names: target variable category names

    Returns:
    n/a

    '''
    
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Function to save model in pickle file

    Parameters:
    model: model to save
    model_filepath: save location

    Returns:
    n/a

    '''
    
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)

        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()