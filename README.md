# Udacity Project
Create a pipeline to classify disaster response messages 

## Installations
This project was written in the Anaconda environment and Python 3.7

## Motivation
The purpose of this project was to apply data engineering methods and analyze text messages sent during disasters to build a model for an API that classifies disaster messages.

## Files
This repository contains the following:
- data
    - disaster_categories.csv   -> data to process 
    - disaster_messages.csv   -> data to process
    - process_data.py
    - DisasterResponse.db   -> database to save clean data to

- models
    - train_classifier.py
    - classifier.pkl   -> saved model 
    
- app
    - template  -> contains html files for the web app
    - run.py  -> Flask file that runs the web app


## Summary
- An ETL pipeline was built to load and clean message data before storing them in a SQLite database.
- A machine learning pipeline was built to train a model to perform multi output classification against 36 different message categories.
- A Flask web app was used to classify user input messages, along with Plotly to display several data visualisations.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
With thanks to:

Udacity - for setting up the project and course contents.

Figure Eight - for providing the data. https://www.figure-eight.com/
