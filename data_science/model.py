import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from collections import Counter
from sklearn.metrics import mean_squared_error, make_scorer

def load_train(file):
    global names
    global nation
    global category
    global currency
    global location
    global mean_lifespan
    global depth_mean
    global depth_std
    global height_mean
    global height_std
    global width_mean
    global width_std
    global mean
    global std
    global mean1
    global std1
    global train_loaded
    global enc
    global lasso
    data = pd.read_csv(open(file,encoding='latin-1'))
    data = data[~np.isnan(data['hammer_price'])]
    data = data[data['hammer_price']>0]

    # average lifespan of artists
    temp = data[~np.isnan(data['artist_death_year'])]
    mean_lifespan = np.mean(temp['artist_death_year']-temp['artist_birth_year'])

    # adding average span to unknown values
    data['artist_death_year'] = data.apply(lambda x: x['artist_birth_year']+mean_lifespan if np.isnan(x['artist_death_year']) else x['artist_death_year'],axis=1)
    
    # auction date to epoch
    data['auction_date'] = data['auction_date'].map(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").timetuple()))
    
    # standard scaling with mean 1950 and std 50
    epoch_1950 = time.mktime(datetime.datetime.strptime("1950-01-01T00:00:00.000Z", "%Y-%m-%dT%H:%M:%S.000Z").timetuple())
    epoch_50 = time.mktime(datetime.datetime.strptime("0050-01-01T00:00:00.000Z", "%Y-%m-%dT%H:%M:%S.000Z").timetuple())
    data['artist_death_year'] = data['artist_death_year'].map(lambda x: (x-1950)/50.0)

    data['artist_birth_year'] = data['artist_birth_year'].map(lambda x: (x-1950)/50.0)

    data['auction_date'] = data['auction_date'].map(lambda x: (x-epoch_1950)/epoch_50)

    # standard scaling the measurements
    depth_mean = np.mean(data['measurement_depth_cm'])
    depth_std = np.std(data['measurement_depth_cm'])
    data['measurement_depth_cm'] = data['measurement_depth_cm'].map(lambda x: (x-depth_mean)/depth_std)

    height_mean = np.mean(data['measurement_height_cm'])
    height_std = np.std(data['measurement_height_cm'])
    data['measurement_height_cm'] = data['measurement_height_cm'].map(lambda x: (x-height_mean)/height_std)

    width_mean = np.mean(data['measurement_width_cm'])
    width_std = np.std(data['measurement_width_cm'])
    data['measurement_width_cm'] = data['measurement_width_cm'].map(lambda x: (x-width_mean)/width_std)
    data = data.drop(['edition', 'title', 'year_of_execution', 'materials'], axis=1)
    
    # standard scaling the known estimates and setting the unknown to zero
    temp = data[~np.isnan(data['estimate_high'])]
    mean = np.mean(temp['estimate_high'])
    std = np.std(temp['estimate_high'])
    data['estimate_high'] = data['estimate_high'].map(lambda x: (x-mean)/std if ~np.isnan(x) else x)

    temp = data[~np.isnan(data['estimate_low'])]
    mean1 = np.mean(temp['estimate_low'])
    std1 = np.std(temp['estimate_low'])
    data['estimate_low'] = data['estimate_low'].map(lambda x: (x-mean1)/std1 if ~np.isnan(x) else x)

    data['estimate_high'] = data['estimate_high'].fillna(0)
    data['estimate_low'] = data['estimate_low'].fillna(0)
    data['location'] = data['location'].fillna("")
    return data

def load_test_data(file,rows):
    global names
    global nation
    global category
    global currency
    global location
    global mean_lifespan
    global depth_mean
    global depth_std
    global height_mean
    global height_std
    global width_mean
    global width_std
    global mean
    global std
    global mean1
    global std1
    global train_loaded
    global enc
    global lasso
    data = pd.read_csv(open(file,encoding='latin-1'))
    data = data[~np.isnan(data['hammer_price'])]
    data = data[data['hammer_price']>0]

    data = data.sample(rows)

    data['artist_death_year'] = data.apply(lambda x: x['artist_birth_year']+mean_lifespan if np.isnan(x['artist_death_year']) else x['artist_death_year'],axis=1)
    data['auction_date'] = data['auction_date'].map(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").timetuple()))
    epoch_1950 = time.mktime(datetime.datetime.strptime("1950-01-01T00:00:00.000Z", "%Y-%m-%dT%H:%M:%S.000Z").timetuple())
    epoch_50 = time.mktime(datetime.datetime.strptime("0050-01-01T00:00:00.000Z", "%Y-%m-%dT%H:%M:%S.000Z").timetuple())
    data['artist_death_year'] = data['artist_death_year'].map(lambda x: (x-1950)/50.0)

    data['artist_birth_year'] = data['artist_birth_year'].map(lambda x: (x-1950)/50.0)

    data['auction_date'] = data['auction_date'].map(lambda x: (x-epoch_1950)/epoch_50)

    data['measurement_depth_cm'] = data['measurement_depth_cm'].map(lambda x: (x-depth_mean)/depth_std)

    height_std = np.std(data['measurement_height_cm'])
    data['measurement_height_cm'] = data['measurement_height_cm'].map(lambda x: (x-height_mean)/height_std)

    data['measurement_width_cm'] = data['measurement_width_cm'].map(lambda x: (x-width_mean)/width_std)
    data = data.drop(['edition', 'title', 'year_of_execution', 'materials'], axis=1)
    
    data['estimate_high'] = data['estimate_high'].map(lambda x: (x-mean)/std if ~np.isnan(x) else x)

    data['estimate_low'] = data['estimate_low'].map(lambda x: (x-mean1)/std1 if ~np.isnan(x) else x)

    data['estimate_high'] = data['estimate_high'].fillna(0)
    data['estimate_low'] = data['estimate_low'].fillna(0)
    data['location'] = data['location'].fillna("")
    return data

def getCateg(data,key,threshold):
    temp = Counter(data[key])
    result = {}
    ind = 0
    for l in temp:
        if l and temp[l]>threshold:
            result[l] = ind
            ind+=1
    return result

names = {}
nation = {}
category = {}
currency = {}
location = {}
mean_lifespan = 0
depth_mean = 0
depth_std = 1
height_mean = 0
height_std = 1
width_mean = 0
width_std = 1
mean = 0
std = 1
mean1 = 0
std1 = 1
train_loaded = False
enc = OneHotEncoder(handle_unknown='ignore',categorical_features=[2,3,5,6,10],sparse=False)
lasso = Lasso(fit_intercept=True, max_iter=10000, tol= 0.1,random_state=42)

def generate_model(data):
    global names
    global nation
    global category
    global currency
    global location
    global mean_lifespan
    global depth_mean
    global depth_std
    global height_mean
    global height_std
    global width_mean
    global width_std
    global mean
    global std
    global mean1
    global std1
    global train_loaded
    global enc
    global lasso
    names = getCateg(data,"artist_name",0)
    nation = getCateg(data,"artist_nationality",0)
    category = getCateg(data,"category",0)
    currency = getCateg(data,"currency",0)
    location = getCateg(data,"location",10)

    data['artist_name'] = data['artist_name'].apply(lambda x: names[x] if x in names else len(names))
    data['artist_nationality'] = data['artist_nationality'].apply(lambda x: nation[x] if x in nation else len(nation))
    data['category'] = data['category'].apply(lambda x: category[x] if x in category else len(category))
    data['currency'] = data['currency'].apply(lambda x: currency[x] if x in currency else len(currency))
    data['location'] = data['location'].apply(lambda x: location[x] if x in location else len(location))

    enc.fit(data)
    numpy_data = enc.transform(data)

    y = numpy_data[:,-4]
    X = np.delete(numpy_data,-4,1)

    lasso.fit(X,y)

def generate_test_data(data):
    global names
    global nation
    global category
    global currency
    global location
    global mean_lifespan
    global depth_mean
    global depth_std
    global height_mean
    global height_std
    global width_mean
    global width_std
    global mean
    global std
    global mean1
    global std1
    global train_loaded
    global enc
    global lasso
    data['artist_name'] = data['artist_name'].apply(lambda x: names[x] if x in names else len(names))
    data['artist_nationality'] = data['artist_nationality'].apply(lambda x: nation[x] if x in nation else len(nation))
    data['category'] = data['category'].apply(lambda x: category[x] if x in category else len(category))
    data['currency'] = data['currency'].apply(lambda x: currency[x] if x in currency else len(currency))
    data['location'] = data['location'].apply(lambda x: location[x] if x in location else len(location))
    numpy_data = enc.transform(data)

    y = numpy_data[:,-4]
    X = np.delete(numpy_data,-4,1)
    return X,y

def predict(file,rows):
    global names
    global nation
    global category
    global currency
    global location
    global mean_lifespan
    global depth_mean
    global depth_std
    global height_mean
    global height_std
    global width_mean
    global width_std
    global mean
    global std
    global mean1
    global std1
    global train_loaded
    global enc
    global lasso
    if train_loaded==False:
        data = load_train('data.csv')
        generate_model(data)
        train_loaded = True
    test_data = load_test_data(file,rows)
    X,y = generate_test_data(test_data)
    y_pred = lasso.predict(X)
    return mean_squared_error(y_pred,y)**0.5