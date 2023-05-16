import pandas as pd
import json
import math
from category_encoders import OrdinalEncoder
from flask import Flask, request, jsonify
from flask_accept import accept
from surprise import dump
from surprise import Reader, Dataset
import os
import pickle
import random

# Load DataFrame from JSON
def json_to_df(json_data):
    data = json.loads(json.dumps(json_data))
    df = pd.json_normalize(data)
    return df

def load_model(model_filename):
    file_name = os.path.expanduser(model_filename)
    loaded_model = dump.load(file_name)

    return loaded_model

def lang(df):
  if df['English'] >= df['Kiswahili']:
    return df['English']
  else:
    return df['Kiswahili']
    

def science(df):
  if df['Mathematics'] >= df['Physics']:
    return df['Mathematics']
  else:
    return df['Physics']
  
def cluster_points(df):
  mult = (df['points']/48)*(df['Overall_Grade']/84)
  cluster = 48 * math.sqrt(mult)
  return cluster

def generate_user_id(df):
    num_users = len(df) + 1
    return f'user_{num_users}'

def get_course_recommendations1(user_id, df):
    KNNBaseline_pickle_model = load_model('model/KNNBaseline_pickled_model')
    # get the inner user id
    inner_user_id = df.to_inner_uid(user_id)

    # get the courses the user has already rated
    rated_courses = set([r[0] for r in df.ur[inner_user_id]])

    # get all courses
    course_ids = [iid for iid in df.all_items()]

    # Get the list of courses the user is interested in
    user_interests = df.loc[df['ID'] == user_id, 'Interest'].unique()
    print(f'User Interest: {user_interests} ')

    # # Create a list of tuples of (course, predicted rating) for each course
    course_ratings = []
    for course in df.loc[df['Interest'].isin(user_interests), 'Course'].unique():
      if course not in rated_courses:
        predicted_rating = KNNBaseline_pickle_model.predict(uid=user_id, iid=course).est
        course_ratings.append((course, predicted_rating))

    # Sort the list of course ratings by predicted rating
    course_ratings_sorted = sorted(course_ratings, key=lambda x: x[1], reverse=True)
    # Create a list of the top three recommended courses that match the user's interests
    recommended_courses = []
    for course_rating in course_ratings_sorted:
      if course_rating[0] not in user_interests:
        recommended_courses.append(course_rating[0])
        if len(recommended_courses) == 3:
            break
    return recommended_courses


def get_three_courses(courses):
    if len(courses) < 3:
          raise random.shuffle(courses)
    return random.sample(courses, 3)

def get_course_recommendations(user_id, df, courses_catalogue):
    # Load the trained KNNBaseline model from a file using pickle
    with open('model/KNNBaseline_pickled_model', 'rb') as file:
        knn_model = pickle.load(file)
    
    # Define a Reader object to parse the dataframe
    reader = Reader(rating_scale=(1, 5))

    # Load the dataframe into a Surprise Dataset object
    # data = Dataset.load_from_df(df[['user_id', 'Interest', 'Job_satisfaction']], reader)

    # Build a testset from the Dataset object
    # testset = data.build_full_trainset()

    # Retain only courses within interest from the user
    interest = df['Interest'].unique()[0]
    filtered_catalogue = courses_catalogue[courses_catalogue['Interest'] == interest]
    # filtered_catalogue = courses_catalogue[courses_catalogue['Weighted_points'] >= cluster_points(df)]

    # Get the list of course names from the filtered course catalogue
    course_names = filtered_catalogue['Course_name'].tolist()

    # Get the list of courses that the user is interested in
    anti_testset = []
    for course_name in course_names:
        if course_name not in course_names:
            anti_testset.append((user_id, interest, 0)) # set rating to 0 (since the user has not taken the course)

    # Get the course details for the recommended courses, sorted by predicted job satisfaction
    recommended_courses = filtered_catalogue.loc[filtered_catalogue['Course_name'].isin(course_names)][['Course_name', 'Job_satisfaction']].sort_values('Job_satisfaction', ascending=False)

    courses = recommended_courses['Course_name'].tolist()

    return get_three_courses(courses)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def say_hello():
   return jsonify({'greeting': 'Hello from ClassifyMe!'})

@app.route('/recommendations', methods=['GET', 'POST'])
@accept('*/*')
def predict():
    # Get user input as JSON
    print(request)
    input_data = request.get_json()

    # Add user ID
    input_data['user_id'] = 'user_120'

    # Add Job satistaction
    input_data['Overall_Grade'] = 76

    input_data['points'] = 22

    # Get DataFrame from JSON input
    df = json_to_df(input_data)
    df['Interest'].unique()

    encoding = {'Public Health': 1,  'Laboratories': 2,
                                            'Nursing': 3, 'Medical Research': 4,
                                            'Therapy': 5, 'Pharmacy': 6, 'Surgery': 7}

    mapping = [{'col': 'Interest', 'mapping': encoding}]
    # Create an OrdinalEncoder object and fit it to the DataFrame
    encoder = OrdinalEncoder(cols=['Interest'], mapping=mapping)
    encoder.fit(df)
    df = encoder.transform(df)

    # Load dataframe
    courses_catalogue = pd.read_csv('datasets/courses_catalogue.csv')

    predictions = get_course_recommendations('user_120', df, courses_catalogue)

    # print(predictions)
    return jsonify({'recommended_courses': predictions})

# Define middleware function
@app.before_request
def middleware():
    # Log the request method and path
    new_headers = dict(request.headers)
    new_headers['Accept'] = 'application/json'
    request.headers = new_headers

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

if __name__ == "__main__":
    app.run()