import pandas as pd
import json
import math
from category_encoders import OrdinalEncoder
from flask import Flask, request, jsonify
from surprise import dump
from surprise import Reader, Dataset
import os
import pickle

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




def get_course_recommendations(user_id, df, courses_catalogue):
    # Load the trained KNNBaseline model from a file using pickle
    with open('model/KNNBaseline_pickled_model', 'rb') as file:
        knn_model = pickle.load(file)

    # Define a Reader object to parse the dataframe
    reader = Reader(rating_scale=(1, 5))

    # Load the dataframe into a Surprise Dataset object
    data = Dataset.load_from_df(df[['user_id', 'Interest', 'Job_satisfaction']], reader)

    # Build a testset from the Dataset object
    testset = data.build_full_trainset()

    # Use the testset as input to the test() method of the KNNBaseline model to generate predictions
    # predictions = knn_model.test(testset)

    # Get the inner user id corresponding to the given user id
    inner_user_id = testset.to_inner_uid(user_id)

    # Get the list of courses in the dataset
    course_ids = [iid for iid in testset.all_items()]

    # Get the list of courses the user is interested in from the dataset
    user_interests = df.loc[df['user_id'] == user_id, 'Interest'].unique()

    # Filter the list of courses to include only those that the user has not rated before
    rated_courses = set([r[0] for r in testset.ur[inner_user_id]])
    unrated_courses = [course_id for course_id in course_ids if course_id not in rated_courses]

    # Use the test() method of the KNNBaseline model to predict the job satisfaction rating for each of the filtered courses for the given user
    predictions = []
    for course_id in unrated_courses:
        prediction = knn_model.predict(uid=user_id, iid=course_id)
        predictions.append(prediction)

    # Create a list of tuples of (course, predicted rating) for each course
    course_ratings = [(prediction.iid, prediction.est) for prediction in predictions]

    # Sort the list of course ratings by predicted rating in descending order
    course_ratings_sorted = sorted(course_ratings, key=lambda x: x[1], reverse=True)

    # Create a list of recommended courses by selecting the top three courses from the sorted list that match the user's interests
    recommended_courses = []
    for course_rating in course_ratings_sorted:
        if course_rating[0] in user_interests:
            recommended_courses.append(course_rating[0])
            if len(recommended_courses) == 3:
                break

    return recommended_courses

app = Flask(__name__)

@app.route('/recommendations', methods=['POST'])
def predict():
    # Get user input as JSON
    input_data = request.get_json()

    # Add user ID
    input_data['user_id'] = 'user_120'

    # Add Job satistaction
    input_data['Job_satisfaction'] = 0.00

    # Get DataFrame from JSON input
    df = json_to_df(input_data)
    df['Interest'].unique()

    mapping = [{'col': 'Interest', 'mapping': {'Public Health': 1,  'Laboratories': 2,
                                            'Nursing': 3, 'Medical Research': 4,
                                            'Therapy': 5, 'Pharmacy': 5, 'Surgery': 6}}]
    # Create an OrdinalEncoder object and fit it to the DataFrame
    encoder = OrdinalEncoder(cols=['Interest'], mapping=mapping)
    encoder.fit(df)

    # Load dataframe
    courses_catalogue = pd.read_csv('datasets/courses_catalogue.csv')

    predictions = get_course_recommendations('user_120', df, courses_catalogue)

    # print(predictions)
    return jsonify({'recommended_courses': predictions})

if __name__ == "__main__":
    app.run()