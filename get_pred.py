import pandas as pd
import json
from category_encoders import OrdinalEncoder
from surprise import Reader, Dataset
import pickle


# Make predictions

def json_to_df(json_data):
    data = json.loads(json.dumps(json_data))
    df = pd.json_normalize(data)
    return df

def get_course_recommendations(user_id, df, courses_catalogue : pd.DataFrame):
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
    # inner_user_id = testset.to_inner_uid(user_id)

    # Get the list of courses in the dataset
    # Filter the course catalogue by interest

    # Compare interests za user
    # Retain only courses within interest from the user
    interest = df['Interest'].unique()[0]
    filtered_catalogue = courses_catalogue[courses_catalogue['Interest'] == interest]

    # Get the list of course names from the filtered course catalogue
    course_names = filtered_catalogue['Course_name'].tolist()

    # Make a prediction of job satisfaction using the KNN model
    # build the anti-testset for the user (i.e. all courses that the user has not taken)

    # Get the list of courses that the user is interested in
    anti_testset = []
    for course_name in course_names:
        if course_name not in course_names:
            anti_testset.append((user_id, interest, 0)) # set rating to 0 (since the user has not taken the course)

    # get the predicted ratings for the anti-testset using the trained model
    predictions = knn_model.predict(testset)
    # predicted_satisfaction = knn_model.predict(testset)
    # Get the course details for the recommended courses, sorted by predicted job satisfaction
    recommended_courses = filtered_catalogue.loc[filtered_catalogue['Course_name'].isin(course_names)][['Course_name', 'Job_satisfaction']].sort_values('Job_satisfaction', ascending=False)

    return predicted_satisfaction, recommended_courses.head(3)
    # Get the list of courses the user is interested in from the dataset
    # user_interests = df.loc[df['user_id'] == user_id, 'Interest'].unique()

    # # Filter the list of courses to include only those that the user has not rated before
    # rated_courses = set([r[0] for r in testset.ur[inner_user_id]])
    # print('Rated Courses')
    # print(rated_courses)
    # unrated_courses = [course_id for course_id in course_ids if course_id not in rated_courses]
    # print('Unrated Courses')
    # print(unrated_courses)

    # unrated_courses = [0, 1, 2]

    # # Use the test() method of the KNNBaseline model to predict the job satisfaction rating for each of the filtered courses for the given user
    # predictions = []
    # for course_id in unrated_courses:
    #     prediction = knn_model.predict(uid=user_id, iid=course_id)
    #     predictions.append(prediction)

    # # Create a list of tuples of (course, predicted rating) for each course
    # course_ratings = [(prediction.iid, prediction.est) for prediction in predictions]

    # # Sort the list of course ratings by predicted rating in descending order
    # course_ratings_sorted = sorted(course_ratings, key=lambda x: x[1], reverse=True)

    # # Create a list of recommended courses by selecting the top three courses from the sorted list that match the user's interests
    # recommended_courses = []
    # for course_rating in course_ratings_sorted:
    #     if course_rating[0] in user_interests:
    #         recommended_courses.append(course_rating[0])
    #         if len(recommended_courses) == 3:
    #             break

    # return recommended_courses

if __name__ == "__main__":
    input_data = {
        "English" : 12,
        "Kiswahili" : 12,
        "Mathematics" : 12,
        "Physics" : 12,
        "Biology" : 12,
        "Chemistry" : 12,
        "Interest" : "Therapy"
    }

    # Add user ID
    input_data['user_id'] = 'user_120'

    # Add Job satistaction
    input_data['Job_satisfaction'] = 0.00

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

    predicted_satisfaction, recommended_courses = get_course_recommendations('user_120', df, courses_catalogue)

    print(recommended_courses)