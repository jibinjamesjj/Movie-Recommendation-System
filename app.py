#loading the packages
from flask import Flask, render_template, redirect, url_for, request, session, flash, Response, jsonify
from functools import wraps
import json
from flask_pymongo import PyMongo
import pprint
import pandas as pd
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings("ignore")

#content based filtering code
data = pd.read_csv('movies_database.csv')
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a', 'an'
tfidf = TfidfVectorizer(stop_words='english')
#Replace NaN with an empty string
data['overview'] = data['overview'].fillna('')
#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data['overview'])
#Convert TFIDF matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = tfidf_matrix.todense()
df = pd.DataFrame(doc_term_matrix, columns=tfidf.get_feature_names(), index=data.overview)
# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#Let's create a dataframe of the similarity matrix with rows and columns as movie titles
sim = pd.DataFrame(cosine_sim, columns=data.original_title, index=data.original_title)
# Create a column of movie titles
indices = pd.Series(data.index, index=data['original_title']).drop_duplicates()
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies in descending order of similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies ignoring the first one as it is itself movie
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return data['original_title'].iloc[movie_indices]

#creating flask object

app = Flask(__name__)

# creating a temporary secret key for sessions
app.secret_key = "peace"

#initializing the mongodb client
#ties our appllication to mongodb instance
mongodb_client = PyMongo(app, uri="mongodb://localhost:27017/Netflax")
Netflax = mongodb_client.db
users = Netflax.users
movies = Netflax.movies
comments = Netflax.comments
posters = Netflax.posters

# login required decorator
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'email' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap



@app.route('/')
@app.route('/index')
@login_required
def index():
    original_title = movies.find_one({"original_title" : "Avatar"},{"_id":0,"original_title":1})
    overview = movies.find_one({"original_title" : "Avatar"},{"_id":0,"overview":1})
    latest_releases = movies.find({"release_date": {"$gt": "2015-09-27"}},{"_id":0,"original_title":1}).limit(10)
    popular_movies = movies.find({"popularity": {"$gt": "150"}})
    action_movies = movies.find({"genre1": "Action"},{"_id":0}).limit(10)
    comedy_movies = movies.find({"genre1": "Comedy"},{"_id":0}).limit(10)
    title_list = movies.aggregate([{ "$sample": { "size": 10 } }])
    # demographic filtering code
    data = pd.read_csv("movies_database.csv")
    C= data['vote_average'].mean()
    m= data['vote_count'].quantile(0.7)
    q_movies = data.copy().loc[data['vote_count'] >= m]
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)
    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    #Sort movies based on score calculated above
    q_movies = q_movies.sort_values('score', ascending=False)
    #Print the top 10 movies
    top_movies = q_movies[['original_title']].head(10)
    top_movies_list = top_movies.values.tolist()
    return render_template('index.html', overview=overview, original_title=original_title, latest_releases=latest_releases, popular_movies=popular_movies, action_movies=action_movies, comedy_movies=comedy_movies, top_movies_list=top_movies_list, title_list=title_list)



@app.route('/details/<movie_name>', methods=['GET'])
@login_required
def details(movie_name):
    #movie_name = "The Dark Knight"
    original_title = movies.find_one({"original_title" : movie_name},{"_id":1,"original_title":1})
    release_date = movies.find_one({"original_title" : movie_name},{"_id":0,"release_date":1})
    runtime = movies.find_one({"original_title" : movie_name},{"_id":0,"runtime":1})
    genre1 = movies.find_one({"original_title" : movie_name},{"_id":0,"genre1":1})
    genre2 = movies.find_one({"original_title" : movie_name},{"_id":0,"genre2":1})
    genre3 = movies.find_one({"original_title" : movie_name},{"_id":0,"genre3":1})
    overview = movies.find_one({"original_title" : movie_name},{"_id":0,"overview":1})
    cast1 = movies.find_one({"original_title" : movie_name},{"_id":0,"cast1":1})
    cast2 = movies.find_one({"original_title" : movie_name},{"_id":0,"cast2":1})
    cast3 = movies.find_one({"original_title" : movie_name},{"_id":0,"cast3":1})
    comments_list = comments.find({"movie_name": movie_name},{"_id":0,"comment":1,"date":1,"username":1}).limit(10)
    poster = posters.find_one({"original_title": movie_name},{"_id":0,"poster":1})
    return render_template('details.html', original_title=original_title, release_date=release_date, runtime=runtime, genre1=genre1, genre2=genre2, genre3=genre3, overview=overview, cast1=cast1, cast2=cast2, cast3=cast3, comments_list=comments_list, poster=poster)



@app.route('/addcomment/<movie_name>', methods=['POST','GET'])
@login_required
def addcomment(movie_name):
    if request.method == 'POST':
        _comment = request.form["comment"]
        _today = str(date.today())
        _username = session['username']
        result = comments.insert_one({"movie_name": movie_name, "comment":_comment, "date":_today, "username":_username})
    original_title = movies.find_one({"original_title" : movie_name},{"_id":1,"original_title":1})
    release_date = movies.find_one({"original_title" : movie_name},{"_id":0,"release_date":1})
    runtime = movies.find_one({"original_title" : movie_name},{"_id":0,"runtime":1})
    genre1 = movies.find_one({"original_title" : movie_name},{"_id":0,"genre1":1})
    genre2 = movies.find_one({"original_title" : movie_name},{"_id":0,"genre2":1})
    genre3 = movies.find_one({"original_title" : movie_name},{"_id":0,"genre3":1})
    overview = movies.find_one({"original_title" : movie_name},{"_id":0,"overview":1})
    cast1 = movies.find_one({"original_title" : movie_name},{"_id":0,"cast1":1})
    cast2 = movies.find_one({"original_title" : movie_name},{"_id":0,"cast2":1})
    cast3 = movies.find_one({"original_title" : movie_name},{"_id":0,"cast3":1})
    comments_list = comments.find({"movie_name" : movie_name},{"_id":0,"comment":1,"date":1,"username":1}).limit(10)
    poster = posters.find_one({"original_title": movie_name},{"_id":0,"poster":1})
    return render_template('details.html', original_title=original_title, release_date=release_date, runtime=runtime, genre1=genre1, genre2=genre2, genre3=genre3, overview=overview, cast1=cast1, cast2=cast2, cast3=cast3, comments_list=comments_list, comment=_comment, poster=poster)



@app.route('/contentbasedf', methods=['POST','GET'])
@login_required
def contentbasedf():
    if request.method == 'POST':
        selected_movie_for_cbf = request.form.get("selected_movie_for_cbf")
        recommended_cbf = get_recommendations(selected_movie_for_cbf)
    return render_template('recommended_movies.html', recommended_cbf=recommended_cbf)



@app.route('/admin', methods=['POST','GET'])
def admin():
    no_of_users = users.count()
    no_of_comments = comments.count()
    no_of_movies = movies.count()
    return render_template('admin.html', no_of_users=no_of_users, no_of_comments=no_of_comments, no_of_movies=no_of_movies)


#route for storing df into database
@app.route('/retrieve')
def retrieve():
    #recommended_cbf = get_recommendations('Avatar')
    return render_template('testing.html', recommended_cbf=recommended_cbf)


@app.route('/login')
def login():
	return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    session.pop('email', None)
    flash('You are Logged Out!')
    return redirect(url_for('login'))


#login
@app.route("/signin", methods = ['GET','POST'])
def signin():
    email = request.form.get("email")
    password = request.form.get("password")
    if email == "admin" and password == "admin":
        return redirect(url_for('admin'))
    message = 'Please login to your account'
    if  "email" in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")
        email_found = users.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']

            if passwordcheck == password:
                session['email'] = True
                username = users.find_one({"email" : email_val},{"_id":0,"username":1})
                session['username'] = username['username']
                return redirect(url_for('index'))
            else:
                if "email" in session:
                    return render_template('login.html')
                message = 'Wrong Password'
                return render_template('login.html', message=message)
        else:
            message = 'Email can\'t be found'
            return render_template('login.html', message=message)
    return render_template('login.html', message=message)


#register
@app.route("/signup", methods = ['GET','POST'])
def register():
    message = ''
    #if "email" in session:
    #    return redirect(url_for("index"))
    if request.method == 'POST':
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm-password")

        user_found = users.find_one({"username": username})
        email_found = users.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name, try logging in!'
            return render_template('login.html', message=message)
        elif email_found:
            message = 'This email already exists in database, try logging in!'
            return render_template('login.html', message=message)
        else:
            user_input = {'username':username, 'email':email, 'password':password}
            users.insert_one(user_input)
            message = 'Account registered successfully'
    return render_template('login.html', message=message)

#insert poster
@app.route("/insertposter")
def insertposter():
    #posters.update_many({}, {"$set":{"poster": "../static/img/posters/Spectre.jpg"}})
    return "Success"

if __name__ == '__main__': 
	app.run(debug=True)