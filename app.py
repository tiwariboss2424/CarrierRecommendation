from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Used for session management

# Load the scaler and model
scaler = pickle.load(open("Models/scaler.pkl", "rb"))
model = pickle.load(open("Models/model.pkl", "rb"))

# List of career class names (should match your model training)
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']


def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == "female" else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Form the feature array in the order expected by your model
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days,
                               extracurricular_activities_encoded, weekly_self_study_hours,
                               math_score, history_score, physics_score, chemistry_score,
                               biology_score, english_score, geography_score, total_score, average_score]])
    scaled_features = scaler.transform(feature_array)
    probabilities = model.predict_proba(scaled_features)

    # Getting top 3 recommendations
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    return top_classes_names_probs


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


@app.route('/pred', methods=['POST', 'GET'])
def pred():
    if request.method == "POST":
        required_fields = [
            'gender', 'part_time_job', 'absence_days', 'extracurricular_activities',
            'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score',
            'chemistry_score', 'biology_score', 'english_score', 'geography_score',
            'total_score', 'average_score'
        ]
        # Ensure all required fields are present
        for field in required_fields:
            if not request.form.get(field):
                return render_template("error.html")
        try:
            gender = request.form['gender']
            part_time_job = request.form['part_time_job'] == 'true'
            absence_days = int(request.form['absence_days'])
            extracurricular_activities = request.form['extracurricular_activities'] == 'true'
            weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
            math_score = float(request.form['math_score'])
            history_score = float(request.form['history_score'])
            physics_score = float(request.form['physics_score'])
            chemistry_score = float(request.form['chemistry_score'])
            biology_score = float(request.form['biology_score'])
            english_score = float(request.form['english_score'])
            geography_score = float(request.form['geography_score'])
            total_score = float(request.form['total_score'])
            average_score = float(request.form['average_score'])
        except ValueError:
            return render_template("error.html")

        # Compute performance parameter based on total_score
        # In this example, assuming each subject is out of 100 with 7 subjects (max 700):
        if total_score < 350:
            performance = "Needs Improvement"
        elif total_score < 490:
            performance = "Good"
        else:
            performance = "Excellent"

        recommendations = Recommendations(
            gender, part_time_job, absence_days, extracurricular_activities,
            weekly_self_study_hours, math_score, history_score, physics_score,
            chemistry_score, biology_score, english_score, geography_score,
            total_score, average_score
        )
        return render_template("results.html", recommendations=recommendations, performance=performance)
    return redirect(url_for("home"))


if __name__ == '__main__':
    app.run(debug=True)