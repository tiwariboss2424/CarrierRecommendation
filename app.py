from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler, model, and class names
try:
    scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
    model = pickle.load(open("Models/model.pkl", 'rb'))
except FileNotFoundError:
    print("Error: Model or Scaler not found. Ensure 'Models/scaler.pkl' and 'Models/model.pkl' exist.")
    scaler = None
    model = None

class_names = [
    'Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
    'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
    'Banker', 'Writer', 'Accountant', 'Designer',
    'Construction Engineer', 'Game Developer', 'Stock Investor',
    'Real Estate Developer'
]


# Function for generating recommendations
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    if scaler is None or model is None:
        return [("Error: Missing Model or Scaler", 0.0)]

    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[
        gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
        weekly_self_study_hours, math_score, history_score, physics_score,
        chemistry_score, biology_score, english_score, geography_score, total_score,
        average_score
    ]])

    # Scale features
    try:
        scaled_features = scaler.transform(feature_array)
    except ValueError:
        print("Error: Feature shape mismatch. Ensure inputs match trained format.")
        return [("Error: Invalid input shape", 0.0)]

    # Predict probabilities
    probabilities = model.predict_proba(scaled_features)

    # Get top 3 recommended studies
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    recommendations = [(class_names[idx], round(probabilities[0][idx], 2)) for idx in top_classes_idx]

    return recommendations


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')



@app.route('/pred', methods=['POST'])
def pred():
    try:
        # Ensure form data exists and is not empty
        if not request.form or not any(request.form.values()):
            return render_template("error.html", error_message="⚠ No Data Entered. Please fill in the required fields.")

        try:
            # Extract and convert form inputs safely
            gender = request.form.get('gender', 'male')
            part_time_job = request.form.get('part_time_job', 'false') == 'true'
            absence_days = int(request.form.get('absence_days', 0))
            extracurricular_activities = request.form.get('extracurricular_activities', 'false') == 'true'
            weekly_self_study_hours = int(request.form.get('weekly_self_study_hours', 0))
            math_score = int(request.form.get('math_score', 0))
            history_score = int(request.form.get('history_score', 0))
            physics_score = int(request.form.get('physics_score', 0))
            chemistry_score = int(request.form.get('chemistry_score', 0))
            biology_score = int(request.form.get('biology_score', 0))
            english_score = int(request.form.get('english_score', 0))
            geography_score = int(request.form.get('geography_score', 0))
            total_score = float(request.form.get('total_score', 0.0))
            average_score = float(request.form.get('average_score', 0.0))

        except ValueError as e:
            print(f"Error converting input values: {e}")
            return render_template("error.html", error_message="⚠ Invalid data entered. Please ensure all fields are numeric.")

        # Generate recommendations
        recommendations = Recommendations(
            gender, part_time_job, absence_days, extracurricular_activities,
            weekly_self_study_hours, math_score, history_score, physics_score,
            chemistry_score, biology_score, english_score, geography_score,
            total_score, average_score
        )

        # If no valid recommendations, redirect to error page
        if not recommendations or recommendations[0][0] == "Error":
            return render_template("error.html", error_message="⚠ No suitable recommendations found. Try refining your input.")

        return render_template("results.html", recommendations=recommendations)

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return render_template("error.html", error_message="⚠ An unexpected error occurred. Please try again.")

        # Generate recommendations
        recommendations = Recommendations(
            gender, part_time_job, absence_days, extracurricular_activities,
            weekly_self_study_hours, math_score, history_score, physics_score,
            chemistry_score, biology_score, english_score, geography_score,
            total_score, average_score
        )

        return render_template('results.html', recommendations=recommendations)

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return render_template("error.html", error_message="⚠ An unexpected error occurred. Please try again.")

    except Exception as e:
        print(f"Error processing form: {e}")
        return render_template("error.html", error_message="⚠ An error occurred while processing your request. Please try again.")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)