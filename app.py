import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

scalers = {
    "rating": pickle.load(open("scaler_rating.pkl", "rb")),
    "average_meal_price": pickle.load(open("scaler_AVG_mealprice.pkl", "rb")),
    "marketing_budget": pickle.load(open("scaler_MK_budget.pkl", "rb")),
    "social_media_followers": pickle.load(open("scaler_Social_Media_Follow.pkl", "rb")),
    "chef_experience_years": pickle.load(open("scaler_Chef_Exp_years.pkl", "rb")),
    "number_of_reviews": pickle.load(open("scaler_Numb_reviews.pkl", "rb")),
    "service_quality_score": pickle.load(open("scaler_Service_Quality_Score.pkl", "rb")),
    "weekend_reservations": pickle.load(open("scaler_weekend.pkl", "rb")),
    "weekday_reservations": pickle.load(open("scaler_weekday.pkl", "rb")),
}

model = pickle.load(open("model.pkl", "rb"))

filtered_df = pd.read_csv("filtered_df.csv")

location_columns = ["Location_Downtown", "Location_Rural", "Location_Suburban"]
cuisine_columns = [
    "Cuisine_American", "Cuisine_French", "Cuisine_Indian",
    "Cuisine_Italian", "Cuisine_Japanese", "Cuisine_Mexican"
]

first_prediction = None

def transform_input(data, scalers):
    location_dummy = [1.0 if col == f"Location_{data['location']}" else 0.0 for col in location_columns]
    cuisine_dummy = [1.0 if col == f"Cuisine_{data['cuisine']}" else 0.0 for col in cuisine_columns]
    transformed_data = [
        float(scalers["rating"].transform([[data["rating"]]])[0][0]),
        float(data["seating_capacity"]),
        float(scalers["average_meal_price"].transform([[data["average_meal_price"]]])[0][0]),
        float(scalers["marketing_budget"].transform([[data["marketing_budget"]]])[0][0]),
        float(scalers["social_media_followers"].transform([[data["social_media_followers"]]])[0][0]),
        float(scalers["chef_experience_years"].transform([[data["chef_experience_years"]]])[0][0]),
        float(scalers["number_of_reviews"].transform([[data["number_of_reviews"]]])[0][0]),
        float(scalers["service_quality_score"].transform([[data["service_quality_score"]]])[0][0]),
        float(1.0 if data["parking_availability"] == "Yes" else 0.0),
        float(scalers["weekend_reservations"].transform([[data["weekend_reservations"]]])[0][0]),
        float(scalers["weekday_reservations"].transform([[data["weekday_reservations"]]])[0][0]),
        *location_dummy,
        *cuisine_dummy,
    ]
    return np.array(transformed_data, dtype=np.float32).reshape(1, -1)

@app.route("/")
def index():
    default_user_input = {
        "restaurant_name": "",
        "country": "",
        "city": "",
        "email": "",
        "location": "Downtown",
        "cuisine": "",
        "rating": 0.0,
        "seating_capacity": 0,
        "average_meal_price": 0.0,
        "marketing_budget": 0.0,
        "social_media_followers": 0,
        "chef_experience_years": 0,
        "number_of_reviews": 0,
        "service_quality_score": 0.0,
        "parking_availability": "No",
        "weekend_reservations": 0,
        "weekday_reservations": 0,
    }
    return render_template("index.html", user_input=default_user_input, search_result=None)


@app.route("/search", methods=["POST"])
def search():
    
    restaurant_name = request.form.get("restaurant_name", "")

    filtered_row = filtered_df[filtered_df["Name"].str.lower() == restaurant_name.lower()]

    user_input = {
        "restaurant_name": restaurant_name,
        "country": "",
        "city": filtered_row["City"].values[0] if not filtered_row.empty else "",
        "email": "",  
        "location": "Downtown",
        "cuisine": filtered_row["Filtered Cuisines"].values[0] if not filtered_row.empty else "",
        "rating": filtered_row["Rating"].values[0] if not filtered_row.empty else 0.0,
        "seating_capacity": 0,
        "average_meal_price": filtered_row["Price Range"].values[0] if not filtered_row.empty else 0.0,
        "marketing_budget": 0.0,
        "social_media_followers": 0,
        "chef_experience_years": 0,
        "number_of_reviews": filtered_row["Number of Reviews"].values[0] if not filtered_row.empty else 0,
        "service_quality_score": 0.0,
        "parking_availability": "No",
        "weekend_reservations": 0,
        "weekday_reservations": 0,
    }

    return render_template("index.html", user_input=user_input, prediction=None, percentage_change=None)

model_NoMK = pickle.load(open("model_NoMK.pkl", "rb"))
model_NoSM = pickle.load(open("model_NoSM.pkl", "rb"))
model_NoMKSM = pickle.load(open("model_NoMKSM.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    global first_prediction

    user_input = {key: request.form[key] for key in request.form.keys()}
    
    transformed_input = transform_input({
        k: float(v) if k in scalers.keys() else v for k, v in user_input.items()
    }, scalers)
    
    marketing_budget = float(user_input.get("marketing_budget", 0))
    social_media_followers = float(user_input.get("social_media_followers", 0))
    
    if marketing_budget == 0 and social_media_followers == 0:
       
        adjusted_input = np.delete(transformed_input, [3, 4], axis=1)  
        prediction = model_NoMKSM.predict(adjusted_input)[0]
    elif marketing_budget == 0:
        
        adjusted_input = np.delete(transformed_input, 3, axis=1)  
        prediction = model_NoMK.predict(adjusted_input)[0]
    elif social_media_followers == 0:
        
        adjusted_input = np.delete(transformed_input, 4, axis=1)  
        prediction = model_NoSM.predict(adjusted_input)[0]
    else:
        prediction = model.predict(transformed_input)[0]

    
    if first_prediction is None:
        first_prediction = prediction

    return render_template("index.html", prediction=f"Predicted Revenue: ${prediction:,.2f}", user_input=user_input)


@app.route("/show_change", methods=["POST"])
def show_change():
    global first_prediction

    
    user_input = {key: request.form[key] for key in request.form.keys()}
    
    
    transformed_input = transform_input({
        k: float(v) if k in scalers.keys() else v for k, v in user_input.items()
    }, scalers)
    
    
    marketing_budget = float(user_input.get("marketing_budget", 0))
    social_media_followers = float(user_input.get("social_media_followers", 0))
    
    if marketing_budget == 0 and social_media_followers == 0:
        
        adjusted_input = np.delete(transformed_input, [3, 4], axis=1)
        prediction = model_NoMKSM.predict(adjusted_input)[0]
    elif marketing_budget == 0:
        
        adjusted_input = np.delete(transformed_input, 3, axis=1)
        prediction = model_NoMK.predict(adjusted_input)[0]
    elif social_media_followers == 0:
        
        adjusted_input = np.delete(transformed_input, 4, axis=1)
        prediction = model_NoSM.predict(adjusted_input)[0]
    else:
        prediction = model.predict(transformed_input)[0]

    
    percentage_change = ((prediction - first_prediction) / first_prediction) * 100 if first_prediction is not None else None

    return render_template(
        "index.html",
        percentage_change=f"Change: {percentage_change:+.2f}%" if percentage_change else None,
        user_input=user_input
    )
if __name__ == "__main__":
    app.run(debug=True)
