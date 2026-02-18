from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Create database
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (hours REAL, attendance REAL, previous_score REAL, result TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    attendance = float(request.form["attendance"])
    prev_score = float(request.form["previous_score"])

    data = np.array([[hours, attendance, prev_score]])
    data = scaler.transform(data)
    
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    result = "PASS" if prediction[0] == 1 else "FAIL"

    pass_prob = round(probability[0][1] * 100, 2)
    fail_prob = round(probability[0][0] * 100, 2)

  
      # Save result
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
              (hours, attendance, prev_score, result))
    conn.commit()
    conn.close()

    return render_template(
    "result.html",
    result=result,
    pass_prob=pass_prob,
    fail_prob=fail_prob
)


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions")
    data = c.fetchall()
    conn.close()
    
    return render_template("dashboard.html", data=data)
 
if __name__ == "__main__":
    app.run(debug=True)
