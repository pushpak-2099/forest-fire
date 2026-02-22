from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)

# Load model
ridge_model = pickle.load(open('model/ridgereg.pkl', 'rb'))
scaler_model = pickle.load(open('model/scaler.pkl', 'rb'))

@application.route("/")
def index():
    return render_template("index.html")

@application.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
<<<<<<< HEAD
        RH = float(request.form.get("RH"))  
=======
        RH = float(request.form.get("RH"))
>>>>>>> 4dbc95ae193ecaa33bf16567f987836adeae2ff8
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        scaled_data = scaler_model.transform(input_data)
        result = ridge_model.predict(scaled_data)

<<<<<<< HEAD
        return render_template("index.html", results=result[0])
=======
        return render_template("home.html", results=result[0])
>>>>>>> 4dbc95ae193ecaa33bf16567f987836adeae2ff8

    return render_template("home.html")

if __name__ == "__main__":
    application.run(debug=True)