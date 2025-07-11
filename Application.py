from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from src.Pipelines.test_pipeline import Predict_pipeline
from src.Pipelines.test_pipeline import CustomData

application = Flask(__name__)
app = application


# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            Number_of_sexual_partners=int(
                request.form.get("Number_of_sexual_partners")
            ),
            Age=int(request.form.get("Age")),
            First_sexual_intercourse=int(request.form.get("First_sexual_intercourse")),
            Num_of_pregnancies=int(request.form.get("Num_of_pregnancies")),
            Smokes=int(request.form.get("Smokes")),
            Smokes_years=float(request.form.get("Smokes_years")),
            Smokes_packs_per_year=float(request.form.get("Smokes_packs_per_year")),
            Hormonal_Contraceptives=int(request.form.get("Hormonal_Contraceptives")),
            Hormonal_Contraceptives_years=float(
                request.form.get("Hormonal_Contraceptives_years")
            ),
            IUD=int(request.form.get("IUD")),
            IUD_years=float(request.form.get("IUD_years")),
            STDs=int(request.form.get("STDs")),
            STDs_number=int(request.form.get("STDs_number")),
            STDs_condylomatosis=int(request.form.get("STDs_condylomatosis")),
            STDs_cervical_condylomatosis=int(
                request.form.get("STDs_cervical_condylomatosis")
            ),
            STDs_vaginal_condylomatosis=int(
                request.form.get("STDs_vaginal_condylomatosis")
            ),
            STDs_vulvo_perineal_condylomatosis=int(
                request.form.get("STDs_vulvo_perineal_condylomatosis")
            ),
            STDs_syphilis=int(request.form.get("STDs_syphilis")),
            STDs_pelvic_inflammatory_disease=int(
                request.form.get("STDs_pelvic_inflammatory_disease")
            ),
            STDs_genital_herpes=int(request.form.get("STDs_genital_herpes")),
            STDs_molluscum_contagiosum=int(
                request.form.get("STDs_molluscum_contagiosum")
            ),
            STDs_AIDS=int(request.form.get("STDs_AIDS")),
            STDs_HIV=int(request.form.get("STDs_HIV")),
            STDs_Hepatitis_B=int(request.form.get("STDs_Hepatitis_B")),
            STDs_HPV=int(request.form.get("STDs_HPV")),
            STDs_Number_of_diagnosis=int(request.form.get("STDs_Number_of_diagnosis")),
            STDs_Time_since_first_diagnosis=float(
                request.form.get("STDs_Time_since_first_diagnosis")
            ),
            STDs_Time_since_last_diagnosis=float(
                request.form.get("STDs_Time_since_last_diagnosis")
            ),
            Dx_CIN=int(request.form.get("Dx_CIN")),
            Dx_HPV=int(request.form.get("Dx_HPV")),
            Dx=int(request.form.get("Dx")),
            Hinselmann=int(request.form.get("Hinselmann")),
            Schiller=int(request.form.get("Schiller")),
            Citology=int(request.form.get("Citology")),
            Biopsy=int(request.form.get("Biopsy")),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = Predict_pipeline()

        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
