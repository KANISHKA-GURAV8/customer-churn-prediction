from flask import Flask,render_template,request
import pandas as pd
import joblib

app=Flask(__name__)
model=joblib.load("best_churn_model.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data={
        "AccountWeeks":float(request.form["AccountWeeks"]),
        "ContractRenewal":float(request.form["ContractRenewal"]),
        "DataPlan":float(request.form["DataPlan"]),
        "DataUsage":float(request.form["DataUsage"]),
        "CustServCalls":int(request.form["CustServCalls"]),
        "DayMins":float(request.form["DayMins"]),
        "DayCalls":int(request.form["DayCalls"]),
        "MonthlyCharge":float(request.form["MonthlyCharge"]),
        "OverageFee":float(request.form["OverageFee"]),
        "RoamMins":float(request.form["RoamMins"])


    }

    input_df=pd.DataFrame([data])
    prediction=model.predict(input_df)[0]
    probability=model.predict_proba(input_df)[0][1]
    result="Customer will churn" if prediction==1 else "customer will not churn"

    return render_template(
        "result.html",
        result=result,
        probability=f"{probability*100:2f}%"
    )

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
