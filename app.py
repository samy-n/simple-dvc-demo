from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import yaml
from prediction_service import prediction

webappp_root="webapp"
static_dir=os.path.join(webappp_root,"static")
template_dir=os.path.join(webappp_root,"templates")

app=Flask(__name__, static_folder=static_dir,template_folder=template_dir)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                data_req = dict(request.form)
                response = prediction.form_response(data_req)
                return render_template("index.html", response=response)

            elif request.json:
                response = prediction.api_response(request)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": e}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__=="__main__":
    app.run(host="localhost",port=5000, debug=True)