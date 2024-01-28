import pickle
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POSt'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            locality = request.form.get('locality'),
            facing = request.form.get('facing'),
            parking = request.form.get('parking'),
            BHK = request.form.get('BHK'),
            bathrooms = request.form.get('bathrooms'),
            area = request.form.get('area'),
            price_per_sqft = request.form.get('price_per_sqft')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)