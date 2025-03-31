from flask import Flask, request, render_template_string
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('model.keras')

HTML_PAGE = '''
<!doctype html>
<html>
    <head><title>Simple AI Prediction</title></head>
    <body style="text-align:center;margin-top:50px;">
        <h2>AI Model Prediction: y = 2x + 15</h2>
        <form method="post">
            <input type="number" name="x_value" placeholder="Enter x" required>
            <button type="submit">Predict y</button>
        </form>
        {% if prediction %}
            <h3>Prediction (y): {{ prediction }}</h3>
        {% endif %}
    </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        x = float(request.form['x_value'])
        x_scaled = x / 50000.0
        y_scaled = model.predict([[x_scaled]])
        y_pred = y_scaled[0][0] * (2 * 50000 + 15)
        prediction = round(y_pred, 2)
    return render_template_string(HTML_PAGE, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
