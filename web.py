import os
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

MODEL_PATH = 'mouse_trajectory_regressor.keras'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

dummy_features = np.array([
    [0, 0, 0],
    [1000, 600, 100]
])
dummy_targets = np.array([
    [0, 0],
    [1000, 600]
])

feature_scaler.fit(dummy_features)
target_scaler.fit(dummy_targets)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Mouse Trajectory Predictor</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #0e1117;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 { margin-bottom: 5px; color: #f1f1f1; }
        p { margin-bottom: 20px; color: #888; font-size: 14px; }
        .container { display: flex; gap: 30px; max-width: 1300px; width: 100%; }
        canvas {
            background: #131722;
            border: 2px solid #30363d;
            border-radius: 8px;
            cursor: crosshair;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .dashboard {
            flex: 1;
            background: #1c212c;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .metric {
            border-left: 4px solid #2196F3;
            padding-left: 15px;
            margin-bottom: 10px;
        }
        .metric-yellow { border-left-color: #FFEB3B; }
        .label { font-size: 12px; color: #aaa; text-transform: uppercase; }
        .value { font-size: 24px; font-weight: bold; margin-top: 5px; }
    </style>
</head>
<body>

    <h1>🎯 Mouse Trajectory Regression Layer</h1>
    <p>Move your mouse inside the canvas to generate predictions from the loaded Keras model.</p>

    <div class="container">
        <canvas id="canvas" width="1000" height="600"></canvas>
        
        <div class="dashboard">
            <div class="metric metric-yellow">
                <div class="label">Predicted Next X</div>
                <div class="value" id="predX">---</div>
            </div>
            <div class="metric metric-yellow">
                <div class="label">Predicted Next Y</div>
                <div class="value" id="predY">---</div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const predXMetric = document.getElementById('predX');
        const predYMetric = document.getElementById('predY');

        let trackingHistory = [];
        let predictedX = null;
        let predictedY = null;

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (trackingHistory.length > 1) {
                ctx.beginPath();
                ctx.moveTo(trackingHistory[0].x, trackingHistory[0].y);
                for (let i = 1; i < trackingHistory.length; i++) {
                    ctx.lineTo(trackingHistory[i].x, trackingHistory[i].y);
                }
                ctx.strokeStyle = '#2196F3';
                ctx.lineWidth = 4;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.stroke();
            }

            if (predictedX !== null && predictedY !== null && !isNaN(predictedX) && !isNaN(predictedY)) {
                ctx.beginPath();
                ctx.arc(predictedX, predictedY, 8, 0, 2 * Math.PI);
                ctx.fillStyle = '#FFEB3B'; 
                ctx.fill();
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            requestAnimationFrame(animate);
        }

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const now = performance.now();

            let dt = 0;
            if (trackingHistory.length > 0) {
                const prev = trackingHistory[trackingHistory.length - 1];
                dt = now - prev.time;
            }

            trackingHistory.push({ x: x, y: y, dt: dt, time: now });

            if (trackingHistory.length > 10) {
                trackingHistory.shift();
            }

            if (trackingHistory.length === 10) {
                const payload = trackingHistory.map(p => [p.x, p.y, p.dt]);

                fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence: payload })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        predictedX = data.x;
                        predictedY = data.y;
                        predXMetric.innerText = predictedX.toFixed(1) + " px";
                        predYMetric.innerText = predictedY.toFixed(1) + " px";
                    }
                })
                .catch(err => {});
            }
        });

        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sequence = np.array(data['sequence'], dtype=np.float32)
        
        if model is not None:
            sequence[:, 0] = np.clip(sequence[:, 0], 0, 1000)
            sequence[:, 1] = np.clip(sequence[:, 1], 0, 600)
            sequence[:, 2] = np.clip(sequence[:, 2], 0, 100)

            scaled_features = feature_scaler.transform(sequence)
            batch_input = np.expand_dims(scaled_features, axis=0)
            
            scaled_pred = model.predict(batch_input, verbose=0)[0]
            real_pred = target_scaler.inverse_transform([scaled_pred])[0]
            
            out_x = float(np.clip(real_pred[0], 0, 1000))
            out_y = float(np.clip(real_pred[1], 0, 600))
            
            return jsonify({'success': True, 'x': out_x, 'y': out_y})
        else:
            last_point = sequence[-1]
            return jsonify({
                'success': True,
                'x': float(np.clip(last_point[0] + 5, 0, 1000)),
                'y': float(np.clip(last_point[1] + 5, 0, 600))
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)