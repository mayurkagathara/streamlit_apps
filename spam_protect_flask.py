import json
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

THRESHOLD = 1000
sw_code_counts = {}
suppressed_sw_codes = set()
suppress_file = "suppressed_sw_codes.json"

def save_suppressed_codes():
    with open(suppress_file, 'w') as f:
        json.dump(list(suppressed_sw_codes), f)

def load_suppressed_codes():
    global suppressed_sw_codes
    try:
        with open(suppress_file, 'r') as f:
            suppressed_sw_codes = set(json.load(f))
    except FileNotFoundError:
        suppressed_sw_codes = set()

def reset_counts():
    global sw_code_counts, suppressed_sw_codes
    sw_code_counts = {}
    suppressed_sw_codes = set()
    save_suppressed_codes()

scheduler = BackgroundScheduler()
scheduler.add_job(func=reset_counts, trigger="interval", hours=24)
scheduler.start()
load_suppressed_codes()

@app.route('/incident', methods=['POST'])
def incident():
    data = request.json
    sw_code = data.get('sw_code')

    if not sw_code:
        return jsonify({'error': 'sw_code is required'}), 400

    if sw_code in suppressed_sw_codes:
        return jsonify({'status': 'limit reached'}), 200

    if sw_code_counts.get(sw_code, 0) < THRESHOLD:
        sw_code_counts[sw_code] = sw_code_counts.get(sw_code, 0) + 1
        share_to_teams(data)
        return jsonify({'status': 'shared'}), 200
    else:
        suppressed_sw_codes.add(sw_code)
        save_suppressed_codes()
        return jsonify({'status': 'limit reached'}), 200

def share_to_teams(payload):
    # Your logic to share the payload to MS Teams
    pass

if __name__ == '__main__':
    app.run(debug=True)
