import logging
from flask import Flask, request, jsonify
import requests
from requests.exceptions import RequestException
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

def create_message(payload):
    payload['channel_id'] = 'XYZ'
    return payload

@app.route('/incident', methods=['POST'])
def incident():
    try:
        payload = request.json
        if not payload:
            raise ValueError("No JSON payload received")
        # Log the received payload
        logging.info(f'Received payload: {payload}')
        processed_payload = create_message(payload)
        response = requests.post('https://myurl.team.com/message', json=processed_payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return jsonify(response.json())
    except ValueError as ve:
        logging.error(f'ValueError: {ve}')
        return jsonify({'error': str(ve)}), 400
    except RequestException as re:
        logging.error(f'RequestException: {re}')
        return jsonify({'error': 'Failed to send request to external service'}), 502
    except HTTPException as he:
        logging.error(f'HTTPException: {he}')
        return jsonify({'error': 'HTTP error occurred'}), he.code
    except Exception as e:
        logging.error(f'Unhandled Exception: {e}')
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890)

# To create the service do the following
# Create a Service File: Create a file named flaskapp.service in /etc/systemd/system/ with the following content:
# [Unit]
# Description=Flask Application

# [Service]
# ExecStart=/usr/bin/python3 /path/to/your/app.py
# Restart=always
# User=nobody
# Group=nogroup
# Environment=PATH=/usr/bin
# Environment=FLASK_APP=/path/to/your/app.py

# [Install]
# WantedBy=multi-user.target

# Reload Systemd and Start the Service:
# sudo systemctl daemon-reload
# sudo systemctl start flaskapp
# sudo systemctl enable flaskapp
