#!/bin/bash

# Define the paths to the Streamlit apps
APP1_PATH="a/b/one.py"
APP2_PATH="a/c/two.py"
APP3_PATH="a/d/three.py"

# Define the ports for each app
APP1_PORT=8051
APP2_PORT=8052
APP3_PORT=8053

# Function to check if an app is running
check_app_status() {
    local app_path=$1
    local app_port=$2
    local app_name=$(basename "$app_path" .py)
    local app_pid=$(pgrep -f "streamlit run $app_path")

    if [ -z "$app_pid" ]; then
        echo "$app_name is not running. Starting $app_name..."
        streamlit run "$app_path" --server.port "$app_port" &
    else
        echo "$app_name is already running (PID: $app_pid)"
    fi
}

# Main loop to monitor app status
while true; do
    check_app_status "$APP1_PATH" "$APP1_PORT"
    check_app_status "$APP2_PATH" "$APP2_PORT"
    check_app_status "$APP3_PATH" "$APP3_PORT"
    sleep 300  # Wait for 5 minutes before checking again
done


# service
# Create a service file (e.g., streamlit-monitor.service) in the /etc/systemd/system/ directory with the following content:
# [Unit]
# Description=Streamlit App Monitor
# After=network.target

# [Service]
# ExecStart=/path/to/your/script.sh
# Restart=always
# User=your_username

# [Install]
# WantedBy=multi-user.target

# sudo systemctl enable streamlit-monitor.service
# sudo systemctl start streamlit-monitor.service
