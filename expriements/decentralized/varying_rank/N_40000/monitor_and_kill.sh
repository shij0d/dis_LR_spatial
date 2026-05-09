#!/bin/bash

# Time interval in seconds
interval=10

# Define the file path, replace {m} with the desired value or leave it dynamic
m_value=100  
file_path="/home/shij0d/documents/dis_LR_spatial/expriements/decentralized/varying_rank/N_40000/m_${m_value}_memeff.pkl"

while true; do
  # Check if the file exists
  if [ -f "$file_path" ]; then
    echo "File $file_path exists. Killing all running Python processes."

    # Kill all Python processes (change 'python3' to 'python' if needed)
    pkill -f python3

    # Optionally exit the loop after killing
    break
  else
    echo "File $file_path does not exist. Checking again in $interval seconds."
  fi

  # Wait for the specified interval before checking again
  sleep $interval
done
