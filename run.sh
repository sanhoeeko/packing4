#!/bin/bash

# copy files
python3 project.py

# run subprojects
for dir in EXEC*/; do
    if [[ -f "$dir/deploy.py" ]]; then
        echo "Running deploy.py in $dir"
        cd "$dir"
        python3 "deploy.py" &
        echo "PID of deploy.py: $!"
        cd ..
    else
        echo "No deploy.py found in $dir"
    fi
done