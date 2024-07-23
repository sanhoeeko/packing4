#!/bin/bash

# copy files
python3 project.py

# remove old output
rm std.out std.err

# run subprojects
for dir in EXEC*/; do
    if [[ -f "$dir/deploy.py" ]]; then
        echo "Running deploy.py in $dir"
        (
            cd "$dir"
            python3 deploy.py > sub.out 
        )
        echo "PID of deploy.py: $!"
    else
        echo "No deploy.py found in $dir"
    fi
done
wait