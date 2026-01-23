#!/bin/bash

# Script to fix file permissions for Docker-created files
# Makes files and directories readable and writable by everyone

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory> [directory2] [directory3] ..."
    echo "Example: $0 datasets robomimic/exps/results"
    exit 1
fi

for dir in "$@"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Directory '$dir' does not exist"
        continue
    fi
    
    echo "Fixing permissions in: $dir"
    
    # Add read/write permissions for all users on all files and directories
    chmod -R a+rw "$dir"
    
    # Also ensure directories are executable so they can be entered
    find "$dir" -type d -exec chmod a+x {} \;
    
    echo "✓ Done fixing permissions in: $dir"
done

echo ""
echo "All permissions fixed!"
