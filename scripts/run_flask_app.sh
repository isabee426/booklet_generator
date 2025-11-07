#!/bin/bash
# Run ARC Puzzle Creator - Flask Version

echo "🧩 Starting ARC Puzzle Creator (Flask)..."
echo ""

# Check if Flask is installed
if ! pip list | grep -q Flask; then
    echo "⚠️  Flask not found. Installing dependencies..."
    pip install -r requirements_flask.txt
    echo ""
fi

# Check if ARC data exists
ARC_DATA_PATH="../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training"
if [ ! -d "$ARC_DATA_PATH" ]; then
    echo "⚠️  Warning: ARC data not found at expected location:"
    echo "   $ARC_DATA_PATH"
    echo "   The app may not work properly without the data."
    echo ""
fi

echo "🚀 Launching Flask web server..."
echo "   The app will open at http://localhost:5000"
echo "   Press Ctrl+C to stop the server"
echo ""

# Run Flask app
python app.py

