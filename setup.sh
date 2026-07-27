#!/bin/bash

echo "Select OS:"
echo "1) Windows"
echo "2) macOS"
echo "3) Ubuntu/Linux"
read -p "Enter choice (1-3): " choice

PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/venv"

case $choice in
    1)
        echo "Setting up Python virtual environment for Windows..."
        python -m venv "$VENV_DIR"

        source "$VENV_DIR/Scripts/activate"

        if [ -f "$PROJECT_DIR/requirements.txt" ]; then
            pip install -r "$PROJECT_DIR/requirements.txt"
        else
            echo "requirements.txt not found!"
        fi
        ;;
    2|3)
        echo "Setting up Python virtual environment..."
        python3 -m venv "$VENV_DIR"

        source "$VENV_DIR/bin/activate"

        if [ -f "$PROJECT_DIR/requirements.txt" ]; then
            pip install -r "$PROJECT_DIR/requirements.txt"
        else
            echo "requirements.txt not found!"
        fi
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "Setup completed."
echo "Virtual environment created at: $VENV_DIR"