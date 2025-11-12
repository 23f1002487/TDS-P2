#!/bin/bash
# Setup script for .env configuration

echo "=========================================="
echo "  TDS-P2 Environment Setup"
echo "=========================================="
echo ""

# Check if .env exists
if [ -f .env ]; then
    echo "✓ .env file exists"
    echo ""
    echo "Current configuration:"
    grep -E "^(STUDENT_EMAIL|AIPIPE_TOKEN|OPENAI_MODEL)=" .env
    echo ""
    read -p "Do you want to update it? (y/n): " update
    if [ "$update" != "y" ]; then
        echo "Keeping existing configuration."
        exit 0
    fi
else
    echo "Creating .env from template..."
    cp .env.example .env
fi

echo ""
echo "Please provide your credentials:"
echo ""

# Get student email
read -p "Student Email (e.g., 23f1002487@ds.study.iitm.ac.in): " email
if [ ! -z "$email" ]; then
    sed -i "s|^STUDENT_EMAIL=.*|STUDENT_EMAIL=$email|" .env
fi

# Get secret
read -p "Student Secret: " secret
if [ ! -z "$secret" ]; then
    sed -i "s|^STUDENT_SECRET=.*|STUDENT_SECRET=$secret|" .env
fi

# Get AIPipe token
echo ""
echo "Get your AIPipe token from: https://aipipe.org"
read -p "AIPipe Token: " token
if [ ! -z "$token" ]; then
    sed -i "s|^AIPIPE_TOKEN=.*|AIPIPE_TOKEN=$token|" .env
fi

# Ask about model
echo ""
echo "Available models:"
echo "  1) openai/gpt-4o-mini (fast, cheap)"
echo "  2) openai/gpt-3.5-turbo (fast)"
echo "  3) openai/gpt-4o (best quality)"
read -p "Choose model (1-3) [default: 1]: " model_choice

case $model_choice in
    2) model="openai/gpt-3.5-turbo" ;;
    3) model="openai/gpt-4o" ;;
    *) model="openai/gpt-4o-mini" ;;
esac

sed -i "s|^OPENAI_MODEL=.*|OPENAI_MODEL=$model|" .env

echo ""
echo "=========================================="
echo "✓ Configuration saved to .env"
echo "=========================================="
echo ""
echo "You can now test AIPipe connectivity:"
echo "  source .venv/bin/activate"
echo "  python3 test_aipipe.py"
echo ""
