#!/bin/bash
echo "Setting up T2I Gender Bias Experiment Pipeline..."

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtualenv and install requirements
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Handle .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=\"your_openai_api_key_here\"" > .env
    echo "HF_API_KEY=\"your_huggingface_token_here\"" >> .env
    echo ""
    echo "IMPORTANT: Please edit the .env file in this directory"
    echo "and add your actual OpenAI API Key and Hugging Face Token."
else
    echo ".env file already exists."
fi

echo ""
echo "Setup complete!"
echo "To run the experiment:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Add your keys to the .env file"
echo "3. Run the script: python run_experiment.py"
