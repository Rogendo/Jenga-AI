

echo "Setting up Jenga-NLP Documentation..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install mkdocs mkdocs-material mkdocs-autorefs mkdocs-awesome-pages-plugin

# Create basic structure (if not exists)
if [ ! -f "mkdocs.yml" ]; then
    mkdocs new .
fi

echo "Setup complete! Run 'mkdocs serve' to start local server."
echo "Documentation will be available at: http://127.0.0.1:8000"