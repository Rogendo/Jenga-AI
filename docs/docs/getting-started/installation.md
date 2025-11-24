# Installation

Follow these instructions to set up Jenga-AI on your local machine. We recommend using a virtual environment to manage dependencies and avoid conflicts with other projects.

## Prerequisites

-   Python 3.9+
-   `pip` (Python package installer)

## 1. Create a Virtual Environment

A virtual environment is a self-contained directory that holds a specific Python installation and its packages.

=== "macOS / Linux"

    ```bash
    # Create a directory for your project
    mkdir my-jenga-project
    cd my-jenga-project

    # Create a virtual environment named 'venv'
    python3 -m venv venv

    # Activate the virtual environment
    source venv/bin/activate
    ```

=== "Windows"

    ```bash
    # Create a directory for your project
    mkdir my-jenga-project
    cd my-jenga-project

    # Create a virtual environment named 'venv'
    python -m venv venv

    # Activate the virtual environment
    venv\Scripts\activate
    ```

After activation, your command prompt will be prefixed with `(venv)`, indicating that you are now working inside the virtual environment.

## 2. Install Jenga-AI

You can install Jenga-AI directly from its source code by cloning the repository.

### Clone the Repository

First, clone the Jenga-AI repository from GitHub:

```bash
git clone https://github.com/Rogendo/Jenga-AI.git
cd Jenga-AI
```

### Install Dependencies

Once your virtual environment is active and you are in the `Jenga-AI` directory, install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This command installs all the necessary libraries, including PyTorch, Hugging Face Transformers, and others.

### Install Jenga-AI in Editable Mode

Finally, install the Jenga-AI framework itself in "editable" mode. This allows you to make changes to the source code and have them immediately reflected in your environment, which is ideal for development.

```bash
pip install -e .
```

## 3. Verify Installation

To ensure that the framework is installed correctly, you can run a quick test by importing one of the core classes in a Python interpreter:

```bash
python
```

Then, within the Python interpreter:

```python
try:
    from multitask_bert.core.config import ExperimentConfig
    print("✅ Jenga-AI installation successful!")
except ImportError as e:
    print(f"❌ Installation failed: {e}")
```

If you see the success message, you are all set! You can now proceed to the **[Quickstart Guide](quickstart.md)** to train your first model.
