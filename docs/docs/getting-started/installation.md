# Installation

This guide provides instructions for installing Jenga-AI. You can choose between a quick installation using pip or a more comprehensive installation from source.

## Quick Install

To quickly install Jenga-AI and its core dependencies, run the following command:

```bash
pip install Jenga-nlp
```

## Installation from Source

For development or to access the latest features, you can install Jenga-AI directly from its source code.

### 1. Clone the Repository

First, clone the Jenga-AI repository from GitHub:

```bash
git clone https://github.com/your-org/Jenga-AI.git
cd Jenga-AI
```
**Note**: Replace `https://github.com/your-org/Jenga-AI.git` with the actual repository URL if it's different.

### 2. Set up a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Once your virtual environment is active, install the required packages:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries for Jenga-AI.