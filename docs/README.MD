# Jenga-NLP Documentation

This directory contains the source files for the Jenga-NLP documentation, which is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## Prerequisites

Before you can build or serve the documentation, you need to have Python and `pip` installed on your system.

## Setup

To get started with the documentation, follow these steps:

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**

    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Building the Documentation

To build the static documentation site, run the following command:

```bash
mkdocs build
```

This will generate the static site in the `site` directory.

## Serving the Documentation Locally

To serve the documentation on a local development server, run the following command:

```bash
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000`. The server will automatically reload the site when you make changes to the documentation source files.

## The `setup-docs.sh` script

For convenience, you can also use the `setup-docs.sh` script to set up the documentation environment. This script will create a virtual environment and install the required dependencies.

To use the script, run the following commands:

```bash
chmod +x setup-docs.sh
./setup-docs.sh