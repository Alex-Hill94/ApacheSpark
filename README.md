# Introductory Apache Spark Workshop

This repository supports an in-person workshop introducing **Apache Spark** and its Python API, PySpark.

The workshop is designed to build an intuition for Spark through practical examples, followed by an activity preparing data for machine learning.

## Getting Started

Clone the repository and navigate into the project directory.

### 1. Set up the environment

Create and test the Conda environment before starting the workshop:

```bash
bash create_env.sh
bash test_env.sh
```

### 2. Work through the examples

Work through the Python scripts in the main directory, first BasicExample.py, then MoreDataExample.py, and finally Functions.py. These examples introduce the core concepts and build an intuition for working with Apache Spark.

### 3. MLlib Activity

Once you have completed the introductory scripts, move into the `MLlib_Activity` folder.

Work through the script to clean and prepare data ahead of machine learning categorisation using Spark MLlib.

## Repository Structure

```text
.
├── create_env.sh
├── test_env.sh
├── *.py
└── MLlib_Activity/
```

**Prerequisites:** Conda and a working Python installation.

Refer to the accompanying presentation for workshop explanations and additional guidance.
