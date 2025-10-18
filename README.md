# 🌸 IRIS Pipeline with DVC, CI/CD, and MLOps Best Practices

## 🎯 Project Overview

This project implements a complete MLOps pipeline for IRIS flower classification using **DVC (Data Version Control)**, **GitHub Actions CI/CD**, and **Google Cloud Storage**. The pipeline demonstrates best practices for data versioning, model training, evaluation, and continuous integration.

## ✨ Key Features

### 🔄 **Data & Model Versioning**
* **DVC Integration**: Track and version datasets and models with DVC
* **GCS Remote Storage**: Store data and models in Google Cloud Storage
* **Version Traversal**: Seamlessly switch between different data and model versions
* **Git Tags**: Use semantic versioning (v1.0, v2.0) for easy navigation

### 🧪 **Testing & Validation**
* **Data Validation**: Comprehensive unit tests for data quality checks
* **Model Evaluation**: Automated testing of model performance and metrics
* **Pytest Integration**: Full test suite with coverage reporting
* **CI/CD Pipeline**: Automated testing on every push and pull request

### 🚀 **CI/CD Pipeline**
* **GitHub Actions**: Automated CI/CD for both `dev` and `main` branches
* **Branch-specific Workflows**: Separate CI pipelines for development and production
* **DVC Integration**: Automated data and model fetching from remote storage
* **CML Reports**: Automated test reports as GitHub comments

### 📊 **Monitoring & Reporting**
* **Test Coverage**: Comprehensive code coverage reporting
* **Performance Metrics**: Automated model evaluation and reporting
* **CML Integration**: Beautiful test reports with Iterative CML
* **Artifact Management**: Automated storage of test results and coverage reports

## 🏗️ Project Structure

```
iris-pipeline/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_processing.py        # Data loading and validation
│   ├── model_training.py         # Model training and evaluation
│   └── dvc_operations.py         # DVC remote operations
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_data_validation.py   # Data validation tests
│   └── test_model_evaluation.py  # Model evaluation tests
├── .github/workflows/            # GitHub Actions CI/CD
│   ├── ci-dev.yml               # CI for dev branch
│   └── ci-main.yml              # CI for main branch
├── main.py                      # Main pipeline script
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── pytest.ini                  # Pytest configuration
├── setup.py                    # Package setup
└── README.md                   # This file
```

## ⚙️ Prerequisites

* **Python 3.8+**
* **Git** with proper configuration
* **Google Cloud Platform** project with billing enabled
* **GitHub repository** with Actions enabled
* **DVC** for data version control
* **Required Python packages** (see `requirements.txt`)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd iris-pipeline
pip install -r requirements.txt
```

### 2. Configure DVC Remote

```bash
dvc remote add -d gcsremote gs://your-bucket/iris-pipeline
```

### 3. Pull Data and Model

```bash
dvc pull iris-dvc-pipeline/v1_data.csv
dvc pull iris-dvc-pipeline/model.joblib
```

### 4. Run Tests

```bash
pytest tests/ -v --cov=src
```

### 5. Run Pipeline

```bash
python main.py --setup-dvc
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Data validation tests
pytest tests/test_data_validation.py -v

# Model evaluation tests
pytest tests/test_model_evaluation.py -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Test Coverage
The project maintains comprehensive test coverage:
- **Data Validation**: 100% coverage of data processing functions
- **Model Evaluation**: 100% coverage of model training and evaluation
- **DVC Operations**: Integration tests for remote operations

## 🔄 CI/CD Pipeline

### Branch Strategy
- **`dev` branch**: Development and testing
- **`main` branch**: Production-ready code

### GitHub Actions Workflows

#### Dev Branch CI (`ci-dev.yml`)
- Runs on every push to `dev` branch
- Pulls data and model from DVC remote
- Runs comprehensive test suite
- Generates coverage reports
- Creates CML test reports as GitHub comments

#### Main Branch CI (`ci-main.yml`)
- Runs on every push to `main` branch
- Includes all dev branch tests
- Additional version traversal testing
- Comprehensive reporting with CML

### Required Secrets
Configure these secrets in your GitHub repository:
- `GCP_SA_KEY`: Google Cloud Service Account JSON key
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

## 📊 DVC Configuration

### Remote Storage
```bash
# GCS bucket configuration
BUCKET_URI = "gs://mlops-course-verdant-victory-473118-k0-unique-week2-2/iris-pipeline"
```

### Version Management
- **v1.0**: Initial dataset and model
- **v2.0**: Augmented dataset and retrained model

### Commands
```bash
# Pull specific version
git checkout v1.0
dvc checkout

# Pull from remote
dvc pull iris-dvc-pipeline/v1_data.csv
dvc pull iris-dvc-pipeline/model.joblib
```

## 🎛️ Configuration

### Main Configuration (`config.yaml`)
```yaml
data:
  test_size: 0.4
  random_state: 42
  feature_columns: [sepal_length, sepal_width, petal_length, petal_width]
  target_column: species

model:
  type: DecisionTreeClassifier
  max_depth: 3
  random_state: 1

evaluation:
  min_accuracy: 0.85
  max_accuracy: 1.0
```

## 📈 Usage Examples

### Basic Pipeline Execution
```bash
python main.py --data-path iris-dvc-pipeline/v1_data.csv
```

### With Data Augmentation
```bash
python main.py --augment-data --data-path iris-dvc-pipeline/v1_data.csv
```

### Version-specific Execution
```bash
python main.py --version v2.0 --setup-dvc
```

### DVC Integration
```bash
python main.py --setup-dvc --data-path iris-dvc-pipeline/v1_data.csv
```

## 🔍 Monitoring & Reports

### CML Test Reports
Every CI run generates comprehensive test reports as GitHub comments, including:
- Test summary and status
- Coverage information
- Pipeline execution details
- Performance metrics

### Coverage Reports
- **HTML Coverage**: Available in CI artifacts
- **XML Coverage**: Uploaded to Codecov
- **Terminal Output**: Real-time coverage during test execution

## 🤝 Contributing

### Development Workflow
1. Create feature branch from `dev`
2. Make changes and add tests
3. Run tests locally: `pytest tests/ -v`
4. Push to `dev` branch
5. Create pull request to `main` branch
6. CI will automatically run tests and generate reports

### Code Quality
- **Pytest**: Comprehensive test coverage
- **Type Hints**: Full type annotation support
- **Logging**: Structured logging throughout
- **Error Handling**: Robust error handling and validation

## 📝 License

This project is part of the MLOps course assignment and follows academic guidelines.

## 👨‍💻 Author

**Vaibhav Gupta**  
Email: 21f2001529@ds.study.iitm.ac.in  
Course: MLOps - Week 2 Assignment

