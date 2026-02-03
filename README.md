# Cloud Resource Management System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**An AI-powered cloud resource management system that optimizes power consumption through intelligent workload prediction and dynamic resource scaling.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🎯 Overview

The **Cloud Resource Management System** is an intelligent solution designed to optimize cloud resource allocation and minimize power consumption in cloud computing environments. By leveraging machine learning algorithms, the system predicts power consumption based on resource utilization metrics and automatically scales resources to maintain optimal performance while reducing energy costs.

This project addresses the growing need for energy-efficient cloud infrastructure by:
- **Predicting** power consumption using AI models
- **Optimizing** resource allocation based on workload patterns
- **Automating** scaling decisions to reduce waste
- **Monitoring** real-time system performance

---

## ✨ Key Features

### 🤖 AI-Powered Prediction
- **Gradient Boosting Regressor** for accurate power consumption forecasting
- **Hyperparameter tuning** using GridSearchCV for optimal model performance
- **Cross-validation** to ensure model reliability

### 📊 Intelligent Data Processing
- Automated data cleaning and preprocessing pipeline
- Feature engineering with one-hot encoding for categorical workload types
- Anomaly detection and removal for cleaner training data
- Synthetic power label generation based on resource utilization

### ⚡ Dynamic Resource Scaling
- Real-time resource monitoring and adjustment
- Threshold-based scaling decisions (scale up/down/hold)
- Simulated VM management for testing scenarios
- Power-aware scaling algorithms

### 📈 Comprehensive Visualization
- Power consumption trends over time
- Active VM tracking and visualization
- Performance metrics and model evaluation
- Automated report generation

---

## 🏗️ System Architecture

The system follows a modular architecture with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preprocessing Layer                  │
│  • CSV Data Loading    • Feature Engineering                │
│  • Data Cleaning       • Power Label Generation             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Machine Learning Layer                     │
│  • Model Training      • Hyperparameter Tuning              │
│  • Cross Validation    • Performance Evaluation             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Resource Management Layer                     │
│  • Power Prediction    • Scaling Decisions                  │
│  • VM Management       • Real-time Monitoring               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Visualization & Reporting                   │
│  • Performance Graphs  • Metrics Dashboard                  │
│  • Simulation Logs     • Result Export                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms

### Machine Learning
- **Gradient Boosting Regressor** - Power prediction model
- **GridSearchCV** - Hyperparameter optimization
- **Cross-validation** - Model validation

### Visualization
- **Matplotlib** - Graph generation and plotting

### Development
- **Git** - Version control
- **MIT License** - Open source licensing

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/GitRijowan/cloud_resource_management.git
cd cloud_resource_management
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

Or create a requirements file and install:

```bash
# Create requirements.txt with:
# pandas>=1.3.0
# numpy>=1.21.0
# scikit-learn>=0.24.0
# matplotlib>=3.4.0

pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib; print('All dependencies installed successfully!')"
```

---

## 🚀 Usage

### Quick Start

1. **Ensure Dataset is Available**
   
   The system expects a CSV file at `data/cloud_dataset.csv`. The dataset should contain the following columns:
   - `Timestamp` - Time of measurement
   - `CPU_Usage` - CPU utilization percentage
   - `Memory_Usage` - Memory utilization percentage
   - `Disk_IO` - Disk I/O operations
   - `Network_IO` - Network I/O operations
   - `Workload_Type` - Type of workload (Web_Service, Database_Query, etc.)
   - `User_ID` - User identifier
   - `Anomaly_Label` - Binary flag for anomalous behavior

2. **Run Data Preprocessing (Optional)**

   ```bash
   python src/data_preprocessing.py
   ```

   This will:
   - Clean the raw dataset
   - Remove anomalies
   - Generate power consumption labels
   - Save cleaned data to `results/cleaned_cloud_data.csv`

3. **Run the Complete System**

   ```bash
   python src/power_system.py
   ```

   This will execute the full pipeline:
   - Load and preprocess data
   - Train the AI model
   - Run resource scaling simulation
   - Generate visualization
   - Save results to the `results/` folder

### Configuration

Edit `src/config.py` to customize system parameters:

```python
# System Simulation Constants
BASE_POWER_WATTS = 100.0      # Base power consumption
CPU_POWER_COEFF = 50.0        # CPU power coefficient
IDLE_THRESHOLD = 0.2          # Threshold for scaling down
HIGH_THRESHOLD = 0.8          # Threshold for scaling up
```

### Advanced Usage

**Custom Dataset Path:**
```python
# In config.py
DATASET_PATH = os.path.join(DATA_DIR, 'your_custom_dataset.csv')
```

**Model Customization:**
```python
# In power_system.py, modify param_grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.8, 0.9, 1.0]
}
```

---

## 📊 Dataset

### Dataset Structure

The project uses a cloud resource utilization dataset with the following characteristics:

| Column | Description | Type | Range |
|--------|-------------|------|-------|
| Timestamp | Measurement time | DateTime | - |
| CPU_Usage | CPU utilization | Float | 0-100% |
| Memory_Usage | Memory utilization | Float | 0-100% |
| Disk_IO | Disk I/O operations | Float | 0-100 |
| Network_IO | Network I/O operations | Float | 0-100 |
| Workload_Type | Type of workload | Categorical | Web_Service, Database_Query, Video_Streaming, Backup, Crypto_Mining |
| User_ID | User identifier | String | - |
| Anomaly_Label | Anomaly flag | Binary | 0 (normal), 1 (anomaly) |

### Workload Types

The system handles multiple workload types:
- **Web_Service** - Standard web application requests
- **Database_Query** - Database operations
- **Video_Streaming** - Media streaming workloads
- **Backup** - Data backup operations
- **Crypto_Mining** - Cryptocurrency mining (flagged as anomaly)

### Data Preprocessing

The preprocessing pipeline includes:
1. **Missing Value Handling** - Drops rows with missing critical features
2. **Anomaly Removal** - Filters out anomalous records for cleaner training
3. **Feature Engineering** - One-hot encoding for workload types
4. **Power Label Generation** - Synthetic power consumption calculation:
   ```
   Power = Base + (Total_Load × Coefficient) + Noise
   ```

---

## 📈 Results

### Model Performance

The trained model achieves:
- **Low Mean Squared Error (MSE)** - Accurate power predictions
- **High R² Score** - Strong correlation between predicted and actual values
- **Cross-validated Results** - Consistent performance across folds

### Visualizations

The system generates comprehensive visualizations saved in `results/`:

1. **Power Consumption Trends**
   - Time-series plot of predicted power consumption
   - Identifies peak usage periods
   
2. **Active VM Count**
   - Dynamic scaling visualization
   - Shows resource optimization in action

3. **Performance Graphs**
   - Model accuracy metrics
   - Prediction vs actual comparisons

### Sample Output

```
[-] Step 1: Loading Preprocessed CSV Data...
[+] Cleaned Data Loaded. Shape: (1000, 15)
[-] Step 2: Training AI Power Prediction Model...
[+] Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
[+] Model Training Complete.
    - Mean Squared Error: 12.3456
    - R^2 Score: 0.9567
[-] Step 3 & 4: Running Dynamic Resource Scaling Simulation...
[+] Simulation Complete.
[+] Graph saved to: results/power_management_results.png
```

---

## 📁 Project Structure

```
cloud_resource_management/
│
├── src/
│   ├── config.py              # System configuration and constants
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   └── power_system.py        # Main system implementation
│
├── data/
│   └── cloud_dataset.csv      # Raw cloud resource dataset
│
├── results/
│   ├── cleaned_cloud_data.csv         # Preprocessed dataset
│   └── power_management_results.png   # Visualization graphs
│
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/GitRijowan/cloud_resource_management.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow Python PEP 8 style guidelines
   - Add comments for complex logic
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   python src/power_system.py
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues

### Areas for Contribution

- 🔧 Additional machine learning models (LSTM, Random Forest, etc.)
- 📊 More visualization options and dashboards
- 🌐 Real cloud provider integration (AWS, Azure, GCP)
- 🧪 Unit tests and integration tests
- 📚 Enhanced documentation and tutorials
- 🎨 Web-based user interface

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 KHAN MD RIJOWAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

**KHAN MD RIJOWAN**

- GitHub: [@GitRijowan](https://github.com/GitRijowan)
- Project: [cloud_resource_management](https://github.com/GitRijowan/cloud_resource_management)

---

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing tools and libraries
- Inspired by the need for energy-efficient cloud computing
- Built with passion for sustainable technology solutions

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/GitRijowan/cloud_resource_management/issues) page
2. Create a new issue with a detailed description
3. Provide sample data and error messages if applicable

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [KHAN MD RIJOWAN](https://github.com/GitRijowan)

</div>