# Choreo2Tasks
Data Scientist Recruitment Tasks - Solutions

## 📋 Task Description
Solutions for the recruitment tasks described in [`recruitment_task.pdf`](recruitment_task.pdf)

## 🚀 Quick Start

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run Task 1: Reach Curve Modeling
cd task1_reach_curve/notebooks/
jupyter notebook 01_reach_curve_exploration.ipynb

# 3. Run Task 2: TensorFlow Optimization
python -m task2_tensorflow_optimization.src.optimization
```

## 📁 Project Structure

```
├── task1_reach_curve_modelling/
│   ├── notebooks/
|   |   └── 01_reach_curve_exploration.ipynb
│   └── data/
|       └── timespends.csv            
├── task2_tensorflow_optimization/
│   └── src/
|       └── main.py
|       └── funcs.py
└── requirements.txt
```

## 📊 Task 1: Reach Curve Modeling

**What it does:** Models reach curve based on timespends data.

**Key features:**
- Interactive Jupyter notebook with visualizations
- TODO: Propose any simple mathematical model
- TODO: Plan - read a paper [Reach Measurement, Optimization and Frequency Capping In Targeted Online Advertising Under k-Anonymity](https://arxiv.org/abs/2501.04882) - and think about formalization of problem and coding.

## 🔧 Task 2: TensorFlow CPU Optimization

**What it does:** Optimizes execution time of a function using TensorFlow on CPU.

**Key features:** 
TODO: try to find yet another methods to speed up computations beyond ```@tf.function``` and ```jitting```. 
