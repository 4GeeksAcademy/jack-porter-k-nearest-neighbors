# K-Nearest Neighbors: Wine Quality Classification

[![Codespaces Prebuilds](https://github.com/4GeeksAcademy/gperdrizet-k-nearest-neighbors/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/4GeeksAcademy/gperdrizet-k-nearest-neighbors/actions/workflows/codespaces/create_codespaces_prebuilds)

A comprehensive machine learning project focused on wine quality classification using K-Nearest Neighbors (KNN) algorithms. This project demonstrates essential machine learning techniques including exploratory data analysis, feature engineering, hyperparameter optimization, and class imbalance handling through practical exercises with real-world wine quality data.


## Project Overview

This project analyzes wine quality data from the **Wine Quality Dataset**, containing physicochemical properties and sensory evaluations of red wines. The dataset provides hands-on experience with:

- Data loading and exploratory data analysis (EDA)
- Feature scaling and preprocessing
- K-Nearest Neighbors algorithm implementation
- Hyperparameter optimization with GridSearchCV
- Class imbalance handling using ADASYN oversampling
- Model comparison and performance evaluation
- Confusion matrix analysis and visualization


## Getting Started

### Option 1: GitHub Codespaces (Recommended)

1. **Fork the Repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - 4Geeks students: set 4GeeksAcademy as the owner - 4Geeks pays for your codespace usage. All others, set yourself as the owner
   - Give the fork a descriptive name. 4Geeks students: I recommend including your GitHub username to help in finding the fork if you lose the link
   - Click "Create fork"
   - 4Geeks students: bookmark or otherwise save the link to your fork

2. **Create a GitHub Codespace**
   - On your forked repository, click the "Code" button
   - Select "Create codespace on main"
   - If the "Create codespace on main" option is grayed out - go to your codespaces list from the three-bar menu at the upper left and delete an old codespace
   - Wait for the environment to load (dependencies are pre-installed)

3. **Start Working**
   - Open `notebooks/mvp.ipynb` in the Jupyter interface for the assignment
   - Open `notebooks/solution.ipynb` to see the complete solution
   - Follow the step-by-step instructions in the notebooks

### Option 2: Local Development

1. **Prerequisites**
   - Git
   - Python >= 3.10

2. **Fork the repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - Optional: give the fork a new name and/or description
   - Click "Create fork"

3. **Clone the repository**
   - From your fork of the repository, click the green "Code" button at the upper right
   - From the "Local" tab, select HTTPS and copy the link
   - Run the following commands on your machine, replacing `<LINK>` and `<REPO_NAME>`

   ```bash
   git clone <LINK>
   cd <REPO_NAME>
   ```

4. **Set Up Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Launch Jupyter & start the notebook**
   ```bash
   jupyter notebook notebooks/mvp.ipynb
   ```


## Project Structure

```
├── .devcontainer/        # Development container configuration
├── data/                 # Data file directory
├── models/               # Directory for trained models (if applicable)
│
├── notebooks/            # Jupyter notebook directory
│   ├── functions.py      # Helper functions for analysis
│   ├── mvp.ipynb         # Assignment notebook (starter template)
│   └── solution.ipynb    # Complete solution notebook
│
├── .gitignore            # Files/directories not tracked by git
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Dataset

This project uses the **Wine Quality Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). The dataset contains red wine quality data with the following key features:

### Physicochemical Properties:
- **Fixed acidity**: Tartaric acid concentration
- **Volatile acidity**: Acetic acid concentration  
- **Citric acid**: Citric acid concentration
- **Residual sugar**: Sugar remaining after fermentation
- **Chlorides**: Salt concentration
- **Free sulfur dioxide**: Free SO2 concentration
- **Total sulfur dioxide**: Total SO2 concentration
- **Density**: Wine density
- **pH**: Acidity/alkalinity measure
- **Sulphates**: Potassium sulphate concentration
- **Alcohol**: Alcohol percentage

### Target Variable:
- **Quality**: Wine quality score (3-8) based on sensory evaluation

**Note**: The original dataset contains quality scores from 3-8, which are binned into three categories for classification:
- Low quality: ≤ 4
- Medium quality: 5-6  
- High quality: ≥ 7


## Learning Objectives

1. **Data Exploration**: Load and examine the wine quality dataset structure
2. **Exploratory Data Analysis**: Analyze feature distributions and correlations
3. **Data Preprocessing**: Handle class imbalance and scale features appropriately  
4. **Feature Engineering**: Transform continuous quality scores into discrete categories
5. **KNN Implementation**: Build and evaluate K-Nearest Neighbors classifiers
6. **Hyperparameter Optimization**: Use GridSearchCV for optimal model parameters
7. **Advanced Techniques**: Apply ADASYN oversampling for class imbalance
8. **Model Comparison**: Compare performance across multiple model variants
9. **Performance Evaluation**: Analyze results using confusion matrices and accuracy metrics


## Technologies Used

- **Python 3.11**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and tools
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical data visualization
- **Imbalanced-learn**: ADASYN oversampling for class imbalance
- **Jupyter**: Interactive development environment

## Key Machine Learning Concepts

- **K-Nearest Neighbors (KNN)**: Distance-based classification algorithm
- **Hyperparameter Optimization**: Systematic parameter tuning using GridSearchCV
- **Cross-Validation**: Robust model evaluation technique
- **Class Imbalance**: Handling uneven target class distributions
- **ADASYN Oversampling**: Adaptive synthetic sampling for minority classes
- **Feature Scaling**: Normalization for distance-based algorithms
- **Model Comparison**: Performance evaluation across multiple approaches

## Contributing

This is an educational project. Contributions for improving the analysis or adding new insights are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
