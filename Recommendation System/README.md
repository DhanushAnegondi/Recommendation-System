# MovieLens Recommendation System😁

This project implements a recommendation system using the MovieLens dataset. The system includes multiple recommendation models such as matrix factorization (ALS), neural collaborative filtering, and XGBoost. Additionally, a hybrid recommendation system combining collaborative filtering and content-based filtering is implemented. The project includes exploratory data analysis (EDA), feature engineering, hyperparameter tuning, cross-validation, model evaluation, and visualizations.

## Table of Contents

- [MovieLens Recommendation System😁](#movielens-recommendation-system)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Models Implemented](#models-implemented)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Visualizations](#visualizations)
  - [Future Work](#future-work)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

The goal of this project is to build a recommendation system that predicts user ratings for movies. The system uses collaborative filtering, content-based filtering, and hybrid methods to provide recommendations. The project also includes extensive exploratory data analysis and visualizations to understand the data better.

## Dataset

The [MovieLens dataset](https://grouplens.org/datasets/movielens/latest/) is used in this project. It consists of millions of ratings and tags applied to movies by users. The dataset is available in various sizes; this project uses the latest full version.

## Installation

To run this project, you need to have Python and the following libraries installed:

- Pandas
- NumPy
- PySpark
- TensorFlow
- XGBoost
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly

You can install the required libraries using pip:

```bash
pip install pandas numpy pyspark tensorflow xgboost scikit-learn matplotlib seaborn plotly
```

## Usage

1. **Download the dataset:**

    ```bash
    !wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
    !unzip ml-latest.zip
    ```

2. **Run the Jupyter Notebook or Python script:**

    Ensure you have all the required libraries installed and then run the notebook or script provided in the repository.

3. **Model Training and Evaluation:**

    The notebook/script includes sections for data loading, EDA, feature engineering, model training, hybrid recommendations, evaluation, and visualization.

## Models Implemented

1. **Matrix Factorization (ALS):**

    Trains a matrix factorization model using Alternating Least Squares (ALS).

2. **Neural Collaborative Filtering:**

    Uses neural networks to learn user and movie embeddings and predict ratings.

3. **XGBoost:**

    Trains a gradient boosting model to predict ratings based on user and movie features.

## Evaluation

Models are evaluated using Root Mean Squared Error (RMSE). Cross-validation and hyperparameter tuning are performed for model optimization.

## Results

The RMSE scores for different models are compared. The hybrid recommendation system shows improved performance by combining the strengths of collaborative and content-based methods.

## Visualizations

The project includes various visualizations for data exploration and model evaluation:

- Distribution of ratings
- Number of ratings per movie
- Number of ratings per user
- Top-rated movies
- Loss curves for neural collaborative filtering model
- Model comparison by RMSE

## Future Work

- Implement additional models such as LightFM.
- Explore other hybrid methods and ensemble techniques.
- Optimize hyperparameters further using advanced techniques like Bayesian Optimization.
- Deploy the recommendation system using a web framework like Flask or Django.

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first. You can contribute by:

- Reporting bugs and issues
- Submitting feature requests
- Creating pull requests with improvements

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.