# Rental Property Prices Prediction

## Introduction

In the dynamic realm of real estate, accurately gauging rental prices is pivotal for property stakeholders. This project addresses the needs of property owners, tenants, and management entities by harnessing data-driven insights. Our goal is to deploy a predictive model that leverages historical rental data and property attributes, facilitating informed decision-making for landlords, empowering tenants with comprehensive rental insights, and aiding property management companies in optimizing portfolio strategies. Explore the power of precise rental predictions with this intuitive and reliable model.

## Table of Contents

1. Key Technologies and Skills
2. Installation
3. Usage
4. Features
5. Contributing
6. License
7. Contact


## Key Technologies and Skills
- Python
- Numpy
- Pandas
- PostgreSQL
- Scikit-Learn
- Plotly
- Matplotlib
- Seaborn
- Streamlit


## Installation

To run this project, you need to install the following packages:

```python
pip install numpy
pip install pandas
pip install psycopg2
pip install scikit-learn==1.3.2
pip install xgboost
pip install plotly
pip install matplotlib
pip install seaborn
pip install streamlit
```

## Usage

To use this project, follow these steps:

1. Clone the repository: ```git clone https://github.com/gopiashokan/Rental-Property-Prices-Prediction.git```
2. Install the required packages: ```pip install -r requirements.txt```
3. Run the Streamlit app: ```streamlit run app.py```
4. Access the app in your browser at ```http://localhost:8501```


## Features

**Data Preprocessing:**

- **Data Understanding**: This dataset encapsulates essential property details, featuring unique IDs, property types, localities, and activation dates. Geospatial coordinates provide precise locations, while lease details and amenities offer insights into tenant preferences. Property characteristics include size, age, and floor details, with rental prices as the key target variable for predictive modeling.

- **Encoding and Data Type Conversion**: The process involves preparing categorical features for modeling by transforming them into numerical representations, considering their inherent nature and relationship with the target variable. Simultaneously, data types are converted to align with the modeling process requirements, ensuring seamless integration and compatibility. This step facilitates the effective utilization of categorical information in the subsequent stages of the project.

- **Handling Null Values**: To address missing values in the dataset, a systematic approach was adopted where most features with null values were replaced with zeros. This strategic substitution ensures uniformity in the dataset, facilitating a more seamless analysis and modeling process. The decision to use zero as a placeholder aligns with maintaining data integrity while mitigating the impact of missing information on subsequent analyses.

- **Feature Improvement**: Prioritizing modeling effectiveness, the dataset is refined through the creation of new features to extract deeper insights and enhance overall efficiency. An evaluation using `Seaborn's Heatmap` highlights key features positively correlated with rent, including type, property size, floor, balconies, bathroom, parking, and amenities count. Conversely, building type demonstrates a negative correlation with rent, providing valuable insights into the significance of each feature in predicting rental prices.


**Machine Learning Regression Model**:

- **Algorithm Assessment**: In the realm of regression, our primary objective is to predict the continuous variable of rent. Our journey begins by splitting the dataset into training and testing subsets. We systematically apply various algorithms, evaluating them based on training and testing accuracy using the R2 (R-squared) metric, which signifies the coefficient of determination. This process allows us to identify the most suitable base algorithm tailored to our specific data.

- **Algorithm Selection**: After thorough evaluation, three contenders, the Random Forest Regressor, Gradient Boosting Regressor and XGB Regressor, demonstrate commendable testing accuracy. Upon checking for any overfitting issues in both training and testing, both models exhibit strong performance without overfitting concerns. I choose the Random Forest Regressor for its ability to strike a balance between interpretability and accuracy, ensuring robust performance on unseen data.

- **Hyperparameter Tuning**: Optimizing model performance is paramount, and we achieve this through a meticulous process of hyperparameter tuning. Leveraging grid search and cross-validation techniques, we systematically explore the hyperparameter space to identify the most effective configuration. This rigorous approach ensures that our model is finely tuned, delivering optimal predictive accuracy and robust generalization across diverse datasets.

- **Model Accuracy and Metrics**: Assessing the performance of our predictive model involves a comprehensive examination of regression metrics. Key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R-squared (R²) are employed to quantify the accuracy and precision of the model's predictions. This holistic evaluation provides valuable insights into the model's ability to effectively capture and predict rental property prices, ensuring a nuanced understanding of its overall performance.

- **Model Persistence**: We conclude this phase by saving our well-trained model to a pickle file. This strategic move enables us to effortlessly load the model whenever needed, streamlining the process of making predictions on rent in future applications.


**Exploratory Data Analysis (EDA) - Streamlit Application:**

**Migrating to SQL**: Following the rent predictions, the data is migrated to a PostgreSQL database. Leveraging PostgreSQL as a Relational Database Management System allows us to store structured data in a tabular format. Utilizing SQL queries, we seamlessly access the required data from the database, facilitating efficient retrieval and analysis of pertinent information.

**Property Type (BHK) Analysis**: Analyzing the average rent based on property types (BHK) provides valuable insights into the rental landscape. This breakdown allows us to understand how different configurations influence rental pricing, offering nuanced perspectives for both property owners and tenants.

**Lease Type and Property Characteristics**: Examining average rents in relation to lease types, property size, and age facilitates a comprehensive understanding of the market. This analysis sheds light on the varying dynamics of lease preferences, property dimensions, and age factors, aiding in strategic decision-making.

**Amenities and Property Features**: Diving into average rents based on amenities count and specific amenities (e.g., gym, lift, swimming pool, internet, AC) provides a detailed view of the impact these features have on rental prices. Understanding tenant preferences for such amenities can inform property management strategies and highlight potential areas for improvement or investment.

**Location and Structural Impact**: Exploring average rents concerning location-specific and structural factors such as parking, facing direction, floor arrangement, total floor count, balconies, and negotiability unravels how these elements contribute to the rental market landscape. This examination aids property stakeholders in making informed decisions in a dynamic real estate environment.

**Prediction**: Users provide input parameters and leveraging the pre-trained Random Forest Regressor model stored in a pickle file, the system predicts weekly sales based on the user's input. Users can experiment with various combinations of input values, allowing our machine learning regression model to dynamically forecast rent tailored to the provided data parameters. This interactive feature enhances user engagement and facilitates personalized predictions.


## Contributing

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.


## License

This project is licensed under the MIT License. Please review the LICENSE file for more details.


## Contact

📧 Email: gopiashokankiot@gmail.com 

🌐 LinkedIn: [linkedin.com/in/gopiashokan](https://www.linkedin.com/in/gopiashokan)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.
