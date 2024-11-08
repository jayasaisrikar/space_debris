# Space Debris Tracker & Predictor

This project is a machine learning-based space debris tracking and prediction tool. It involves data processing, analysis, visualization, and prediction of space debris behavior, such as orbital decay. The application is built using **Streamlit** for the frontend and leverages **scikit-learn** for predictive modeling. The application analyzes space debris datasets, predicts the time spent in orbit, and visualizes decay patterns using **Plotly**.

## Features

### 1. **Data Preprocessing**
   - The dataset includes details about space debris, such as:
     - Launch Date
     - Decay Date
     - Country Code
     - Object Type
     - Size (RCS Size)
   - Missing values are handled, and new features like "Time in Orbit" are derived by calculating the time between the launch and decay dates.
   - Categorical data such as RCS Size are mapped to numerical values to be used for machine learning models.

### 2. **Machine Learning Model**
   - A **pre-trained model** is used to predict certain outcomes related to space debris, such as time spent in orbit.
   - The model is loaded from a `space_debris_predictor.pkl` file.
   - The prediction is done using features such as country code, object type, and orbital parameters (eccentricity, inclination, etc.).

### 3. **Visualizations**
   - **Plotly** is used to create interactive graphs and visualizations.
   - The user can visualize the decay patterns, debris objects' time in orbit, and other relevant factors affecting space debris.
   
### 4. **Interactive Dashboard (Streamlit)**
   - An interactive dashboard built with **Streamlit** allows users to:
     - Upload space debris data (CSV format).
     - Run predictive models on the dataset.
     - Visualize various aspects of the data.
     - View the predicted "Time in Orbit" for debris objects.

## Dataset
The project uses a dataset of space debris, stored in a CSV file named `space_decay.csv`. This dataset contains essential features related to space objects and their orbital characteristics.

## Files

### 1. `app.py`
   - This file contains the Streamlit application code. It:
     - Loads the pre-trained model (`space_debris_predictor.pkl`).
     - Preprocesses the dataset (CSV format).
     - Generates visualizations for time in orbit and other debris-related statistics.
     - Allows users to interact with the data and view predictions.

### 2. `space-debris.ipynb`
   - This Jupyter notebook file contains data analysis and model development code. It includes:
     - Data preprocessing using pandas and NumPy.
     - Model training and evaluation using **scikit-learn**.
     - Detailed analysis and exploration of space debris data.
     - Visualization of various space debris characteristics using **matplotlib** and **seaborn**.
     
## Requirements

### Libraries
   - **Streamlit**: `streamlit`
   - **pandas**: `pandas`
   - **numpy**: `numpy`
   - **scikit-learn**: `scikit-learn`
   - **joblib**: `joblib`
   - **plotly**: `plotly`
   - **matplotlib**: `matplotlib`
   - **seaborn**: `seaborn`

### Installation
   1. Clone the repository:
      ```bash
      git clone <repository-url>
      ```
   2. Navigate to the project directory:
      ```bash
      cd space-debris-tracker
      ```
   3. Install the required libraries:
      ```bash
      pip install -r requirements.txt
      ```
   
### Running the Application
   - To run the Streamlit application:
     ```bash
     streamlit run app.py
     ```

   - The app will open in your browser, allowing you to upload a dataset and interact with the dashboard.

## Model
The predictive model used in this project is trained to estimate the time a space object spends in orbit based on various features. It uses scikit-learn's machine learning tools for model development and joblib for saving/loading the trained model.


## Author
This project was developed by Jaya Sai Srikar. 

Feel free to reach out for any questions or contributions!

---
