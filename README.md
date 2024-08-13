# Flask Machine Learning App

This project is a web application built using Flask, which allows users to make predictions using pre-trained machine learning models. The application supports both classification and regression tasks, leveraging Random Forest and Lasso Regression models, respectively.

## Features

- **Classification:** Predicts whether a loan will default based on various input features using a Random Forest Classifier.
- **Regression:** Predicts the amount of some financial variable (e.g., loan amount, interest amount) based on input features using a Lasso Regression model.
- **Web Interface:** A user-friendly interface for inputting data and viewing predictions.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.x
- Flask
- joblib
- pandas

You can install the required Python packages using the following command:

```bash
pip install Flask joblib pandas
```

## Project Structure

- `app.py`: The main Flask application file.
- `random_forest_classifier_pipeline.joblib`: The pre-trained Random Forest Classifier model.
- `lasso_regression_pipeline.joblib`: The pre-trained Lasso Regression model.
- `templates/`: Directory containing HTML templates (`home.html`, `classification.html`, `regression.html`).

## Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/flask-ml-app.git
   cd flask-ml-app
   ```

2. Ensure the pre-trained models (`random_forest_classifier_pipeline.joblib` and `lasso_regression_pipeline.joblib`) are in the project directory.

3. Run the Flask application:

   ```bash
   python app.py
   ```

4. Open your web browser and go to `http://127.0.0.1:5000/` to access the web interface.

## Usage

### Home Page

On the home page, you can select the type of prediction task:

- **Classification**: Directs you to a page where you can input features related to loan default prediction.
- **Regression**: Directs you to a page where you can input features related to financial predictions.

### Classification Page

Input the required features and submit the form to get a prediction on whether a loan will default.

### Regression Page

Input the required features and submit the form to get a financial prediction.

## Error Handling

The application is designed to handle errors gracefully, displaying appropriate error messages if any issues occur during the prediction process.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The machine learning models were trained using appropriate datasets and libraries.
- Flask was used to develop the web application framework.

---

This README file provides an overview of the project, including installation instructions, usage details, and acknowledgments. Make sure to adjust any specific details according to your actual project setup.
