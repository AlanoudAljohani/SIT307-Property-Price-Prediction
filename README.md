# SIT307 Property Price Prediction

This project was developed for the SIT307 8.1D task. It uses machine learning to predict property sale prices in Liverpool, Burwood and Chatswood.

## Project Files

* `SIT307_8.1D.ipynb` – data analysis, feature engineering and model evaluation
* `syd.xlsx` – collected housing dataset
* `app.py` – Streamlit web application
* `property_price_model.joblib` – trained Linear Regression pipeline
* `requirements.txt` – required Python packages
* `app_interface.png` and `app_prediction.png` – application screenshots

## Model

Three regression models were compared using five-fold cross-validation:

* Linear Regression
* Random Forest
* Gradient Boosting

Linear Regression was selected as the final model because it achieved the best overall validation RMSE and R² and showed the most stable performance.

## Running the Application

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

Start the Streamlit application:

```bash
python3 -m streamlit run app.py
```

The application will open in a web browser at:

```text
http://localhost:8501
```

Enter the property information and select **Predict Sale Price** to receive an estimated price in Australian dollars.

## Important Note

The prediction is an estimate based on a small dataset and should not replace a professional property valuation.
