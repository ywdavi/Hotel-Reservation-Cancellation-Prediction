# Hotel Reservation Cancellation Prediction

## Description
This project was completed in 2022 as part of my early exploration of machine learning techniques using RStudio. The goal of the project is to build a classification model capable of predicting whether hotel reservations will be canceled. We use the **Hotel Reservations Dataset**, which contains 36,238 reservations from a major hotel chain in the USA. The project includes data preprocessing, feature engineering, and various classification models to make accurate predictions.

## Objectives
- Predict which hotel reservations are likely to be canceled.
- Explore factors affecting reservation cancellations, such as the number of special requests and booking lead time.
- Evaluate multiple classification models, focusing on sensitivity and ROC for optimal performance.

## Key Steps

### Data Preprocessing and Feature Engineering:
- Generated additional features, such as `arrival_weekday`, from existing date variables to improve predictions.
- Removed unnecessary features like `Booking_ID` and handled categorical variables appropriately.
- Split the dataset into training, validation, and scoring sets, ensuring balanced distribution of the target variable.

### Model Selection:
- Applied **Random Forest** and **Boruta** algorithms to select relevant features, removing less important ones like `no_of_previous_cancellations`.
- Evaluated models based on **ROC** curve performance and sensitivity, focusing on correctly identifying canceled bookings.

### Classification Models:
- Tested various models, including **Generalized Linear Model**, **Lasso Regression**, **Neural Networks**, **Gradient Boosting**, and **Bagging Trees**.
- **Bagging Trees** was chosen as the final model due to its superior performance, capturing 64% of cancellations in the top 20% of high-risk predictions.

## Results
- The final **Bagging Trees** model performed well, with sensitivity set at 0.35, balancing true positive and false positive rates.
- Important variables affecting cancellation predictions include `lead_time`, `no_of_special_requests`, and `avg_price_per_room`.
- The model was applied to the scoring dataset, providing insights into which reservations were likely to be canceled, with a validation accuracy of 64% for high-risk bookings.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hotel-reservation-cancellation.git
