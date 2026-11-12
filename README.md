# Crop Yield Prediction Using Machine Learning

## Project Overview
This project predicts crop yield using Machine Learning regression algorithms. It uses agricultural data from different Indian states, including crop type, year, season, area, rainfall, fertilizer and pesticide usage.

## Dataset
- Total Records: 19,689
- Total Features: 10
- Target Variable: Yield
- Dataset: [Crop Yield in Indian States – Kaggle](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset)

## Algorithms Used
1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. K-Nearest Neighbors (KNN) Regression
5. Random Forest Regression

## Model Performance

| Algorithm | R² Score |
|---|---:|
| Simple Linear Regression | 0.0004 |
| Multiple Linear Regression | 0.8105 |
| Polynomial Regression | 0.0040 |
| KNN Regression | 0.9874 |
| Random Forest Regression | 0.9801 |

**Best Performing Model:** KNN Regression

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Files
- `Crop_Yield_Prediction.py` – Python source code
- `crop_yield.csv.zip` – Dataset
- `model_comparison.csv` – Model performance results

## How to Run
1. Download or clone this repository.
2. Extract the dataset ZIP file.
3. Install the required Python libraries.
4. Run the Python file.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Crop_Yield_Prediction.py
```

## Conclusion
Five regression algorithms were implemented and evaluated. KNN Regression achieved the highest R² score of 0.9874 among the tested models.

## Author
Mauli Mahiya  
B.Tech Computer Science and Engineering
