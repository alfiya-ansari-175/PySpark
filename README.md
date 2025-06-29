
# 📊 Customer Segmentation using Bank Dataset (PySpark Project)

This project focuses on customer segmentation using the Bank Churners dataset. It uses PySpark for scalable data processing, feature engineering, and clustering (KMeans) to understand customer behavior and predict churn.

## 📁 Dataset

- **Source:** [Bank Churners dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Format:** CSV
- **Key Columns:** Customer_Age, Total_Trans_Ct, Total_Trans_Amt, Attrition_Flag, etc.

## 🚀 Project Goals

- Preprocess and clean the dataset using PySpark.
- Perform RFM (Recency, Frequency, Monetary) analysis.
- Use KMeans clustering to group similar customers.
- Interpret clusters with respect to churn.

## 🛠️ Technologies Used

- Apache Spark (PySpark)
- PySpark MLlib (Clustering, Feature Engineering)
- Google Colab / Jupyter Notebook

## 📦 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/customer-segmentation-pyspark.git
   cd customer-segmentation-pyspark
   ```

2. Install PySpark:
   ```bash
   pip install pyspark
   ```

3. Run the notebook using Jupyter or Google Colab.

## 🔄 Workflow Summary

1. **Data Loading & Cleaning**
   - Read dataset using `SparkSession`
   - Dropped irrelevant classifier columns
   - Created a new binary column `churn` (1 for churned, 0 for existing customers)

2. **RFM Feature Engineering**
   - Discretized:
     - `Months_on_book` (Recency)
     - `Total_Trans_Ct` (Frequency)
     - `Total_Trans_Amt` (Monetary)
   - Used `QuantileDiscretizer` to split each into 5 bins

3. **Feature Vectorization**
   - Used `VectorAssembler` and `StandardScaler` for clustering

4. **KMeans Clustering**
   - Ran KMeans for `k=2` to `k=10`
   - Selected the best `k` based on **Silhouette Score**
   - Clustered the data and evaluated churn distribution

## 📈 Main Insights

- **Churn Feature Creation:** Attrited customers were successfully encoded as 1, helping in later segmentation.

- **RFM Segmentation:** Customers were divided based on:
  - Time with the bank (`Months_on_book`)
  - Transaction count (`Total_Trans_Ct`)
  - Total transaction amount (`Total_Trans_Amt`)

- **Best K in Clustering:** The optimal number of clusters (`k`) was selected using **Silhouette Score**, achieving the highest cohesion and separation for the chosen `k`.

- **Cluster-Level Churn Analysis:**
  - Some clusters had significantly higher churn than others.
  - High transaction frequency and value were associated with **lower churn**.
  - Clusters with low engagement (low frequency/value) showed **high churn**, which can help in targeting retention efforts.

## 📌 Future Improvements

- Add visualizations using matplotlib or seaborn (via `.toPandas()`).
- Implement additional clustering methods (e.g., DBSCAN).
- Use advanced models for churn prediction (e.g., logistic regression, random forest).

## 👤 Author

**Alfiya Ansari**  
Master’s in Web and Data Science – University of Koblenz  
GitHub: [your-username](https://github.com/your-username)
