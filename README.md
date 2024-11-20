# Anomaly Detection Tool with Streamlit

This project is a Python-based application designed to analyze data in `.clsx` file format and provide anomaly detection using various techniques. The application runs locally with **Streamlit**, offering a user-friendly interface for exploring data and visualizing anomalies.

---

## Features

1. **Upload `.clsx` File:**  
   Upload your `.clsx` data file for processing.

2. **Data Viewing Options:**  
   Various modes for visualizing the dataset, including tabular and graphical representations.

3. **Anomaly Detection Techniques:**  
   Select from multiple anomaly detection algorithms:
   - **Threshold-based Detection (`ThresholdAD`)**  
     Detect anomalies exceeding pre-defined thresholds.
   - **Quantile-based Detection (`QuantileAD`)**  
     Identify observations significantly deviating from expected quantiles.
   - **Interquartile Range (`IQR-AD`)**  
     Anomaly detection based on the IQR formula.
   - **Z-score Analysis (`Z-scoreAD`)**  
     Detect anomalies using standard deviation-based thresholds.
   - **Persistence Analysis (`PersistAD`)**  
     Positive and negative persistence analysis using sliding windows.
   - **Local Outlier Factor (`LofAD`)**  
     Identify anomalies using density-based measures.
   - **One-Class SVM**  
     Machine learning-based anomaly detection.
   - **One-Class SVM with SGD (`OneClassSVM [SGD]`)**  
     Stochastic Gradient Descent implementation of One-Class SVM.

4. **Visualization Tools:**  
   Graphs to highlight detected anomalies.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/anomaly-detection-streamlit.git
   cd anomaly-detection-streamlit
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`.

---

## Usage

1. Start the application by running the Streamlit command.
2. Upload a `.clsx` file containing your dataset.
3. Choose a viewing mode to explore the data.
4. Select an anomaly detection technique from the dropdown menu.
5. View the anomalies detected using the chosen method.

---

## Anomaly Detection Methods Overview

### ThresholdAD
- **Usage:** Detect values outside the range `[low, high]`.
- **Example:**  
  ```python
  threshold_detector = ThresholdAD(low=300, high=900000000)
  anomalies = threshold_detector.detect(balance_amt)
  plot(anomalies)
  ```

### QuantileAD
- **Usage:** Identify significant deviations based on quantiles.
- **Example:**  
  ```python
  quantile_detector = QuantileAD(low=0.05, high=0.80)
  anomalies = quantile_detector.fit_detect(balance_amt)
  plot(anomalies)
  ```

### IQR-AD
- **Usage:** Anomaly detection using the IQR formula.
- **Example:**  
  ```python
  iqr_detector = InterQuartileRangeAD(c=1.5)
  anomalies = iqr_detector.fit_detect(balance_amt)
  plot(anomalies)
  ```

### Z-scoreAD
- **Usage:** Standard deviation-based detection.
- **Example:**  
  ```python
  zscore_detector = zskorad(balance_amt)
  anomalies = zscore_detector
  plot(anomalies)
  ```

### PersistAD
- **Positive Persistence:**
  ```python
  perad_detector1 = PersistAD(c=4.0, side="positive", window=5)
  anomalies = perad_detector1.fit_detect(balance_amt)
  plot(anomalies)
  ```
- **Negative Persistence:**
  ```python
  perad_detector2 = PersistAD(c=4.0, side="negative")
  anomalies = perad_detector2.fit_detect(balance_amt)
  plot(anomalies)
  ```

### LofAD
- **Usage:** Density-based anomaly detection.
- **Example:**  
  ```python
  lof_detector = lofad(balance_amt)
  anomalies = lof_detector
  plot(anomalies)
  ```

### One-Class SVM
- **Usage:** Anomaly detection with a machine learning model.
- **Example:**  
  ```python
  ocSVM_detector = ocSVM(balance_amt)
  anomalies = ocSVM_detector
  plot(anomalies)
  ```

### One-Class SVM with SGD
- **Usage:** Incremental training using SGD.
- **Example:**  
  ```python
  sgd_detector = svm_sgd(balance_amt, acc_no)
  anomalies = sgd_detector
  plot(anomalies)
  ```

---

## Dependencies

- Python 3.8+
- Streamlit
- pandas
- matplotlib
- scikit-learn
- pyod

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## Contact

For issues or suggestions, contact [your_email@example.com](mailto:your_email@example.com).
```

Replace placeholders like `https://github.com/your-repo/anomaly-detection-streamlit.git` and `your_email@example.com` with actual details.
