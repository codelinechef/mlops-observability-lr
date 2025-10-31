# Linear Regression Observability Project

This project is a demonstration of a linear regression model implemented using PyTorch, enriched with full observability features utilizing Prometheus for metrics, Galileo for AI logging, and Grafana for visualization. The solution is suitable for modern AI/ML workflow demonstrations, and is lightweight enough for small-scale or educational deployments.

---

### Features

- Linear regression using PyTorch
- Synthetic or CSV-based data loading
- Automated data preprocessing (scaling, one-hot encoding)
- Training with metrics logging (MSE, R2, training loss, CPU/GPU utilization)
- Metrics exposure via Prometheus on port 8000
- AI observability via Galileo integration (traces, spans, predictions)
- Optional hardware metrics tracking (CPU/GPU)
- Easy Grafana dashboard integration

---

## Installation

1. **Clone the repository**  
   ```
   git clone <your-repo-url>
   cd <your-project-directory>
   ```

2. **Install Python Dependencies**  
   Use the provided requirements.
   ```
   pip install -r requirement.txt
   ```
   Required packages: torch, numpy, scikit-learn, matplotlib, seaborn, prometheus-client, requests, python-dotenv, psutil

3. **Configure Environment Variables**  
   Create a `.env` file containing:  
   - `GALILEO_API_KEY`
   - `GALILEO_PROJECT`
   - `GALILEO_LOGSTREAM`
   - Other relevant keys for Galileo usage  
   Or set these in your shell environment.

4. **Set Up Prometheus**  
   Use the provided `prometheus.yml` for configuration.  
   Configure Prometheus server with the provided scrape job:
   ```
   global:
     scrape_interval: 5s
     evaluation_interval: 5s
   scrape_configs:
     - job_name: 'linearregressiontraining'
       metrics_path: /metrics
       static_configs:
         - targets: ['host.docker.internal:8000']
   ```
   Start Prometheus with:
   ```
   prometheus --config.file=prometheus.yml
   ```

5. **Optional: Set Up Grafana**  
   - Connect Grafana to Prometheus as a data source.
   - Import or create dashboards for metrics/observability.

---

## Usage

### 1. Train Linear Regression Model

- **Default Synthetic Dataset:** No CSV file required.
- **Custom Dataset:** Place `insurance.csv` (or similar) in the `Dataset` directory.
- **To train/run:**
  ```
  python LinearRegression.py
  ```
- **Model Parameters:** Configurable within the script (NSAMPLES, NFEATURES, NOISE, EPOCHS, LR, BATCHSIZE, etc).

### 2. Metrics Exposure

- Prometheus metrics are available at `http://localhost:8000/metrics`.
  - Exposed metrics: training loss, batch count, test MSE, R2 score, CPU %, GPU memory usage.

### 3. Observability with Galileo

- Ensure your `.env` contains a valid Galileo API key and project/logstream names.
- The scripts automatically create projects/log streams (if missing) and push traces, spans, and predictions for every run.

### 4. Monitoring Dashboard

- Use Grafana to visualize Prometheus metrics and, via Galileo, see AI trace logs and model predictions.
- Import relevant dashboards for linear regression model metrics or AI observability.

### 5. Smoke Test for Galileo

- Run the Galileo smoke test for tracing AI observability setup:
  ```
  python galileo_smoke_test.py
  ```
- This verifies the integration and trace appearance in your Galileo UI.

---

## Project Structure

| File                  | Purpose                                                                  |
|-----------------------|--------------------------------------------------------------------------|
| LinearRegression.py   | Main script for data processing, training, metrics, and observability    |
| galileo_smoke_test.py | Test script for Galileo integration and smoke testing                    |
| prometheus.yml        | Prometheus server scrape configuration                                   |
| requirement.txt       | Required Python packages                                                 |

---

## Monitoring Workflow

- **Prometheus** scrapes metrics from your model training run.
- **Grafana** visualizes these metrics for quick diagnostics.
- **Galileo** provides distributed traces, model input/output logging, and prediction monitoring for deep AI observability.

### Example Metrics Tracked

- Training Epoch, Training Loss, Batch Count
- Test MSE, Test R2 Score
- Hardware usage (CPU/GPU)
- Prediction Inputs/Outputs

---

## Troubleshooting

- If metrics do not appear, ensure the training script is running and Prometheus is scraping port 8000.
- For Galileo errors, confirm API key and project settings in `.env`.
- Grafana needs Prometheus as a data source configured correctly.

---

## References

- [Prometheus Python Monitoring Guide]
- [Galileo AI Observability Documentation]
- [Grafana Cloud Observability Setup]
- [PyTorch Linear Regression Implementation]

---

This README provides a clear, robust overview aimed at recruiters, engineers, and contributors interested in ML model monitoring and observability. For further details and visual dashboard templates, see the referenced documentation.
```

You can save this text content into a file named `README.md` in your project folder. If you want, I can prepare and upload the file here for you as well. Let me know!
