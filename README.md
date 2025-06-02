
# Real-Time Loan Application Processing

This project is a real-time machine learning pipeline that processes incoming loan applications and classifies them as **approved** or **rejected**. It uses **Apache Kafka** for streaming, **scikit-learn** for model training, and **AWS S3** for storing trained models. The entire workflow is containerized using Docker and supports on-demand retraining and cloud integration.


## Project Features

- Real-time data ingestion and prediction using **Kafka Producer/Consumer**
- Machine learning model for loan approval classification
- Retraining pipeline for model improvement with new data
- **Exploratory Data Analysis (EDA)** and **feature engineering**
- **Dockerized deployment** for easy local setup
- **Model persistence to AWS S3**

---

## Project Structure

```

.
├── Data/                        # Raw and preprocessed loan data
├── Exploration/                # Notebooks/scripts for EDA and feature analysis
├── FeatureImportances/        # Feature importance visuals from trained model
├── Models/                     # Trained model files
├── test/                       # Scripts for testing pipeline components
├── Kafka\_Producer.py           # Streams loan applications into Kafka
├── Kafka\_Consumer.py           # Consumes messages and performs predictions
├── main.py                     # Orchestrates the full pipeline
├── retraining.py               # Retrains model with new data
├── upload\_model\_to\_s3.py       # Uploads trained model to AWS S3
├── docker-compose.yaml         # Defines Docker services (Kafka, Zookeeper, etc.)
├── README.md                   # Project documentation

````
## 🖼 Architecture Diagram

![Architecture Diagram](./assets/architecture.png)


---

## Technology Stack

- **Programming Language:** Python
- **Streaming:** Apache Kafka
- **Machine Learning:** scikit-learn
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib
- **Cloud Integration:** AWS S3
- **Containerization:** Docker, Docker Compose

---

## How to Run the Project Locally

### Prerequisites

- Docker and Docker Compose installed
- Python 3.8+
- AWS CLI configured (for uploading to S3)

### Step-by-Step

```bash
# Step 1: Clone the repo
git clone https://github.com/khatgarhaastha/Real-Time-Loan-Application-Processing.git
cd Real-Time-Loan-Application-Processing

# Step 2: Start Kafka and Zookeeper
docker-compose up -d

# Step 3: Start the Kafka producer (in one terminal)
python Kafka_Producer.py

# Step 4: Start the Kafka consumer (in another terminal)
python Kafka_Consumer.py
````

---

## Model Training & Retraining

To train or retrain the model:

```bash
python retraining.py
```

To upload the model to AWS S3:

```bash
python upload_model_to_s3.py
```

---

## Example Use Cases

* Automated real-time loan decisioning systems
* Stream-based ML model integration in FinTech
* Scalable fraud and creditworthiness assessment

---


