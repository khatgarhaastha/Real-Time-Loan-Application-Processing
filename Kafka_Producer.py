import pandas as pd
import json
import time
from kafka import KafkaProducer

# Load CSV data
data = pd.read_csv("Data/german_credit_data.csv")

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_data_in_batches(data, batch_size=10):
    # Divide data into batches of size 'batch_size'
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size].to_dict(orient='records')
        print(f"Sending batch: {batch}")
        
        # Send batch to Kafka
        producer.send('batch-topic', value=batch)
        
        # Wait a bit to simulate time between batches
        time.sleep(20)

# Run the batch sender
send_data_in_batches(data)
producer.close()
