from kafka import KafkaConsumer
import json
import joblib
import pandas as pd
from main import process_dataset, train_model, train_LR_model

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'batch-topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for batches...")
for message in consumer:
    batch = message.value
    print("Received batch:", batch)
    df = pd.DataFrame(batch)

    # Process the batch
    df = process_dataset(df)

    print(df)

    # load Model 
    model_path = 'Models/RFModel.pkl'
    
    # train model 
    best_DT_model, accuracy = train_model(df, model_path)

    best_LR_model, accuracy = train_LR_model(df, model_path)

    # Print the results
    print("Best Decision Tree Model:", best_DT_model)
    print("Best Logistic Regression Model:", best_LR_model)
    
    
        
    print("Finished processing batch.\n")
