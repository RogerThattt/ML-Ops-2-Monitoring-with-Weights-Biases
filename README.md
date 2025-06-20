# ML-Ops-2-Monitoring-with-Weights-Biases

<html>

+------------------------+       +------------------------+
|  Network Devices / OSS| ----> | Streaming / Batch Data |
| (Routers, RAN, etc.)  |       |  Ingestion (Kafka/S3)  |
+------------------------+       +------------------------+
                                         |
                                         v
+---------------------+      +-------------------------+
| Feature Engineering | ---> |   Model Training (DBX)  |
|   (time windowing)  |      |    Autoencoders / LSTM  |
+---------------------+      +-------------------------+
                                         |
                                         v
                                +------------------------+
                                |   Weights & Biases     |
                                |  Experiment Tracking   |
                                |   Model Monitoring     |
                                +------------------------+
                                         |
                                         v
                                +------------------------+
                                | Real-time Inference    |
                                |   (SageMaker, Vertex)  |
                                +------------------------+
</html>
