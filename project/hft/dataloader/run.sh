docker run -d --name="dataloader" --restart="unless-stopped" --network="host" --log-opt max-size=10m thesis/dataloader:0.0.2
