import dask_gateway

cluster = dask_gateway.GatewayCluster()
client = cluster.get_client()
cluster.scale(4)
print(cluster.dashboard_link)

import time

time.sleep(60)