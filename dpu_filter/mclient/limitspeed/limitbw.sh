# Convert 50 Mbps to Kbps
downlink_speed_kbps=50000  # 50 Mbps = 50000 Kbps
uplink_speed_kbps=50000    # 50 Mbps = 50000 Kbps

# Set Wondershaper limits
sudo wondershaper ens785f1np1 $downlink_speed_kbps $uplink_speed_kbps
