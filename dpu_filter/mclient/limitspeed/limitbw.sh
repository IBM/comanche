# Convert 50 Mbps to Kbps
#downlink_speed_kbps=50000  # 50 Mbps = 50000 Kbps
#uplink_speed_kbps=50000    # 50 Mbps = 50000 Kbps


downlink_speed_kbps = 700000  # 150 Mbps = 150,000 Kbps
uplink_speed_kbps = 700000    # 150 Mbps = 150,000 Kbps


#downlink_speed_kbps = 300000  # 300 Mbps = 300,000 Kbps
#uplink_speed_kbps = 300000    # 300 Mbps = 300,000 Kbps


#downlink_speed_kbps = 450000  # 450 Mbps = 450,000 Kbps
#uplink_speed_kbps = 450000    # 450 Mbps = 450,000 Kbps

# Set Wondershaper limits
sudo wondershaper ens785f1np1 400000 400000

#sudo wondershaper ens785f1np1 $downlink_speed_kbps $uplink_speed_kbps

# Set Wondershaper limits
#sudo wondershaper ens785f1np1 $downlink_speed_kbps $uplink_speed_kbps
