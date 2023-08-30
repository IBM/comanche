cd build
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
#sudo ./doca_ips -a auxiliary:mlx5_core.sf.2,sft_en=1 -a auxiliary:mlx5_core.sf.3,sft_en=1 -- --cdo /tmp/signatures.cdo -p -a 03:00.0
sudo ./doca_ips -a auxiliary:mlx5_core.sf.4,sft_en=1 -a auxiliary:mlx5_core.sf.5,sft_en=1 -l 0-1 -- --cdo /tmp/signatures.cdo -p -a 03:00.0
