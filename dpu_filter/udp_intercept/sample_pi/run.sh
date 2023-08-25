cd build
sudo ./doca_pi -a auxiliary:mlx5_core.sf.2,sft_en=1 -a auxiliary:mlx5_core.sf.3,sft_en=1 -- --cdo /tmp/signatures.cdo -p -a 03:00.0
