git clone https://github.com/electrum/tpch-dbgen.git
cd tpch-dbgen
make


#1GB data set -s scale factor
./dbgen -s 1 -f -v

