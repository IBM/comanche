#!/bin/bash
DIRS="./include src/include src/lib/common src/lib/kivati_client src/python/kivati src/nvme/nvme_memory"
sloccount -- $DIRS
