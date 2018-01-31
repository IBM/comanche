.load ./mddb.so
CREATE VIRTUAL TABLE metadata USING mddb(pci=06:00.0,partition=0);
.quit

