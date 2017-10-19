// number of total netowrk ports. At present max supported is one only. Do not change this value.
#define TOTAL_PORTS 1

#define PORT_CONFIGURE 0 

// IP of the interface 1, port 0

#define IP_INTERFACE_1 "192.168.78.2"

#define TARGET_MELLANOX

#ifdef  TARGET_MELLANOX
#define DISABLE_FAKE
#endif
