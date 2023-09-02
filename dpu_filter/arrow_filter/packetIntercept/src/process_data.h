#ifdef __cplusplus
extern "C" {
#endif

// Declare your C++ function using C linkage
//int processParquetFile(const char* filename);

unsigned char* processParquetFile(const char* filename, size_t* buffer_size);
#ifdef __cplusplus
}
#endif
