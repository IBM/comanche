#ifdef __cplusplus
extern "C" {
#endif

// Declare your C++ function using C linkage
unsigned char* processParquetData(const unsigned char* data_buffer, size_t data_size, size_t *filtered_buffer_size);

#ifdef __cplusplus
}
#endif
