#include <stdio.h>
#include "process_data.h" // Include your header file

int main() {
    const char* filename = "data.parquet";
    int result = processParquetFile(filename);
    if (result == 0) {
        printf("Processing successful\n");
    } else {
        printf("Processing failed\n");
    }
    return 0;
}
