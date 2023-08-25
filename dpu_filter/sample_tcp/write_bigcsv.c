#include <stdio.h>

#define FILENAME "/mnt/ssd/bigdata.csv"
#define TARGET_FILE_SIZE_BYTES 1048576 // 1 MB in bytes

int main() {
    int numRows;
    int rowSizeBytes = sizeof(int) + 1; // Size of each row (integer + newline)

    numRows = TARGET_FILE_SIZE_BYTES / rowSizeBytes;

    FILE* file = fopen(FILENAME, "w");

    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    for (int i = 1; i <= numRows; i++) {
        fprintf(file, "%d\n", i); // Write each number on a new row
    }

    fclose(file);
    printf("CSV file created successfully in /mnt/ssd.\n");

    return 0;
}
