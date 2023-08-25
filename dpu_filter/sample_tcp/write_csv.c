#include <stdio.h>
#include <string.h>

#define FILENAME "/mnt/ssd/data.csv"

int main() {
    int numRows;

    printf("Enter the number of rows: ");
    scanf("%d", &numRows);

    if (numRows <= 0) {
        printf("Number of rows must be greater than 0.\n");
        return 1;
    }

    FILE* file = fopen(FILENAME, "w");

    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    //fprintf(file, "Numbers\n"); // Write the header row

    for (int i = 1; i <= numRows; i++) {
        fprintf(file, "%d\n", i); // Write each number on a new row
    }

    fclose(file);
    printf("CSV file created successfully in /mnt/ssd.\n");

    return 0;
}
