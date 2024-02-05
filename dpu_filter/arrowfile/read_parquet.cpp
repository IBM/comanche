#include <iostream>
#include <memory>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/statistics.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <ParquetFilePath>" << std::endl;
    return 1;
  }

  std::string parquetFilePath = argv[1];

  try {
    // Get the Parquet file path from the command-line argument
   

    // Open the Parquet file
    std::shared_ptr<parquet::ParquetFileReader> reader =
        parquet::ParquetFileReader::OpenFile(parquetFilePath);

    // Get the file metadata
    std::shared_ptr<parquet::FileMetaData> metadata = reader->metadata();

    // Print metadata information
    std::cout << "Created By: " << metadata->created_by() << std::endl;
    std::cout << "Number of Row Groups: " << metadata->num_row_groups() << std::endl;

    // Loop through row groups and print their metadata
    for (int i = 0; i < metadata->num_row_groups(); ++i) {
      std::shared_ptr<parquet::RowGroupMetaData> row_group_metadata =
          metadata->RowGroup(i);
      std::cout << "Row Group " << i << ":\n";
      std::cout << "  Number of Columns: " << row_group_metadata->num_columns()
                << std::endl;

      // Loop through columns in the row group and print statistics
      for (int j = 0; j < row_group_metadata->num_columns(); ++j) {
        std::shared_ptr<parquet::ColumnChunkMetaData> column_metadata =
            row_group_metadata->ColumnChunk(j);

        std::cout << "  Column " << j << ":\n";
        //std::cout << "    Column Name: " << column_metadata->column_name()
                //  << std::endl;
        std::cout << "    Total Rows: " << column_metadata->num_values()
                  << std::endl;

        // Check statistics for the column
        std::shared_ptr<parquet::Statistics> column_stats =
            column_metadata->statistics();

        if (column_stats) {
          std::cout << "    Min: " << column_stats->EncodeMin() << std::endl;
          std::cout << "    Max: " << column_stats->EncodeMax() << std::endl;
          std::cout << "    Distinct Count: " << column_stats->distinct_count()
                    << std::endl;
        } else {
          std::cout << "    No statistics available for this column"
                    << std::endl;
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

     // Initialize Arrow
    arrow::Status status = arrow::Status::OK();

    // Open Parquet file for reading
    std::shared_ptr<arrow::io::RandomAccessFile> input;
    status = arrow::io::ReadableFile::Open(parquetFilePath).Value(&input);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file: " << status.ToString() << std::endl;
        return 1;
    }

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &arrow_reader);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
        return 1;
    }

    // Read entire file as a single Arrow table
    std::shared_ptr<arrow::Table> table;
    status = arrow_reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
        return 1;
    }

    // Display the Arrow table's schema
    std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    std::cout << "Table Data:\n" << table->ToString() << std::endl;


  return 0;
}
