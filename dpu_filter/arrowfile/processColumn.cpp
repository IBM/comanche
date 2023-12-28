#include <iostream>
#include <vector>
#include <fstream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/statistics.h>
#include <chrono>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/statistics.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include <arrow/builder.h>
#include <cstdlib> // Include this header for string conversion

std::string processParquetFile(const char* filename, int filter_id) {
    const std::string path_to_file = filename;

    std::cout << "Start" << std::endl;

    // Open Parquet file reader
    arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> readable_file =
        arrow::io::ReadableFile::Open(path_to_file);
    if (!readable_file.ok()) {
        std::cerr << "Error opening Parquet file: " << readable_file.status().ToString() << std::endl;
        return "";
    }

    arrow::Status status = arrow::Status::OK();

    // Create an Arrow memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(readable_file.ValueOrDie(), pool, &arrow_reader);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
        return "";
    }

    // Ensure there is at least one row group
    if (arrow_reader->num_row_groups() == 0) {
        std::cerr << "No row groups found in the Parquet file." << std::endl;
        return "";
    }

    // Read the 'ID' column from RowGroup 0
    int row_group_index = 0;
    int id_column_index = 1; // 1 corresponds to the second column ("ID")

    // Read the specific column from RowGroup 0
    std::shared_ptr<arrow::Table> id_table;
    status = arrow_reader->ReadRowGroup(row_group_index, {id_column_index}, &id_table);
    if (!status.ok()) {
        std::cerr << "Error reading Arrow table for 'ID' column: " << status.ToString() << std::endl;
        return "";
    }

    // Extract the 'ID' column
    std::shared_ptr<arrow::Array> id_array = id_table->column(0)->chunk(0);

    // Define the filtering condition
    auto id_field = arrow::compute::field_ref("ID");
    auto filter_id_scalar = arrow::compute::literal(filter_id);

    auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(id_table);

    // 2: Build ScannerOptions for a Scanner to do a basic filter operation
    auto options = std::make_shared<arrow::dataset::ScanOptions>();
    options->filter = arrow::compute::less(id_field, filter_id_scalar);

    // 3: Build the Scanner
    auto builder = arrow::dataset::ScannerBuilder(dataset, options);
    auto scanner = builder.Finish();

    // 4: Perform the Scan and make a Table with the result
    arrow::Result<std::shared_ptr<arrow::Table>> result = scanner.ValueUnsafe()->ToTable();
    if (!result.ok()) {
        std::cerr << "Error filtering Arrow Table: " << result.status().ToString() << std::endl;
        return "";
    }

    auto filtered_table = result.ValueOrDie();

    // Extract filtered 'ID' values
    std::shared_ptr<arrow::Array> filtered_id_array = filtered_table->column(0)->chunk(0);

    // Now, iterate through other columns and copy pages matching the filtered 'ID' values
    std::vector<std::shared_ptr<arrow::RecordBatch>> copied_batches;

    for (int col_index = 0; col_index < arrow_reader->num_columns(); ++col_index) {
        // Skip the 'ID' column, as it's already filtered
        if (col_index == id_column_index) {
            continue;
        }

        std::shared_ptr<arrow::Table> col_table;
        status = arrow_reader->ReadRowGroup(row_group_index, {col_index}, &col_table);
        if (!status.ok()) {
            std::cerr << "Error reading Arrow table for column " << col_index << ": " << status.ToString() << std::endl;
            return "";
        }

        // Copy pages matching the filtered 'ID' values
        std::shared_ptr<arrow::Table> copied_table;
        status = arrow::dataset::FilterTableByIndices(*col_table, filtered_id_array, &copied_table);
        if (!status.ok()) {
            std::cerr << "Error copying pages for column " << col_index << ": " << status.ToString() << std::endl;
            return "";
        }

        // Convert the copied table to RecordBatches
        auto num_batches = copied_table->num_record_batches();
        for (int batch_index = 0; batch_index < num_batches; ++batch_index) {
            copied_batches.push_back(copied_table->Slice(batch_index, 1)->ToRecordBatch());
        }
    }

    // Serialize the filtered RecordBatches to the output Parquet file
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string output_filename = "filtered_output_" + std::to_string(filter_id) + ".parquet";

    status = parquet::arrow::WriteTable(arrow::Table::FromRecordBatches(copied_batches), arrow::default_memory_pool(),
                                        arrow::io::FileOutputStream::Open(output_filename).ValueOrDie());
    if (!status.ok()) {
        std::cerr << "Error writing filtered Arrow table to Parquet file: " << status.ToString() << std::endl;
        return "";
    }

    return output_filename;
}

int main(int argc, char* argv[]) {
    if (argc != 3) { // Update the argument count check
        std::cerr << "Usage: " << argv[0] << " <parquet_filename> <Filter_id>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    int filter_id = std::atoi(argv[2]); // Convert command-line argument to an integer

    std::string output_filename = processParquetFile(filename, filter_id);

    if (!output_filename.empty()) {
        std::cout << "Processing successful. Output file: " << output_filename << std::endl;
    } else {
        std::cerr << "Processing failed." << std::endl;
    }

    return output_filename.empty() ? 1 : 0;
}


/**#include <iostream>
#include <vector>
#include <fstream> 
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/statistics.h>
#include <chrono>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/statistics.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include <arrow/builder.h>
#include <cstdlib> // Include this header for string conversion

std::string processParquetFile(const char* filename, int filter_id) {
    const std::string path_to_file = filename;

    std::cout << "Start" << std::endl;

    // Open Parquet file reader
    arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> readable_file =
        arrow::io::ReadableFile::Open(path_to_file);
    if (!readable_file.ok()) {
        std::cerr << "Error opening Parquet file: " << readable_file.status().ToString() << std::endl;
        return "";
    }

    arrow::Status status = arrow::Status::OK();

    // Create an Arrow memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    status = parquet::arrow::OpenFile(readable_file.ValueOrDie(), pool, &arrow_reader);
    if (!status.ok()) {
        std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
        return "";
    }

    // Ensure there is at least one row group
    if (arrow_reader->num_row_groups() == 0) {
        std::cerr << "No row groups found in the Parquet file." << std::endl;
        return "";
    }

// Print min and max statistics for the 'ID' column in each row group
for (int row_group_index = 0; row_group_index < arrow_reader->num_row_groups(); ++row_group_index) {
    auto metadata = arrow_reader->parquet_reader()->metadata();
    auto row_group = metadata->RowGroup(row_group_index);

    std::cout << "Row Group " << row_group_index << " Statistics:" << std::endl;
    // Assuming 'ID' is the second column (index 1), adjust as needed
    int column_index = 1;
    auto column_chunk = row_group->ColumnChunk(column_index);
    
    // Decode and print the 'ID' column statistics
    std::string min_stat_str(column_chunk->statistics()->EncodeMin());
    std::string max_stat_str(column_chunk->statistics()->EncodeMax());

    // Decode binary data to obtain the actual values
    int64_t min_stat;
    int64_t max_stat;
    memcpy(&min_stat, min_stat_str.data(), sizeof(int64_t));
    memcpy(&max_stat, max_stat_str.data(), sizeof(int64_t));

    std::cout << "Min: " << min_stat << std::endl;
    std::cout << "Max: " << max_stat << std::endl;
}

    // Read the second column ("ID") from RowGroup 0
    int row_group_index = 0;
    int column_index = 1; // 1 corresponds to the second column ("ID")

    // Read the specific column from RowGroup 0
    std::shared_ptr<arrow::Table> table;
    //status = arrow_reader->ReadRowGroup(row_group_index, &table);
    status = arrow_reader->ReadRowGroup(row_group_index, {column_index}, &table);
    if (!status.ok()) {
        std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
        return "";
    }

    // Display the Arrow table's schema
    std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    std::cout << "Table Data:\n" << table->ToString() << std::endl;




    // Define the filtering condition
    
    auto id_field = arrow::compute::field_ref("ID");
    auto filter_id_scalar = arrow::compute::literal(filter_id);


    auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);


    // 2: Build ScannerOptions for a Scanner to do a basic filter operation
    auto options = std::make_shared<arrow::dataset::ScanOptions>();
    options->filter = arrow::compute::less(id_field, filter_id_scalar);



    // 3: Build the Scanner
    auto builder = arrow::dataset::ScannerBuilder(dataset, options);
    auto scanner = builder.Finish();

     

    // 4: Perform the Scan and make a Table with the result
    arrow::Result<std::shared_ptr<arrow::Table>> result = scanner.ValueUnsafe()->ToTable();
    if (!result.ok()) {
        std::cerr << "Error filtering Arrow Table: " << result.status().ToString() << std::endl;
        return "";
    }
    auto filtered_table = result.ValueOrDie();

   

    // Display the Arrow table's schema
    std::cout << "Filtered Table Schema:\n" << filtered_table->schema()->ToString() << std::endl;

    // Display the Arrow table's data
    std::cout << "Filtered Table Data:\n" << filtered_table->ToString() << std::endl;
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string output_filename = "filtered_output_" + std::to_string(filter_id) + ".parquet";

    // Serialize the filtered table to the output Parquet file
    status = parquet::arrow::WriteTable(*filtered_table, arrow::default_memory_pool(),
                                        arrow::io::FileOutputStream::Open(output_filename).ValueOrDie());
    if (!status.ok()) {
        std::cerr << "Error writing filtered Arrow table to Parquet file: " << status.ToString() << std::endl;
        return "";
    }

    return output_filename;
}

int main(int argc, char* argv[]) {
    if (argc != 3) { // Update the argument count check
        std::cerr << "Usage: " << argv[0] << " <parquet_filename> <Filter_id>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    int filter_id = std::atoi(argv[2]); // Convert command-line argument to an integer
    
    std::string output_filename = processParquetFile(filename, filter_id);

    if (!output_filename.empty()) {
        std::cout << "Processing successful. Output file: " << output_filename << std::endl;
    } else {
        std::cerr << "Processing failed." << std::endl;
    }

    return output_filename.empty() ? 1 : 0;
}*/
