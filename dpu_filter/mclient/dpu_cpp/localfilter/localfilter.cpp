#include <pistache/endpoint.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <chrono>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h> // Include this header
#include "/home/ubuntu/json/include/nlohmann/json.hpp"


using namespace Pistache;
using json = nlohmann::json;


struct S3Handler : public Http::Handler {
    HTTP_PROTOTYPE(S3Handler)

    void onRequest(const Http::Request& req, Http::ResponseWriter response) override {


        json requestJson = json::parse(req.body());

            // Extract bucket, key, and SQL expression
        std::string bucket = requestJson["bucket"];
        std::string key = requestJson["key"];
        std::string sqlExpression = requestJson["sql"];

        // Initialize AWS SDK
        Aws::SDKOptions options;
        Aws::InitAPI(options);

        // MinIO server connection parameters
        Aws::String minioEndpointUrl = "https://10.10.10.18:9000";
        Aws::String awsAccessKey = "minioadmin";
        Aws::String awsSecretKey = "minioadmin";

        // Create S3 client configuration
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.endpointOverride = minioEndpointUrl;
        clientConfig.scheme = Aws::Http::Scheme::HTTPS;
        clientConfig.verifySSL = false;

        // Create AWSCredentials object
        Aws::Auth::AWSCredentials credentials(awsAccessKey, awsSecretKey);

        // Create S3 client
        Aws::S3::S3Client s3Client(credentials, clientConfig,
                                    Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);


        try {
            // Get the object from S3
            Aws::S3::Model::GetObjectRequest getObjectRequest;
            getObjectRequest.SetBucket(bucket.c_str());
            getObjectRequest.SetKey(key.c_str());

            auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
            if (getObjectOutcome.IsSuccess()) {
                // Read the Parquet data from S3
                auto& objectStream = getObjectOutcome.GetResult().GetBody();
                
                // Read the stream into a vector: takes long time
                //std::vector<char> data(std::istreambuf_iterator<char>(objectStream), {});

/////////////////////// Faster way Estimate the size of the stream to reserve vector capacity upfront (optional)
                std::vector<char> data;
                
                objectStream.seekg(0, std::ios::end);
                std::streamsize size = objectStream.tellg();
                objectStream.seekg(0, std::ios::beg);

                if (size > 0) {
                    data.reserve(static_cast<size_t>(size));
                }

                // Use a buffer to read chunks of the stream
                constexpr std::streamsize bufferSize = 8192*8; // Example buffer size, adjust as needed
                char buffer[bufferSize];

                while (objectStream.read(buffer, bufferSize) || objectStream.gcount() > 0) {
                    data.insert(data.end(), buffer, buffer + objectStream.gcount());
                }

                // Create an Arrow buffer from the vector
                auto arrowBuffer = arrow::Buffer::Wrap(data.data(), data.size());
                // Create an Arrow BufferReader from the Arrow buffer
                auto bufferReader = std::make_shared<arrow::io::BufferReader>(arrowBuffer);

///////////////////////////////////////////////
                // Open Parquet file reader
                std::unique_ptr<parquet::arrow::FileReader> arrowReader;
                auto status = parquet::arrow::OpenFile(bufferReader, arrow::default_memory_pool(), &arrowReader);
                if (!status.ok()) {
                    std::cerr << "Error opening Parquet file reader: " << status.ToString() << std::endl;
                    return;
                }

                // Read entire file as a single Arrow table
                std::shared_ptr<arrow::Table> table;
                status = arrowReader->ReadTable(&table);
                if (!status.ok()) {
                    std::cerr << "Error reading Arrow table: " << status.ToString() << std::endl;
                    return;
                }


                // Wrap the Table in an InMemoryDataset
                std::shared_ptr<arrow::dataset::Dataset> dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);

                // Build ScannerOptions for a Scanner to apply filter operation
                auto options = std::make_shared<arrow::dataset::ScanOptions>();
                options->filter = arrow::compute::less(
            
                //Need to parse http request
                arrow::compute::field_ref("ID"), 
                arrow::compute::literal(120)); // Adjust filter condition as needed

                // Build the Scanner
                auto builder = arrow::dataset::ScannerBuilder(dataset, options);
                auto scanner = builder.Finish();

                // Perform the Scan and retrieve filtered result as Table
                auto result_table = scanner.ValueOrDie()->ToTable();

                // Print the Arrow table's schema
                //std::cout << "Table Schema:\n" << table->schema()->ToString() << std::endl;
                // Print the Arrow table's data
                //std::cout << "Table Data:\n" << result_table.ValueUnsafe()->ToString() << std::endl;

                // Assuming 'filtered_result_json' contains the filtered data in JSON format
                std::string filtered_result_json = result_table.ValueUnsafe()->ToString();

                // Send the JSON response
                response.send(Http::Code::Ok, filtered_result_json, MIME(Application, Json));


// Write filtered data to Parquet file
/*std::shared_ptr<arrow::io::FileOutputStream> outfile;
ARROW_THROW_NOT_OK(arrow::io::FileOutputStream::Open("filtered_data.parquet", &outfile));
ARROW_THROW_NOT_OK(parquet::arrow::WriteTable(*result_table, arrow::default_memory_pool(), outfile, static_cast<int64_t>(result_table->num_rows())));
outfile->Close();

// Send the Parquet file as response
std::ifstream file("filtered_data.parquet", std::ios::binary);
std::ostringstream oss;
oss << file.rdbuf();
std::string filtered_data_parquet = oss.str();

response.send(Http::Code::Ok, filtered_data_parquet, MIME(Application, Parquet));*/

                // Send the JSON response
            
                //response.send(Http::Code::Ok, "jo");
            } else {
                // Error handling
                std::cerr << "Failed to get object: " << getObjectOutcome.GetError().GetMessage() << std::endl;
                response.send(Http::Code::Internal_Server_Error, "Failed to get object from S3");
            }
        } catch (const std::exception& e) {
            // Exception handling
            std::cerr << "Exception: " << e.what() << std::endl;
            response.send(Http::Code::Internal_Server_Error, "Exception occurred");
        }

        // Shutdown AWS SDK
        Aws::ShutdownAPI(options);
    }
};

int main() {
    Http::listenAndServe<S3Handler>(Pistache::Address("*:8080"));
    return 0;
}
