
#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/SelectObjectContentRequest.h>
#include <aws/s3/model/SelectObjectContentHandler.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

const char* sql_query = R"(
    SELECT SUM(l_extendedprice * l_discount) AS revenue
    FROM S3Object
    WHERE l_shipdate >= '1994-01-01 00:00:00' AND l_shipdate < '1995-01-01 00:00:00'
    AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24000
)";

void ProcessRecords(Aws::S3::Model::SelectObjectContentHandler& handler, double& total_revenue) {
    handler.SetRecordsEventCallback([&total_revenue](const Aws::S3::Model::RecordsEvent& records_event) {
        std::string payload(records_event.GetPayload().begin(), records_event.GetPayload().end());
        json json_records = json::parse(payload);
        std::cout << json_records << std::endl; // Debug: Print the JSON record
        total_revenue = json_records["revenue"].get<double>();
    });

    handler.SetStatsEventCallback([](const Aws::S3::Model::StatsEvent& stats_event) {
        std::cout << "Processed " << stats_event.GetDetails().GetBytesProcessed()
                  << " bytes in " << stats_event.GetDetails().GetBytesReturned()
                  << " bytes returned" << std::endl;
    });

    handler.SetEndEventCallback([]() {
        std::cout << "End of response" << std::endl;
    });
}

int main() {


    if (setenv("AWS_EC2_METADATA_DISABLED", "true", 1) != 0) {
        // Handle error if needed
    }
    Aws::SDKOptions options;
    Aws::InitAPI(options);

    auto program_start_time = std::chrono::high_resolution_clock::now();

    {
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.endpointOverride = "http://10.10.10.18:9000";  // Your MinIO endpoint
        clientConfig.verifySSL = false;
        clientConfig.scheme = Aws::Http::Scheme::HTTP;

        Aws::Auth::AWSCredentials credentials("minioadmin", "minioadmin");
        Aws::S3::S3Client s3_client(credentials, clientConfig, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);

        auto s3_client_start_time = std::chrono::high_resolution_clock::now();

        Aws::S3::Model::SelectObjectContentRequest select_request;
        select_request.SetBucket("mycsvbucket");
        select_request.SetKey("sampledata/lineitem_converted.parquet");
        select_request.SetExpressionType(Aws::S3::Model::ExpressionType::SQL);
        select_request.SetExpression(sql_query);

        Aws::S3::Model::InputSerialization input_serialization;
        input_serialization.SetParquet(Aws::S3::Model::ParquetInput());
        select_request.SetInputSerialization(input_serialization);

        Aws::S3::Model::OutputSerialization output_serialization;
        output_serialization.SetJSON(Aws::S3::Model::JSONOutput());
        select_request.SetOutputSerialization(output_serialization);

        double total_revenue = 0.0;

        Aws::S3::Model::SelectObjectContentHandler handler;
        ProcessRecords(handler, total_revenue);
        select_request.SetEventStreamHandler(handler);

        auto query_start_time = std::chrono::high_resolution_clock::now();

        auto outcome = s3_client.SelectObjectContent(select_request);

        auto query_end_time = std::chrono::high_resolution_clock::now();

        if (!outcome.IsSuccess()) {
            std::cerr << "Failed to execute S3 Select query: " << outcome.GetError().GetMessage() << std::endl;
        }



        std::cout << "Total revenue: " << total_revenue << std::endl;

        auto program_end_time = std::chrono::high_resolution_clock::now();

        // Print the durations
        std::cout << "S3 client initialization time: "
                  << std::chrono::duration<double>(s3_client_start_time - program_start_time).count() << " seconds" << std::endl;
        std::cout << "S3 Select query execution time: "
                  << std::chrono::duration<double>(query_end_time - query_start_time).count() << " seconds" << std::endl;
          std::cout << "Total program execution time: "
                  << std::chrono::duration<double>(program_end_time - program_start_time).count() << " seconds" << std::endl;
    }

    Aws::ShutdownAPI(options);
    return 0;
}
