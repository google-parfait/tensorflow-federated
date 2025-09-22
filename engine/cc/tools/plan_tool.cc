// Tool for creating and parsing plans

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "absl/flags/usage.h"
#include "engine/cc/aggregation/plan.pb.h"
#include "google/protobuf/util/json_util.h"
#include "absl/flags/parse.h"

constexpr const char* kUsageMessage = R"(Usage:
  plan_tool build <output_bin>
    Build a binary Plan proto from JSON read from stdin.
  plan_tool parse <input_bin>
    Parse a binary Plan proto to JSON written to stdout.
  plan_tool sample
    Print a minimal Plan JSON with only ServerPhaseV2.aggregations filled.
)";

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(kUsageMessage);
  absl::ParseCommandLine(argc, argv);
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() < 1) {
    std::cerr << kUsageMessage;
    return 1;
  }

  const std::string& command = args[0];
  if (command == "sample") {
    // Minimal Plan with only ServerPhaseV2.aggregations filled
    engine::tff::Plan plan;
    auto* spv2 = plan.add_phase()->mutable_server_phase_v2();
    auto* agg = spv2->add_aggregations();
    agg->set_intrinsic_uri("federated_sum");
    // Add a minimal input_tensor argument
    auto* arg = agg->add_intrinsic_args();
    auto* input_tensor = arg->mutable_input_tensor();
    input_tensor->set_name("client_tensor");
    input_tensor->set_dtype(::tensorflow::DT_FLOAT);
    input_tensor->mutable_shape()->add_dim()->set_size(10);
    // Add a minimal output_tensor
    auto* output_tensor = agg->add_output_tensors();
    output_tensor->set_name("aggregated_tensor");
    output_tensor->set_dtype(::tensorflow::DT_FLOAT);
    output_tensor->mutable_shape()->add_dim()->set_size(10);
    std::string json;
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    auto status = google::protobuf::util::MessageToJsonString(plan, &json, options);
    if (!status.ok()) {
      std::cerr << "Failed to convert Plan proto to JSON: " << status.ToString() << std::endl;
      return 1;
    }
    std::cout << json;
    return 0;
  }
  if (command == "build") {
    if (args.size() != 2) {
      std::cerr << kUsageMessage;
      return 1;
    }
    const std::string& output_bin = args[1];
    std::string json((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    engine::tff::Plan plan;
    auto status = google::protobuf::util::JsonStringToMessage(json, &plan);
    if (!status.ok()) {
      std::cerr << "Failed to parse JSON: " << status.ToString() << std::endl;
      return 1;
    }
    std::ofstream out(output_bin, std::ios::binary);
    if (!out) {
      std::cerr << "Failed to open output file: " << output_bin << std::endl;
      return 1;
    }
    if (!plan.SerializeToOstream(&out)) {
      std::cerr << "Failed to serialize Plan proto." << std::endl;
      return 1;
    }
    std::cout << "Plan binary written to: " << output_bin << std::endl;
  } else if (command == "parse") {
    if (args.size() != 2) {
      std::cerr << kUsageMessage;
      return 1;
    }
    const std::string& input_bin = args[1];
    std::ifstream in(input_bin, std::ios::binary);
    if (!in) {
      std::cerr << "Failed to open input binary: " << input_bin << std::endl;
      return 1;
    }
    engine::tff::Plan plan;
    if (!plan.ParseFromIstream(&in)) {
      std::cerr << "Failed to parse Plan proto from binary." << std::endl;
      return 1;
    }
    std::string json;
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    auto status = google::protobuf::util::MessageToJsonString(plan, &json, options);
    if (!status.ok()) {
      std::cerr << "Failed to convert Plan proto to JSON: " << status.ToString() << std::endl;
      return 1;
    }
    std::cout << json;
  } else {
    std::cerr << kUsageMessage;
    return 1;
  }
  return 0;
}
