#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

#include <iostream>
#include <string>
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "absl/flags/flag.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"
#include "absl/flags/parse.h"
#include "nlohmann/json.hpp"  // For JSON parsing (header-only)
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include "absl/flags/usage.h"


int build(const std::string& dst) {
  nlohmann::json src;
  try {
    std::cin >> src;
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse JSON from stdin: " << e.what() << std::endl;
    return 1;
  }

  tensorflow_federated::aggregation::tensorflow::TensorflowCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  if (!builder) {
    std::cerr << "Failed to create checkpoint builder" << std::endl;
    return 1;
  }

  if (!src.is_array()) {
    std::cerr << "JSON input must be a list of tensors." << std::endl;
    return 1;
  }

  for (const auto& tensor_json : src) {
    tensorflow_federated::aggregation::TensorProto tensor_proto;
    if (!google::protobuf::util::JsonStringToMessage(tensor_json.dump(), &tensor_proto).ok()) {
      std::cerr << "Failed to parse tensor JSON to TensorProto." << std::endl;
      return 1;
    }

    auto tensor_or = tensorflow_federated::aggregation::Tensor::FromProto(tensor_proto);
    if (!tensor_or.ok()) {
      std::cerr << "Failed to create tensor from proto: " << tensor_or.status() << std::endl;
      return 1;
    }
    absl::Status add_status = builder->Add(tensor_proto.name(), *tensor_or);
    if (!add_status.ok()) {
      std::cerr << "Failed to add tensor: " << add_status << std::endl;
      return 1;
    }
  }

  auto res = builder->Build();
  if (!res.ok()) {
    std::cerr << "Failed to build checkpoint: " << res.status() << std::endl;
    return 1;
  }

  std::string checkpoint;
  absl::CopyCordToString(*res, &checkpoint);
  std::ofstream out(dst, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open output file: " << dst << std::endl;
    return 1;
  }

  out.write(checkpoint.data(), checkpoint.size());
  out.close();
  return 0;
}

ABSL_FLAG(std::string, to, "", "Output checkpoint file");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(R"(
Checkpoint Tool: Convert a JSON list of tensors to a TensorFlow checkpoint file.

Usage:
  checkpoint_tool build --to=<filename> < input.json
  checkpoint_tool --command=build --to=<filename> < input.json

Arguments:
  build            Run the tool in build mode (required).
  --to=<filename>  Output checkpoint file path (required).
  < input.json     JSON file containing a list of tensor protos (see below).

Input JSON format:
  [
    {
      "name": "weights",
      "dtype": "DT_FLOAT",
      "shape": { "dim_sizes": [2, 2] },
      "values": [1.0, 2.0, 3.0, 4.0]
    },
    ...
  ]

For more details, see the documentation or source code.

Example:
  checkpoint_tool build --to=ckpt.bin < input.json

)");

  absl::ParseCommandLine(argc, argv);
  std::string command;
  if (argc > 1) {
    command = argv[1];
  } else {
    std::cerr << "Usage: " << argv[0] << " --command=build --to=<filename> < input.json" << std::endl;
    std::cerr << "   or: " << argv[0] << " build --to=<filename> < input.json" << std::endl;
    return 1;
  }

  int retcode = 1;
  if (command == "build") {
    std::string target_path = absl::GetFlag(FLAGS_to);
    if (target_path.empty()) {
      std::cerr << "--to flag must be specified." << std::endl;
      return 1;
    }

    retcode = build(target_path);
  } else {
    std::cerr << "Unknown command: " << command << std::endl;
    std::cerr << "Usage: " << argv[0] << " --command=build --to=<filename> < input.json" << std::endl;
    std::cerr << "   or: " << argv[0] << " build --to=<filename> < input.json" << std::endl;
    return 1;
  }

  if (retcode != 0) {
    std::cerr << "Command failed with code: " << retcode << std::endl;
  } else {
    std::cout << "Command completed successfully." << std::endl;
  }

  return retcode;
}
