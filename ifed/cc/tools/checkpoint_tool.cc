#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "absl/flags/flag.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_parser_factory.h"
#include "absl/flags/parse.h"
#include "nlohmann/json.hpp"
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

int parse(const std::string& src) {
  // Read checkpoint file
  std::ifstream in(src, std::ios::binary);
  if (!in) {
    std::cerr << "Failed to open checkpoint file: " << src << std::endl;
    return 1;
  }
  std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();

  absl::Cord serializedCheckpoint;
  serializedCheckpoint.Append(std::string(buffer.begin(), buffer.end()));

  tensorflow_federated::aggregation::tensorflow::TensorflowCheckpointParserFactory factory;
  auto parser = factory.Create(serializedCheckpoint);
  if (!parser.ok()) {
    std::cerr << "Failed to create checkpoint parser: " << parser.status() << std::endl;
    return 1;
  }

  const auto tensor_names = parser.value()->ListTensorsNames();
  nlohmann::json tensor_list = nlohmann::json::array();
  for (const auto& name : tensor_names.value()) {
    const auto tensor = parser.value()->GetTensor(name);
    if (!tensor.ok()) {
      std::cerr << "Failed to get tensor " << name << ": " << tensor.status() << std::endl;
      return 1;
    }
    std::string json_str;
    auto tensor_proto = tensor.value().ToProto();
    tensor_proto.set_name(name);
    google::protobuf::util::MessageToJsonString(tensor_proto, &json_str);
    tensor_list.push_back(nlohmann::json::parse(json_str));
  }

  std::cout << tensor_list.dump(2) << std::endl;
  return 0;
}


int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(R"(
Checkpoint Tool: Build or parse TensorFlow checkpoints from JSON or binary files.

Usage:
  checkpoint_tool build <output.ckpt> < input.json
  checkpoint_tool parse <input.ckpt> > output.json

Commands:
  build            Convert checkpoint in JSON from stdin to binary checkpoint file.
  parse            Parse binary checkpoint file and write checkpoint content as JSON to stdout.

Arguments:
  <output.ckpt>      Output checkpoint file path (for build).
  <input.ckpt>       Input checkpoint file path (for parse).

Input JSON format:
  [
    {
      "name": "weights",
      "dtype": "DT_FLOAT",
      "shape": { "dim_sizes": [2, 2] },
      "float_val": [1.0, 2.0, 3.0, 4.0]
    },
    ...
  ]

Examples:
  checkpoint_tool build ckpt.bin < input.json
  checkpoint_tool parse ckpt.bin > output.json

)");

  absl::ParseCommandLine(argc, argv);
  if (argc < 3 && (argc < 2 || std::string(argv[1]) == "build")) {
    std::cerr << "Usage: " << argv[0] << " build <output.ckpt> < input.json\n"
              << argv[0] << " parse <input.ckpt>" << std::endl;
    return 1;
  }
  std::string command = argv[1];
  int retcode = 1;
  if (command == "build") {
    std::string target_path = argv[2];
    retcode = build(target_path);
  } else if (command == "parse") {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " parse <input.ckpt>" << std::endl;
      return 1;
    }
    std::string src_path = argv[2];
    retcode = parse(src_path);
  } else {
    std::cerr << "Unknown command: " << command << std::endl;
    std::cerr << "Usage: " << argv[0] << " build <output.ckpt> < input.json\n"
              << argv[0] << " parse <input.ckpt>" << std::endl;
    return 1;
  }

  if (retcode != 0) {
    std::cerr << "Command failed with code: " << retcode << std::endl;
  } else {
    std::cout << "Command completed successfully." << std::endl;
  }

  return retcode;
}
