// Tool for creating and parsing plans
#include <iostream>
#include <fstream>
#include <string>
#include "ifed/cc/aggregation/plan.pb.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, input, "", "Input plan file (binary)");
ABSL_FLAG(std::string, output, "", "Output plan file (binary)");
ABSL_FLAG(std::string, text, "", "Text proto file to parse or write");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::string input = absl::GetFlag(FLAGS_input);
  std::string output = absl::GetFlag(FLAGS_output);
  std::string text = absl::GetFlag(FLAGS_text);

  if (!input.empty()) {
    ifed::engine::tff::Plan plan;
    std::ifstream in(input, std::ios::binary);
    plan.ParseFromIstream(&in);
    if (!text.empty()) {
      std::ofstream txt(text);
      google::protobuf::TextFormat::Print(plan, &txt);
      std::cout << "Plan written as text proto to: " << text << std::endl;
    } else {
      std::cout << "Plan parsed from binary." << std::endl;
    }
  }
  if (!text.empty() && !output.empty()) {
    ifed::engine::tff::Plan plan;
    std::ifstream txt(text);
    std::string text_proto((std::istreambuf_iterator<char>(txt)), std::istreambuf_iterator<char>());
    google::protobuf::TextFormat::ParseFromString(text_proto, &plan);
    std::ofstream out(output, std::ios::binary);
    plan.SerializeToOstream(&out);
    std::cout << "Plan written as binary to: " << output << std::endl;
  }
  return 0;
}
