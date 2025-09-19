# Tools for creating and parsing checkpoints and plans

cc_binary(
	name = "checkpoint_tool",
	srcs = ["checkpoint_tool.cc"],
	deps = [
		"//tensorflow_federated/cc/core/impl/aggregation/tensorflow:tensorflow_checkpoint_builder_factory",
    "//tensorflow_federated/cc/core/impl/aggregation/tensorflow:tensorflow_checkpoint_parser_factory",
		"@com_google_absl//absl/flags:flag",
		"@com_google_absl//absl/flags:parse",
		"@com_google_absl//absl/status:statusor",
		"@com_google_absl//absl/strings:cord",
    "@nlohmann_json//:json",
	],
)
