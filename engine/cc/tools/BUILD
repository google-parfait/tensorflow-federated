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

cc_binary(
	name = "plan_tool",
	srcs = ["plan_tool.cc"],
	deps = [
		"//engine/cc/aggregation:plan_cc_proto",
		"@com_google_absl//absl/flags:parse",
		"@com_google_absl//absl/strings:str_format",
		"@com_google_protobuf//:protobuf",
		#"@com_google_protobuf//:protobuf_util",
	],
)
