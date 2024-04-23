package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "converters",
    srcs = ["converters.cc"],
    hdrs = ["converters.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:tstring",
        "@org_tensorflow//tensorflow/tsl/platform:refcount",
    ],
)

cc_test(
    name = "converters_test",
    srcs = ["converters_test.cc"],
    deps = [
        ":converters",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:vector_string_data",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:parse_text_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_library(
    name = "checkpoint_reader",
    srcs = ["checkpoint_reader.cc"],
    hdrs = ["checkpoint_reader.h"],
    deps = [
        ":converters",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/c:checkpoint_reader",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/c:tf_status_helper",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "checkpoint_reader_test",
    srcs = ["checkpoint_reader_test.cc"],
    deps = [
        ":checkpoint_reader",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "checkpoint_writer",
    srcs = ["checkpoint_writer.cc"],
    hdrs = ["checkpoint_writer.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_test(
    name = "checkpoint_writer_test",
    srcs = ["checkpoint_writer_test.cc"],
    deps = [
        ":checkpoint_reader",
        ":checkpoint_writer",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
    ],
)

cc_library(
    name = "tensorflow_checkpoint_builder_factory",
    srcs = ["tensorflow_checkpoint_builder_factory.cc"],
    hdrs = ["tensorflow_checkpoint_builder_factory.h"],
    deps = [
        ":checkpoint_writer",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@org_tensorflow//tensorflow/core/platform:env",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_test(
    name = "tensorflow_checkpoint_builder_factory_test",
    srcs = ["tensorflow_checkpoint_builder_factory_test.cc"],
    deps = [
        ":tensorflow_checkpoint_builder_factory",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "tensorflow_checkpoint_parser_factory",
    srcs = ["tensorflow_checkpoint_parser_factory.cc"],
    hdrs = ["tensorflow_checkpoint_parser_factory.h"],
    deps = [
        ":checkpoint_reader",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@org_tensorflow//tensorflow/core/platform:env",
    ],
)

cc_test(
    name = "tensorflow_checkpoint_parser_factory_test",
    srcs = ["tensorflow_checkpoint_parser_factory_test.cc"],
    deps = [
        ":tensorflow_checkpoint_parser_factory",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)
