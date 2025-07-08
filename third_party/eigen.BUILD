package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

cc_library(
    name = "eigen",
    hdrs = glob([
        "Eigen/**",
        "unsupported/Eigen/**",
    ]),
    defines = [
        "EIGEN_MAX_ALIGN_BYTES=64",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
