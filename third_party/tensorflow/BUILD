package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

exports_files([
    "tf_deps.bzl",
    "internal_visibility.patch",
    "fix_pywrap_win_def.patch",
    "exports.patch",
])

config_setting(
    name = "tff_pip_build",
    define_values = {"tff_pip_build": "true"},
    visibility = ["//visibility:public"],
)
