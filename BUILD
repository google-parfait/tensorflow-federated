load("@rules_license//rules:license.bzl", "license")
# load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(
    default_applicable_licenses = [":package_license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "package_license",
    package_name = "tensorflow_federated",
    license_kinds = ["@rules_license//licenses/spdx:Apache-2.0"],
)

licenses(["notice"])

exports_files(["LICENSE"])

# compile_pip_requirements(
#     name = "requirements",
#     src = "requirements.in",
#     extra_args = [
#         "--allow-unsafe",
#         "--resolver=backtracking",
#     ],
#     requirements_txt = "requirements.txt",
# )
