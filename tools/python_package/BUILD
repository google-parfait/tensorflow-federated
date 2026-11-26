load("@rules_python//python:defs.bzl", "py_test")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load(
    "@tff_wheel_config//:config.bzl",
    "AUTHOR",
    "AUTHOR_EMAIL",
    "CLASSIFIERS",
    "DISTRIBUTION",
    "PROJECT_URLS",
    "PYTHON_REQUIRES",
    "REQUIRES",
    "SUMMARY_WITH_KEYWORDS",
    "VERSION",
)
load(":wheel_staging.bzl", "strip_wheel_extensions")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "python_package_tool",
    packages = ["//tools/python_package/..."],
)

licenses(["notice"])

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

PYTHON_TAG = select({
    "@rules_python//python/config_settings:is_python_3.12": "cp312",
    "@rules_python//python/config_settings:is_python_3.13": "cp313",
})

# The package labels are needed so Bazel includes the `__init__.py` files.
# Modules are imported directly rather than via parent packages, so targets in
# the dependency graph do not depend on these `__init__.py` libraries.
py_package(
    name = "wheel_pkg",
    packages = ["tensorflow_federated"],
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/cc/core/impl/executors:executor_bindings",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_bindings",
        "//tensorflow_federated/proto",
        "//tensorflow_federated/proto/v0",
        "//tensorflow_federated/python",
        "//tensorflow_federated/python/analytics/hierarchical_histogram",
        "//tensorflow_federated/python/common_libs",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/core/environments",
        "//tensorflow_federated/python/core/environments/tensorflow_backend",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend",
        "//tensorflow_federated/python/core/impl",
        "//tensorflow_federated/python/core/impl/compiler",
        "//tensorflow_federated/python/core/impl/execution_contexts",
        "//tensorflow_federated/python/core/impl/executor_stacks",
        "//tensorflow_federated/python/core/impl/executors",
    ],
)

strip_wheel_extensions(
    name = "stripped_wheel_pkg",
    src = ":wheel_pkg",
    is_macos = select({
        "@platforms//os:macos": True,
        "//conditions:default": False,
    }),
)

py_wheel(
    name = "wheel",
    abi = PYTHON_TAG,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    classifiers = CLASSIFIERS,
    description_content_type = "text/markdown",
    description_file = "//:README.md",
    distribution = DISTRIBUTION,
    extra_distinfo_files = {
        "//:LICENSE": "LICENSE",
    },
    platform = select({
        ":linux_x86_64": "manylinux_2_28_x86_64",
        ":linux_aarch64": "manylinux_2_28_aarch64",
        ":macos_arm64": "macosx_12_0_arm64",
        ":macos_x86_64": "macosx_10_9_x86_64",
    }),
    project_urls = PROJECT_URLS,
    python_requires = PYTHON_REQUIRES,
    python_tag = PYTHON_TAG,
    requires = REQUIRES,
    strip_path_prefixes = ["tools/python_package/stripped_pkg"],
    summary = SUMMARY_WITH_KEYWORDS,
    tags = ["manual"],
    twine_binary = None,
    version = VERSION,
    deps = [":stripped_wheel_pkg"],
)

py_test(
    name = "wheel_test",
    srcs = ["wheel_test.py"],
    args = [
        "--wheel=$(rootpath :wheel)",
        "--pyproject=$(rootpath //:pyproject_toml)",
        "--readme=$(rootpath //:README.md)",
        "--version_file=$(rootpath //tensorflow_federated:version.py)",
        "--test_computation=$(rootpath :test_computation.py)",
        "--verify_script=$(rootpath :verify_python_package.py)",
    ],
    data = [
        ":test_computation.py",
        ":verify_python_package.py",
        ":wheel",
        "//:README.md",
        "//:pyproject_toml",
        "//tensorflow_federated:version.py",
    ],
    tags = ["manual"],
    deps = [
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//cachetools",
        "@pypi//dm_tree",
        "@pypi//dp_accounting",
        "@pypi//federated_language",
        "@pypi//grpcio",
        "@pypi//ml_dtypes",
        "@pypi//numpy",
        "@pypi//portpicker",
        "@pypi//scipy",
        "@pypi//tensorflow",
        "@pypi//tqdm",
    ],
)

sh_binary(
    name = "test_python_package",
    srcs = ["test_python_package.sh"],
    data = [
        "get_hermetic_python.sh",
        "test_computation.py",
        "verify_python_package.py",
        "@rules_python//python:current_py_toolchain",
    ],
    deps = ["@bazel_tools//tools/bash/runfiles"],
)
