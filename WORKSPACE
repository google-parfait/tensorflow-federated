# Dependencies for the TensorFlow Federated bazel environment.
#
# If you add a new or update a repository, follow these guidelines:
#
# *   Repositories must be deterministic (i.e., not a branch).
# *   Prefer to use an `http_archive` rule with a `sha256` parameter.
# *   Prefer for the repository to be a released product (i.e., a tag).
# *   Configuration must be documented when a commit is used instead of a
#     released product.

workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

#
# Direct dependencies
#

http_archive(
    name = "platforms",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
    ],
    sha256 = "3384eb1c30762704fbe38e440204e114154086c8fc8a8c2e3e28441028c019a8",
)

# Warning:
# for nsync compilation you have to install g++: apt-get update && apt-get install -y g++

http_archive(
    name = "bazel_skylib",
    sha256 = "3b620033ca48fcd6f5ef2ac85e0f6ec5639605fa2f627968490e52fc91a9932f",
    strip_prefix = "bazel-skylib-1.3.0",
    url = "https://github.com/bazelbuild/bazel-skylib/archive/refs/tags/1.3.0.tar.gz",
)

http_archive(
    name = "rules_cc",
    sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6",
    strip_prefix = "rules_cc-0.1.5",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz",
)

# http_archive(
#     name = "com_github_grpc_grpc",
#     sha256 = "76900ab068da86378395a8e125b5cc43dfae671e09ff6462ddfef18676e2165a",
#     strip_prefix = "grpc-1.50.0",
#     url = "https://github.com/grpc/grpc/archive/refs/tags/v1.50.0.tar.gz",
# )

http_archive(
    name = "com_github_grpc_grpc",
    patches = [
        # Enable dynamic_lookup linkopt on macOS
        "//third_party/grpc:dynamic_lookup.patch",
    ],
    sha256 = "76900ab068da86378395a8e125b5cc43dfae671e09ff6462ddfef18676e2165a",
    strip_prefix = "grpc-1.50.0",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.50.0.tar.gz"],
)

http_archive(
    name = "com_google_benchmark",
    sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce",
    strip_prefix = "benchmark-1.8.3",
    url = "https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz",
)

http_archive(
    name = "com_google_cc_differential_privacy",
    sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
    strip_prefix = "differential-privacy-3.0.0/cc",
    url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
)

http_archive(
    name = "com_google_differential_privacy",
    sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
    strip_prefix = "differential-privacy-3.0.0",
    url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
)

# Register a clang C++ toolchain.
git_repository(
    name = "toolchains_llvm",
    remote = "https://github.com/bazel-contrib/toolchains_llvm.git",
    tag = "1.0.0",
)

load("@toolchains_llvm//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_versions = {
        "": "15.0.6",
        "linux-aarch64": "15.0.6",
        "darwin-aarch64": "15.0.7",
    },
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()

# This commit is determined by
# https://github.com/tensorflow/tensorflow/blob/v2.16.1/third_party/absl/workspace.bzl#L10.
http_archive(
    name = "com_google_absl",
    sha256 = "0320586856674d16b0b7a4d4afb22151bdc798490bb7f295eddd8f6a62b46fea",
    strip_prefix = "abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30",
    url = "https://github.com/abseil/abseil-cpp/archive/fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz",
)

http_archive(
    name = "com_google_googletest",
    sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
    strip_prefix = "googletest-release-1.12.1",
    url = "https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz",
)

http_archive(
    name = "eigen",
    build_file = "//third_party:eigen.BUILD",
    sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
    strip_prefix = "eigen-4c38131a16803130b66266a912029504f2cf23cd",
    url = "https://gitlab.com/libeigen/eigen/-/archive/4c38131a16803130b66266a912029504f2cf23cd/eigen-4c38131a16803130b66266a912029504f2cf23cd.tar.gz",
)

http_archive(
    name = "federated_language",
    patches = [
        "//third_party/federated_language:numpy.patch",
        "//third_party/federated_language:proto_library_loads.patch",
        "//third_party/federated_language:python_deps.patch",
        # Must come after `python_deps.patch`, this patches the output of `python_deps.patch`.
        "//third_party/federated_language:structure_visibility.patch",
    ],
    repo_mapping = {
        "@protobuf": "@com_google_protobuf",
    },
    sha256 = "51e13f9ce23c9886f34e20c5f4bd7941b6335867405d3b4f7cbc704d6f89e820",
    strip_prefix = "federated-language-16e734b633e68b613bb92918e6f3304774853e9b",
    url = "https://github.com/google-parfait/federated-language/archive/16e734b633e68b613bb92918e6f3304774853e9b.tar.gz",
)

http_archive(
    name = "rules_java",
    sha256 = "c7bd858a132c7b8febe040a90fa138c2e3e7f0bce47122ac2ad43906036a276c",
    urls = [
        "https://github.com/bazelbuild/rules_java/releases/download/8.3.0/rules_java-8.3.0.tar.gz",
    ],
)

load("@rules_java//java:repositories.bzl", "rules_java_dependencies", "rules_java_toolchains")

rules_java_dependencies()

rules_java_toolchains()

RULES_JVM_EXTERNAL_TAG = "6.0"

RULES_JVM_EXTERNAL_SHA = "85fd6bad58ac76cc3a27c8e051e4255ff9ccd8c92ba879670d195622e7c0a9b7"

http_archive(
    name = "rules_jvm_external",
    sha256 = RULES_JVM_EXTERNAL_SHA,
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    url = "https://github.com/bazelbuild/rules_jvm_external/releases/download/%s/rules_jvm_external-%s.tar.gz" % (RULES_JVM_EXTERNAL_TAG, RULES_JVM_EXTERNAL_TAG),
)

load("@rules_jvm_external//:repositories.bzl", "rules_jvm_external_deps")

rules_jvm_external_deps()

load("@rules_jvm_external//:setup.bzl", "rules_jvm_external_setup")

rules_jvm_external_setup()

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    name = "maven",
    artifacts = [
        "com.google.guava:guava:32.1.2-jre",
        "com.google.code.findbugs:jsr305:3.0.2",
        "junit:junit:4.13.2",
    ],
    repositories = [
        "https://repo1.maven.org/maven2",
    ],
)


# The version of TensorFlow should match the version in
# https://github.com/google-parfait/tensorflow-federated/blob/main/requirements.txt.
http_archive(
    name = "org_tensorflow",
    patches = [
        # This patch enables googleapi Java and Python proto rules such as
        # @com_google_googleapis//google/rpc:rpc_java_proto.
        "//third_party/tensorflow:googleapis_proto_rules.patch",
        # This patch works around failures in GitHub infrastructure to
        # download versions of LLVM pointed to by non-HEAD TensorFlow.
        # TODO(team): Remove this patch when resolved.
        "//third_party/tensorflow:llvm_url.patch",
        # The version of googleapis imported by TensorFlow 2.14 doesn't provide
        # `py_proto_library` targets for //google/longrunning.
        "//third_party/tensorflow:googleapis.patch",
        # This patch replaces tf_gen_op_wrapper_py's dependency on @tensorflow
        # with @pypi_tensorflow.
        "//third_party/tensorflow:tf_gen_op_wrapper_py.patch",
        # gRPC v1.48.0-pre1 and later include zconf.h in addition to zlib.h;
        # TensorFlow's build rule for zlib only exports the latter.
        "//third_party/tensorflow:zlib.patch",
        # These patches enables building TensorFlow Federated from source by
        # fixing visibility in TensorFlow.
        "//third_party/tensorflow:internal_visibility.patch",
        "//third_party/tensorflow:tf2xla_visibility.patch",

        "//third_party/tensorflow:snappy_include.patch",
        # allow self-signed certificates on macos (otherwise SecureTransport will be used which does not allow that)
        "//third_party/tensorflow:curl_ssl_macos.patch",


        # "//third_party/tensorflow:python_toolchain.patch",

    ],
    sha256 = "ce357fd0728f0d1b0831d1653f475591662ec5bca736a94ff789e6b1944df19f",
    strip_prefix = "tensorflow-2.14.0",
    url = "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.14.0.tar.gz",
)

load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Update protobug from TensorFlow to enable dynamic_lookup on macOS for Bazel 7
# Check https://github.com/protocolbuffers/protobuf/pull/19785/files
tf_http_archive(
    name = "com_google_protobuf",
    patch_file = [
        "@org_tensorflow//third_party/protobuf:protobuf.patch",
        # Enable dynamic_lookup linkopt on macOS
        "@org_tensorflow_federated//third_party/protobuf:dynamic_lookup_macos.patch",
    ],
    sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
    strip_prefix = "protobuf-3.21.9",
    system_build_file = "@org_tensorflow//third_party/systemlibs:protobuf.BUILD",
    system_link_files = {
        "@org_tensorflow///third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        "@org_tensorflow///third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
    },
    urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip"),
)

# This commit is determined by
# https://github.com/tensorflow/tensorflow/blob/v2.16.1/third_party/pybind11_abseil/workspace.bzl#L11.
http_archive(
    name = "pybind11_abseil",
    sha256 = "0223b647b8cc817336a51e787980ebc299c8d5e64c069829bf34b69d72337449",
    strip_prefix = "pybind11_abseil-2c4932ed6f6204f1656e245838f4f5eae69d2e29",
    url = "https://github.com/pybind/pybind11_abseil/archive/2c4932ed6f6204f1656e245838f4f5eae69d2e29.tar.gz",
)

tf_http_archive(
    name = "pybind11_bazel",
    patch_file = [
        "@org_tensorflow//third_party/pybind11_bazel:pybind11_bazel.patch",
        # Enable dynamic_lookup linkopt on macOS
        "@org_tensorflow_federated//third_party/pybind11_bazel:dynamic_lookup.patch",
    ],
    sha256 = "516c1b3a10d87740d2b7de6f121f8e19dde2c372ecbfe59aef44cd1872c10395",
    strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
    urls = tf_mirror_urls("https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.tar.gz"),
)

# This commit is determined by
# https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/workspace2.bzl#L788.
http_archive(
    name = "pybind11_protobuf",
    sha256 = "ba2c54a8b4d1dd0a68c58159e37b1f863c0d9d1dc815558288195493bcc31682",
    strip_prefix = "pybind11_protobuf-80f3440cd8fee124e077e2e47a8a17b78b451363",
    url = "https://github.com/pybind/pybind11_protobuf/archive/80f3440cd8fee124e077e2e47a8a17b78b451363.tar.gz",
)

http_archive(
    name = "rules_license",
    sha256 = "8c1155797cb5f5697ea8c6eac6c154cf51aa020e368813d9d9b949558c84f2da",
    strip_prefix = "rules_license-0.0.8",
    url = "https://github.com/bazelbuild/rules_license/archive/refs/tags/0.0.8.tar.gz",
)

# Python rules for Bazel
http_archive(
    name = "rules_python",
    patches = [
        "//third_party/patches:py_package.patch",
    ],
    sha256 = "690e0141724abb568267e003c7b6d9a54925df40c275a870a4d934161dc9dd53",
    strip_prefix = "rules_python-0.40.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.40.0/rules_python-0.40.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

load(
    "//tools/python_version:python_repo.bzl",
    "python_repository",
)

python_repository(name = "python_version_repo")

load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")

python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    minor_mapping = {
        "3.10": "3.10.11",
        "3.11": "3.11.6",
    },
    python_version = HERMETIC_PYTHON_VERSION,
)

#
# Inlined transitive dependencies, grouped by direct dependency.
#

# Required by pybind11_abseil and pybind11_protobuf.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1",
    strip_prefix = "pybind11-2.9.2",
    url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz",
)

# Required by com_github_grpc_grpc. This commit is determined by
# https://github.com/grpc/grpc/blob/v1.50.0/bazel/grpc_deps.bzl#L344.
http_archive(
    name = "upb",
    sha256 = "017a7e8e4e842d01dba5dc8aa316323eee080cd1b75986a7d1f94d87220e6502",
    strip_prefix = "upb-e4635f223e7d36dfbea3b722a4ca4807a7e882e2",
    url = "https://github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
)

#
# Transitive dependencies, grouped by direct dependency.
#

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1(with_rules_cc = False)

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Unable to load the `grpc_extra_deps` from gRPC because they conflict with the
# dependencies required by TensorFlow.
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# Engine deps
http_archive(
    name = "nlohmann_json",
    url = "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz",
    strip_prefix = "json",
    build_file_content = """
cc_library(
    name = "json",
    hdrs = glob(["single_include/nlohmann/*.hpp"]),
    includes = ["single_include"],
    visibility = ["//visibility:public"],
)
""",
)
