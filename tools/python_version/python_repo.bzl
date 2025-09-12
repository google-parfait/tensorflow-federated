"""
Repository rule to set python version.
Can be set via build parameter "--repo_env=TF_PYTHON_VERSION=3.10"
Defaults to 3.10.
"""

VERSIONS = ["3.10", "3.11"]
DEFAULT_VERSION = "3.10"

def _python_repository_impl(repository_ctx):
    repository_ctx.file("BUILD", "")
    version = repository_ctx.os.environ.get("TF_PYTHON_VERSION", "")
    if version == "":
        print("TF_PYTHON_VERSION not set, defaulting to 3.10")
        version=DEFAULT_VERSION
    elif version not in VERSIONS:
        fail("Error: Specified Python version '%s' is not supported. Valid versions are: %s" % (
            version, ", ".join(VERSIONS)))
    python_tag="cp"+version.replace(".", "")
    abi_tag="cp"+version.replace(".", "")

    repository_ctx.file(
        "py_version.bzl",
        "HERMETIC_PYTHON_VERSION = \"%s\"\nPYTHON_TAG = \"%s\"\nABI_TAG = \"%s\"\nHOST_OS = \"%s\"" %
        (version, python_tag, abi_tag, repository_ctx.os.name),
)

python_repository = repository_rule(
    implementation = _python_repository_impl,
    environ = ["TF_PYTHON_VERSION"],
)
