# Description:
#   Six provides simple utilities for wrapping over differences between Python 2
#   and Python 3.

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

# Rename `six.py` to `__init__.py`.
#
# Without this, code building under bazel that depends on the PIP-installed
# version of `six` will pull in the empty `__init__.py` rather than `six.py`,
# resulting in errors like "No module named 'six.moves'" or
# "module 'six' has no attribute 'PY3'".
genrule(
    name = "rename",
    srcs = ["six.py"],
    outs = ["__init__.py"],
    cmd = "cat $< >$@",
)

py_library(
    name = "six",
    srcs = [":__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
