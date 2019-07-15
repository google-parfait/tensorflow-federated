"""Helpers to check version of bazel."""

def _extract_version_number(bazel_version):
    """Extracts the semantic version number from a version string

    Args:
      bazel_version: the version string that begins with the semantic version
        e.g. "1.2.3rc1 abc1234" where "abc1234" is a commit hash.

    Returns:
      The semantic version string, like "1.2.3".
    """
    for i in range(len(bazel_version)):
        c = bazel_version[i]
        if not (c.isdigit() or c == "."):
            return bazel_version[:i]
    return bazel_version

# Parse the bazel version string from `native.bazel_version`.
# e.g.
# "0.10.0rc1 abc123d" => (0, 10, 0)
# "0.3.0" => (0, 3, 0)
def _parse_bazel_version(bazel_version):
    """Parses a version string into a 3-tuple of ints

    int tuples can be compared directly using binary operators (<, >).

    Args:
      bazel_version: the Bazel version string

    Returns:
      An int 3-tuple of a (major, minor, patch) version.
    """
    version = _extract_version_number(bazel_version)
    return tuple([int(n) for n in version.split(".")])

def check_bazel_version_equal(bazel_version):
    """Checks bazel version is equal to `bazel_version.`

    Args:
        bazel_version: Semantic version string documenting Bazel version to
          use.
    """
    if "bazel_version" not in dir(native):
        fail("\nCurrent Bazel version is lower than 0.2.1, expected %s\n" % bazel_version)
    elif not native.bazel_version:
        fail(("\nCurrent Bazel is not a release version, cannot check for " +
              "compatibility. Make sure that you are running at Bazel %s.\n") % bazel_version)

    if _parse_bazel_version(native.bazel_version) != _parse_bazel_version(bazel_version):
        fail("\nCurrent Bazel version is {}, expected {}\n".format(
            native.bazel_version,
            bazel_version,
        ))

parse_bazel_version = _parse_bazel_version
