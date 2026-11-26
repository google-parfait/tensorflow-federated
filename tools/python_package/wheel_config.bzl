# Copyright 2026, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Repository rule parsing pyproject.toml and version.py for py_wheel."""

_PROJECT_SECTION = "[project]"
_PROJECT_URLS_SECTION = "[project.urls]"

def _strip_toml_comment(line):
    """Strips an unquoted trailing '#' comment from a TOML line."""
    in_quote = ""
    escaped = False
    for idx in range(len(line)):
        ch = line[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote == '"':
            escaped = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = ""
            continue
        if ch in ("'", '"'):
            in_quote = ch
        elif ch == "#":
            return line[:idx]
    return line

def _unquote(raw):
    """Removes surrounding TOML single or double quotes from a scalar."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        inner = text[1:-1]
        if text[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    return text

def _split_top_level(text, sep):
    """Splits text on sep outside quotes and nested brackets/braces."""
    parts = []
    start = 0
    in_quote = ""
    escaped = False
    depth = 0
    for idx in range(len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote == '"':
            escaped = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = ""
            continue
        if ch in ("'", '"'):
            in_quote = ch
        elif ch in ("[", "{"):
            depth += 1
        elif ch in ("]", "}"):
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts

def _parse_list_items(raw_list):
    """Parses a TOML '[...]' literal into stripped non-empty raw elements."""
    text = raw_list.strip()
    if not (text.startswith("[") and text.endswith("]")):
        fail("Expected TOML list literal, got: %r" % raw_list)
    inner = text[1:-1].strip()
    if not inner:
        return []
    items = []
    for part in _split_top_level(inner, ","):
        item = part.strip()
        if item:
            items.append(item)
    return items

def _parse_inline_table(raw_table):
    """Parses a TOML '{ k = v, ... }' inline table into a string dict."""
    text = raw_table.strip()
    if not (text.startswith("{") and text.endswith("}")):
        fail("Expected TOML inline table, got: %r" % raw_table)
    result = {}
    inner = text[1:-1].strip()
    if not inner:
        return result
    for entry in _split_top_level(inner, ","):
        pair = entry.strip()
        if not pair:
            continue
        if "=" not in pair:
            fail("Expected key = value in inline table entry: %r" % pair)
        key, val = pair.split("=", 1)
        result[_unquote(key)] = _unquote(val)
    return result

def _bracket_delta(text):
    """Returns net '[' minus ']' balance outside quoted strings."""
    delta = 0
    in_quote = ""
    escaped = False
    for idx in range(len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote == '"':
            escaped = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = ""
            continue
        if ch in ("'", '"'):
            in_quote = ch
        elif ch == "[":
            delta += 1
        elif ch == "]":
            delta -= 1
    return delta

def _parse_version(content):
    """Extracts __version__ from version.py."""
    for line in content.splitlines():
        stripped = _strip_toml_comment(line).strip()
        if stripped.startswith("__version__") and "=" in stripped:
            _, rhs = stripped.split("=", 1)
            version = _unquote(rhs)
            if version:
                return version
    fail("Could not find __version__ in version.py.")

def _parse_pyproject(content):
    """Parses PEP 621 [project] and [project.urls] metadata from pyproject.toml."""
    section = ""
    project = {}
    urls = {}
    pending_key = ""
    pending_buf = []
    pending_depth = 0

    for raw_line in content.splitlines():
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue

        if pending_key:
            pending_buf.append(line)
            pending_depth += _bracket_delta(line)
            if pending_depth <= 0:
                project[pending_key] = " ".join(pending_buf)
                pending_key = ""
                pending_buf = []
                pending_depth = 0
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line
            continue

        if "=" not in line:
            continue

        raw_key, raw_val = line.split("=", 1)
        key = _unquote(raw_key)
        val = raw_val.strip()

        if section == _PROJECT_SECTION:
            if val.startswith("[") and _bracket_delta(val) > 0:
                pending_key = key
                pending_buf = [val]
                pending_depth = _bracket_delta(val)
            else:
                project[key] = val
        elif section == _PROJECT_URLS_SECTION:
            urls[key] = _unquote(val)

    if pending_key:
        fail("Unterminated TOML list for key %r in pyproject.toml." % pending_key)

    name = _unquote(project.get("name", ""))
    description = _unquote(project.get("description", ""))
    requires_python = _unquote(project.get("requires-python", ""))
    keywords = [_unquote(x) for x in _parse_list_items(project.get("keywords", "[]"))]
    classifiers = [
        _unquote(x)
        for x in _parse_list_items(project.get("classifiers", "[]"))
    ]
    dependencies = [
        _unquote(x)
        for x in _parse_list_items(project.get("dependencies", "[]"))
    ]
    authors = [
        _parse_inline_table(x)
        for x in _parse_list_items(project.get("authors", "[]"))
    ]
    author_name = authors[0].get("name", "") if authors else ""
    author_email = authors[0].get("email", "") if authors else ""

    required = {
        "name": name,
        "description": description,
        "requires-python": requires_python,
        "authors.name": author_name,
        "authors.email": author_email,
        "keywords": keywords,
        "classifiers": classifiers,
        "dependencies": dependencies,
        "project.urls": urls,
    }
    for field_name, field_value in required.items():
        if not field_value:
            fail("Missing required field %r in pyproject.toml." % field_name)

    return {
        "name": name,
        "description": description,
        "requires_python": requires_python,
        "author_name": author_name,
        "author_email": author_email,
        "keywords": keywords,
        "classifiers": classifiers,
        "dependencies": dependencies,
        "urls": urls,
    }

def _tff_wheel_config_impl(ctx):
    version = _parse_version(ctx.read(ctx.path(ctx.attr.version_py)))
    meta = _parse_pyproject(ctx.read(ctx.path(ctx.attr.pyproject_toml)))
    summary_with_keywords = (
        meta["description"] + "\nKeywords: " + ",".join(meta["keywords"])
    )

    ctx.file("BUILD.bazel", "")
    ctx.file(
        "config.bzl",
        "\n".join([
            "# Generated by tff_wheel_config from pyproject.toml and version.py.",
            "DISTRIBUTION = %r" % meta["name"],
            "VERSION = %r" % version,
            "SUMMARY_WITH_KEYWORDS = %r" % summary_with_keywords,
            "PYTHON_REQUIRES = %r" % meta["requires_python"],
            "REQUIRES = %r" % meta["dependencies"],
            "CLASSIFIERS = %r" % meta["classifiers"],
            "PROJECT_URLS = %r" % meta["urls"],
            "AUTHOR = %r" % meta["author_name"],
            "AUTHOR_EMAIL = %r" % meta["author_email"],
            "",
        ]),
    )

tff_wheel_config = repository_rule(
    implementation = _tff_wheel_config_impl,
    attrs = {
        "pyproject_toml": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "version_py": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
)
