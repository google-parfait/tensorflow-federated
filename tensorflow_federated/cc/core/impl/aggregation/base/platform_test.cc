/*
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/base/platform.h"

#include "googletest/include/gtest/gtest.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/base_name.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {

namespace {

TEST(PlatformTest, ConcatPath) {
  auto combined = ConcatPath("first", "second");
#if _WIN32
  ASSERT_EQ(combined, "first\\second");
#else
  ASSERT_EQ(combined, "first/second");
#endif
}

TEST(PlatformTest, StripTrailingPathSeparator) {
#if _WIN32
  ASSERT_EQ(StripTrailingPathSeparator("path\\"), "path");
  ASSERT_EQ(StripTrailingPathSeparator("dir/path"), "dir/path");
#else
  ASSERT_EQ(StripTrailingPathSeparator("path/"), "path");
  ASSERT_EQ(StripTrailingPathSeparator("dir/path"), "dir/path");
#endif
}

TEST(PlatformTest, ReadWriteString) {
  auto file = aggregation::TemporaryTestFile(".dat");
  ASSERT_EQ(WriteStringToFile(file, "Ein Text").code(), OK);
  auto status_or_string = ReadFileToString(file);
  ASSERT_TRUE(status_or_string.ok()) << status_or_string.status();
  ASSERT_EQ(status_or_string.value(), "Ein Text");
}

TEST(PlatformTest, ReadWriteCord) {
  auto file = aggregation::TemporaryTestFile(".dat");
  // Make cord with two chunks.
  absl::Cord content("Ein");
  content.Append(" Text");
  ASSERT_EQ(WriteCordToFile(file, content).code(), OK);
  auto status_or_cord = ReadFileToCord(file);
  ASSERT_TRUE(status_or_cord.ok()) << status_or_cord.status();
  ASSERT_EQ(status_or_cord.value(), "Ein Text");
}

TEST(PlatformTest, ReadStringFails) {
  ASSERT_FALSE(ReadFileToString("foobarbaz").ok());
}

TEST(PlatformTest, ReadCordFails) {
  ASSERT_FALSE(ReadFileToCord("foobarbaz").ok());
}

TEST(PlatformTest, BaseName) {
  ASSERT_EQ(BaseName(ConcatPath("foo", "bar.x")), "bar.x");
}

TEST(PlatformTest, FileExists) {
  auto file = aggregation::TemporaryTestFile(".dat");
  ASSERT_EQ(WriteStringToFile(file, "Ein Text").code(), OK);
  ASSERT_TRUE(FileExists(file));
}

TEST(PlatformTest, FileExistsNot) { ASSERT_FALSE(FileExists("foobarbaz")); }

}  // namespace

}  // namespace tensorflow_federated
