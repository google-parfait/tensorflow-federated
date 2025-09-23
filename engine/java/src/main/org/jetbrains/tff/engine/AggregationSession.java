// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package org.jetbrains.tff.engine;

import static com.google.common.base.Preconditions.checkArgument;

/// A simple wrapper around the checkpoint aggregator.
public final class AggregationSession implements AutoCloseable {
  /// Creates a new session, based on the plan.
  public static AggregationSession createFromByteArray(byte[] plan) {
    try {
      byte[] configuration = extractConfiguration(plan);
      long sessionHandle = createNativeFromByteArray(configuration);
      return new AggregationSession(sessionHandle, configuration);
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /// Accumulates the list of checkpoint via nested tensor aggregators in memory.
  public void accumulate(byte[][] checkpoints) {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      runAccumulate(scopedHandle.get(), checkpoints);
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /// Merges the list of serialized aggregators in memory.
  public void mergeWith(byte[][] serialized) {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      mergeWith(scopedHandle.get(), configuration, serialized);
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /// Serialized aggregator in memory to a byte[].
  public byte[] serialize() {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      return serialize(scopedHandle.get());
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /// Builds a report from the session.
  public byte[] report() {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      return runReport(scopedHandle.get());
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /// Safety net finalizer for cleanup of the wrapped native resource.
  @Override
  protected void finalize() throws Throwable {
    assert !sessionHandle.isValid();
  }

  private final NativeHandle sessionHandle;
  private final byte[] configuration;

  private AggregationSession(long handle, byte[] configuration) {
    checkArgument(handle != 0);
    this.sessionHandle = new NativeHandle(handle);
    this.configuration = configuration;
  }

  /// Exception handlers that catch AggregationException should call this method in order to convert
  /// them to a generic IllegalStateException.
  private static IllegalStateException onAggregationException(AggregationException e) {
    return new IllegalStateException("Native aggregation session exception", e);
  }

  /// Closes the session, releasing resources. This must be run in the same thread as create.
  /// @throws IllegalStateException with a wrapped AggregationException if closing was not
  /// successful.
  @Override
  public void close() {
    if (!sessionHandle.isValid()) {
      return;
    }
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      closeNative(scopedHandle.release());
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  static {
    NativeLibraryLoader.getInstance().loadLibrary();
  }

  // Native API
  // ==========

  /// Starts a session based on the given plan, returning a handle for it. */
  static native long createNativeFromByteArray(byte[] plan) throws AggregationException;

  /// CAREFUL: don't make the following native calls static because it can cause a race condition
  /// between the native execution and the object finalize() call.

  /// Closes the session. The handle is not usable afterwards.
  native void closeNative(long session) throws AggregationException;

  /// Accumulates the provided checkpoint using the native session.
  native void runAccumulate(long session, byte[][] checkpoints) throws AggregationException;

  /// Merges the serialized aggregator using the native session.
  native void mergeWith(long session, byte[] configuration, byte[][] serialized)
      throws AggregationException;

  /// Serializes the internal state of the aggregator using the native session.
  native byte[] serialize(long session) throws AggregationException;

  /// Creates a report using the native session.
  native byte[] runReport(long session) throws AggregationException;

  /// Extracts the configuration from the plan.
  static native byte[] extractConfiguration(byte[] plan) throws AggregationException;
}
