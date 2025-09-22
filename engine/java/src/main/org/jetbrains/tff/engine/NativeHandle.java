// Copyright 2023 Google LLC
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

import com.google.common.base.Preconditions;
import java.util.concurrent.locks.ReentrantLock;

/**
 * This class is used to wrap a native handle used with JNI and guard access to a native object that
 * the native handle represents.
 * NativeHandle attempts to enforce the following assumptions:
 * 1) Native calls to the same native object cannot be made concurrently from multiple threads.
 * 2) Native handle cannot be resolved (and the underlying native object can't be destroyed) while it
 *    is used in an ongoing native call.
 * 3) Native handle cannot be used after it has been resolved.
 *
 * <p>Creating native object and storing the handle:
 *
 * <pre>{@code
 * nativeHandle = new NativeHandle(createNativeObjectXyz());
 * }</pre>
 *
 * <p>Making native call requires acquiring the handle in order to get the
 * underlying handle value:
 *
 * <pre>{@code
 * try (NativeHandle.ScopedHandle scopedHandle = nativeHandle.acquire()) {
 *   callNativeMethodXyz(scopedHandle.get());
 * }
 * }</pre>
 *
 * <p>Destroying native object requires acquiring the handle too and releasing the underlying
 * handle value:
 *
 * <pre>{@code
 * try (NativeHandle.ScopedHandle scopedHandle = nativeHandle.acquire()) {
 *   destroyNativeObjectXyz(scopedHandle.release());
 * }
 * }</pre>
 */
public final class NativeHandle {

  /**
   * This lock is used to disallow concurrent native calls with this handle.
   */
  private final ReentrantLock lock = new ReentrantLock();

  /**
   * Address of the native object
   */
  private volatile long handle = 0;

  public NativeHandle(long handle) {
    Preconditions.checkState(handle != 0);
    this.handle = handle;
  }

  /**
   * Returns true if NativeHandle has been initialized with a valid handle; false once the native
   * handle has been released.
   */
  public boolean isValid() {
    lock.lock();
    try {
      return handle != 0;
    } finally {
      lock.unlock();
    }
  }

  /**
   * Returns true when the native handle has been locked by the current thread, which means the code
   * calling this method should be within the try-with-resource statement.
   */
  public boolean isOwnedByCurrentThread() {
    return lock.isHeldByCurrentThread();
  }

  /**
   * Acquire native handle for a native call
   */
  public ScopedHandle acquire() {
    return new ScopedHandle();
  }

  /**
   * Implements auto-closable semantics for the lock to be used in try-with-resource statements and
   * guards the native handle by locking in the constructor and unlocking in the automatically
   * called close() method.
   */
  public class ScopedHandle implements AutoCloseable {

    private boolean isClosed = false;

    ScopedHandle() {
      Preconditions.checkState(
          lock.tryLock(), "Attempting to make concurrent calls with native handle");
      Preconditions.checkState(isValid(), "Native handle has already been destroyed");
    }

    private void checkAcquiredAndValid() {
      Preconditions.checkState(
          isOwnedByCurrentThread() && !isClosed,
          "Can't use native handle after ScopedHandle has already been closed");
      Preconditions.checkState(isValid(), "Native handle has already been destroyed");
    }

    /**
     * Implementation of AutoCloseable.close(), unlocks the {@link NativeHandle.lock}.
     */
    @Override
    public void close() {
      isClosed = true;
      lock.unlock();
    }

    /**
     * Gets the underlying native handle value.
     */
    public long get() {
      checkAcquiredAndValid();
      return handle;
    }

    /**
     * Releases the underlying native handle value from {@link NativeHandle} ownership and resets
     * the NativeHandle to a disposed state.
     */
    public long release() {
      checkAcquiredAndValid();
      long oldHandle = handle;
      handle = 0;
      return oldHandle;
    }
  }
}
