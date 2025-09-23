// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
package org.jetbrains.tff.engine;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.atomic.AtomicReference;

/**
 * This class is used to load the Engine TFF Aggregation shared library from within the jar.
 * The shared library is extracted to a temp folder and loaded from there.
 */
public class NativeLibraryLoader {
  //singleton
  private static final NativeLibraryLoader instance = new NativeLibraryLoader();
  private static final String LIBRARY_NAME = "libaggregation-jni";
  private static final String sharedLibraryName = LIBRARY_NAME + getOsExtension();

  private enum LibraryState {
    NOT_LOADED,
    LOADING,
    LOADED
  }

  private static final AtomicReference<LibraryState> libraryLoaded =
      new AtomicReference<>(LibraryState.NOT_LOADED);

  private static final String getOsExtension() {
    String os = System.getProperty("os.name").toLowerCase();
    if (os.contains("mac")) {
      return ".dylib";
    } else if (os.contains("nux") || os.contains("nix")) {
      return ".so";
    } else {
      throw new UnsupportedOperationException("Unsupported OS: " + os);
    }
  }

  private static boolean DEBUG_LOADING = "true".equals(System.getProperty("ENGINE_TFF_JAVA_DEBUG_NLL", "false"));

  public static NativeLibraryLoader getInstance() {
    return instance;
  }

  public void loadLibrary() {
    if (libraryLoaded.get() == LibraryState.LOADED) {
      return;
    }

    if (libraryLoaded.compareAndSet(LibraryState.NOT_LOADED, LibraryState.LOADING)) {
      final String tmpDir = System.getProperty("java.io.tmpdir");
      try {
        System.load(loadLibraryFromJarToTemp(tmpDir).getAbsolutePath());
      } catch (final IOException e) {
        libraryLoaded.set(LibraryState.NOT_LOADED);
        throw new RuntimeException("Unable to load the TFF Aggregation JNI shared library", e);
      }

      libraryLoaded.set(LibraryState.LOADED);
    } else {
      if (DEBUG_LOADING) {
        System.out.println("Another thread is loading the TFF Aggregation JNI shared library, waiting...");
      }

      waitForLibraryToBeLoaded();
    }
  }

  private static void waitForLibraryToBeLoaded() {
    final long wait = 10; // Time to wait before re-checking if another thread loaded the library
    final long timeout =
        10 * 1000; // Maximum time to wait for another thread to load the library (10 seconds)
    long waited = 0;
    try {
      while (libraryLoaded.get() == LibraryState.LOADING) {
        Thread.sleep(wait);
        waited += wait;

        if (waited >= timeout) {
          throw new RuntimeException(
              "Exceeded timeout whilst trying to load the TFF Aggregation JNI shared library");
        }
      }
    } catch (final InterruptedException e) {
      // restore interrupted status
      Thread.currentThread().interrupt();
      throw new RuntimeException("Interrupted whilst trying to load the TFF Aggregation JNI shared library", e);
    }
  }

  private File createTemp(final String tmpDir, final String libraryFileName) throws IOException {
    // create a temporary file to copy the library to
    final File temp;
    if (tmpDir == null || tmpDir.isEmpty()) {
      temp = File.createTempFile("libaggregation-jni", getOsExtension());
    } else {
      final File parentDir = new File(tmpDir);
      if (!parentDir.exists()) {
        throw new RuntimeException(
            "Directory: " + parentDir.getAbsolutePath() + " does not exist!");
      }
      temp = new File(parentDir, libraryFileName);
      if (temp.exists() && !temp.delete()) {
        throw new RuntimeException(
            "File: " + temp.getAbsolutePath() + " already exists and cannot be removed.");
      }
      if (!temp.createNewFile()) {
        throw new RuntimeException("File: " + temp.getAbsolutePath() + " could not be created.");
      }
    }
    if (temp.exists()) {
      temp.deleteOnExit();
      return temp;
    } else {
      throw new RuntimeException("File " + temp.getAbsolutePath() + " does not exist.");
    }
  }

  private File loadLibraryFromJarToTemp(final String tmpDir) throws IOException {
    try (InputStream is = getClass().getClassLoader().getResourceAsStream(sharedLibraryName)) {
      if (is != null) {
        final File temp = createTemp(tmpDir, sharedLibraryName);
        System.out.println("Extracting " + sharedLibraryName + " to " + temp.getAbsolutePath());
        Files.copy(is, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
        return temp;
      } else {
        if (DEBUG_LOADING) {
          System.out.println("Unable to find: " + sharedLibraryName + " on the classpath");
        }
      }
    }

    throw new RuntimeException(sharedLibraryName + " was not found inside JAR, and there is no fallback.");
  }

  /**
   * Private constructor to disallow instantiation
   */
  private NativeLibraryLoader() {}
}
