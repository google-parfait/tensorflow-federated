# Copyright 2011 Google Inc. All Rights Reserved.
#
# Description:
#   liblzma is an LZMA2 compression library which has zlib-like APIs.

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

# liblzma is in public domain.
# Note that we removed all (L)GPL code from the xz package.
licenses(["unencumbered"])

exports_files(["LICENSE"])

xz_dir = "src/"

lzma_dir = xz_dir + "liblzma/"

filegroup(
    name = "liblzma_source",
    srcs = [
        xz_dir + "common/tuklib_cpucores.c",
        xz_dir + "common/tuklib_exit.c",
        xz_dir + "common/tuklib_mbstr_fw.c",
        xz_dir + "common/tuklib_mbstr_width.c",
        xz_dir + "common/tuklib_open_stdxxx.c",
        xz_dir + "common/tuklib_physmem.c",
        xz_dir + "common/tuklib_progname.c",
        lzma_dir + "check/check.c",
        lzma_dir + "check/crc32_fast.c",
        lzma_dir + "check/crc32_table.c",
        lzma_dir + "check/crc64_fast.c",
        lzma_dir + "check/crc64_table.c",
        lzma_dir + "common/alone_decoder.c",
        lzma_dir + "common/alone_encoder.c",
        lzma_dir + "common/auto_decoder.c",
        lzma_dir + "common/block_buffer_decoder.c",
        lzma_dir + "common/block_buffer_encoder.c",
        lzma_dir + "common/block_decoder.c",
        lzma_dir + "common/block_encoder.c",
        lzma_dir + "common/block_header_decoder.c",
        lzma_dir + "common/block_header_encoder.c",
        lzma_dir + "common/block_util.c",
        lzma_dir + "common/common.c",
        lzma_dir + "common/easy_buffer_encoder.c",
        lzma_dir + "common/easy_decoder_memusage.c",
        lzma_dir + "common/easy_encoder.c",
        lzma_dir + "common/easy_encoder_memusage.c",
        lzma_dir + "common/easy_preset.c",
        lzma_dir + "common/filter_buffer_decoder.c",
        lzma_dir + "common/filter_buffer_encoder.c",
        lzma_dir + "common/filter_common.c",
        lzma_dir + "common/filter_decoder.c",
        lzma_dir + "common/filter_encoder.c",
        lzma_dir + "common/filter_flags_decoder.c",
        lzma_dir + "common/filter_flags_encoder.c",
        lzma_dir + "common/hardware_cputhreads.c",
        lzma_dir + "common/hardware_physmem.c",
        lzma_dir + "common/index.c",
        lzma_dir + "common/index_decoder.c",
        lzma_dir + "common/index_encoder.c",
        lzma_dir + "common/index_hash.c",
        lzma_dir + "common/outqueue.c",
        lzma_dir + "common/stream_buffer_decoder.c",
        lzma_dir + "common/stream_buffer_encoder.c",
        lzma_dir + "common/stream_decoder.c",
        lzma_dir + "common/stream_encoder.c",
        lzma_dir + "common/stream_encoder_mt.c",
        lzma_dir + "common/stream_flags_common.c",
        lzma_dir + "common/stream_flags_decoder.c",
        lzma_dir + "common/stream_flags_encoder.c",
        lzma_dir + "common/vli_decoder.c",
        lzma_dir + "common/vli_encoder.c",
        lzma_dir + "common/vli_size.c",
        lzma_dir + "delta/delta_common.c",
        lzma_dir + "delta/delta_decoder.c",
        lzma_dir + "delta/delta_encoder.c",
        lzma_dir + "lz/lz_decoder.c",
        lzma_dir + "lz/lz_encoder.c",
        lzma_dir + "lz/lz_encoder_mf.c",
        lzma_dir + "lzma/fastpos_table.c",
        lzma_dir + "lzma/lzma2_decoder.c",
        lzma_dir + "lzma/lzma2_encoder.c",
        lzma_dir + "lzma/lzma_decoder.c",
        lzma_dir + "lzma/lzma_encoder.c",
        lzma_dir + "lzma/lzma_encoder_optimum_fast.c",
        lzma_dir + "lzma/lzma_encoder_optimum_normal.c",
        lzma_dir + "lzma/lzma_encoder_presets.c",
        lzma_dir + "rangecoder/price_table.c",
        lzma_dir + "simple/arm.c",
        lzma_dir + "simple/armthumb.c",
        lzma_dir + "simple/ia64.c",
        lzma_dir + "simple/powerpc.c",
        lzma_dir + "simple/simple_coder.c",
        lzma_dir + "simple/simple_decoder.c",
        lzma_dir + "simple/simple_encoder.c",
        lzma_dir + "simple/sparc.c",
        lzma_dir + "simple/x86.c",
    ] + select({
        "//conditions:default": [lzma_dir + "check/sha256.c"],
    }),
)

cc_library(
    name = "liblzma",
    hdrs = [lzma_dir + "api/lzma.h"],
    include_prefix = "third_party/liblzma",
    strip_include_prefix = lzma_dir + "api",
    deps = ["liblzma_impl"],
)

cc_library(
    name = "liblzma_impl",
    srcs = glob([
        xz_dir + "common/*.h",
        lzma_dir + "api/lzma/*.h",
        lzma_dir + "check/*.h",
        lzma_dir + "common/*.h",
        lzma_dir + "delta/*.h",
        lzma_dir + "lz/*.h",
        lzma_dir + "lzma/*.h",
        lzma_dir + "rangecoder/*.h",
        lzma_dir + "simple/*.h",
    ]) + [
        lzma_dir + "api/lzma.h",
        "config.h",
        ":liblzma_source",
    ],
    copts = [
        "-std=c99",
        "-DHAVE_CONFIG_H",
    ],
    includes = [
        xz_dir + "common/",
        lzma_dir + "api/",
        lzma_dir + "check/",
        lzma_dir + "common/",
        lzma_dir + "delta/",
        lzma_dir + "lz/",
        lzma_dir + "lzma/",
        lzma_dir + "rangecoder/",
        lzma_dir + "simple/",
        ".",
    ],
    visibility = ["//visibility:private"],
)

prog_dir = xz_dir + "xz/"

cc_binary(
    name = "xz",
    srcs = [
        prog_dir + "args.c",
        prog_dir + "coder.c",
        prog_dir + "file_io.c",
        prog_dir + "hardware.c",
        prog_dir + "list.c",
        prog_dir + "main.c",
        prog_dir + "message.c",
        prog_dir + "mytime.c",
        prog_dir + "options.c",
        prog_dir + "signals.c",
        prog_dir + "suffix.c",
        prog_dir + "util.c",
        prog_dir + "args.h",
        prog_dir + "coder.h",
        prog_dir + "file_io.h",
        prog_dir + "hardware.h",
        prog_dir + "list.h",
        prog_dir + "main.h",
        prog_dir + "message.h",
        prog_dir + "mytime.h",
        prog_dir + "options.h",
        prog_dir + "private.h",
        prog_dir + "signals.h",
        prog_dir + "suffix.h",
        prog_dir + "util.h",
        xz_dir + "common/tuklib_cpucores.c",
        xz_dir + "common/tuklib_exit.c",
        xz_dir + "common/tuklib_mbstr_fw.c",
        xz_dir + "common/tuklib_mbstr_width.c",
        xz_dir + "common/tuklib_open_stdxxx.c",
        xz_dir + "common/tuklib_physmem.c",
        xz_dir + "common/tuklib_progname.c",
    ],
    copts = [
        "-std=c99",
        "-DHAVE_CONFIG_H",
    ],
    includes = [
        xz_dir + "common/",
        lzma_dir + "api/",
    ],
    linkopts = ["-lpthread"],
    deps = [
        ":liblzma",
    ],
)

genrule(
    name = "config_h",
    outs = ["config.h"],
    cmd = """cat - > $@ <<'EOF'
/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* How many MiB of RAM to assume if the real amount cannot be determined. */
#define ASSUME_RAM 128

/* Define to 1 if translation of program messages to the user's native
   language is requested. */
/* #undef ENABLE_NLS */

/* Define to 1 if bswap_16 is available. */
#define HAVE_BSWAP_16 1

/* Define to 1 if bswap_32 is available. */
#define HAVE_BSWAP_32 1

/* Define to 1 if bswap_64 is available. */
#define HAVE_BSWAP_64 1

/* Define to 1 if you have the <byteswap.h> header file. */
#if defined(__APPLE__)
// byteswap.h does not exist in Apple SDKs.
// This #define must be entirely eliminated (not 0), as usages are checking
// for it being defined.
#undef HAVE_BYTESWAP_H
#else
#define HAVE_BYTESWAP_H 1
#endif

/* Define to 1 if the system has the type `CC_SHA256_CTX'. */
#if defined(__APPLE__)
#define HAVE_CC_SHA256_CTX 1
#endif

/* Define to 1 if you have the `CC_SHA256_Init' function. */
#if defined(__APPLE__)
#define HAVE_CC_SHA256_INIT 1
#endif

/* Define to 1 if you have the MacOS X function CFLocaleCopyCurrent in the
   CoreFoundation framework. */
/* #undef HAVE_CFLOCALECOPYCURRENT */

/* Define to 1 if you have the MacOS X function CFPreferencesCopyAppValue in
   the CoreFoundation framework. */
/* #undef HAVE_CFPREFERENCESCOPYAPPVALUE */

/* Define to 1 if crc32 integrity check is enabled. */
#define HAVE_CHECK_CRC32 1

/* Define to 1 if crc64 integrity check is enabled. */
#define HAVE_CHECK_CRC64 1

/* Define to 1 if sha256 integrity check is enabled. */
#define HAVE_CHECK_SHA256 1

/* Define to 1 if you have the `clock_gettime' function. */
#if !defined(__APPLE__)
#define HAVE_CLOCK_GETTIME 1
#endif

/* Define to 1 if you have the <CommonCrypto/CommonDigest.h> header file. */
#if defined(__APPLE__)
#define HAVE_COMMONCRYPTO_COMMONDIGEST_H 1
#endif

/* Define if the GNU dcgettext() function is already present or preinstalled.
   */
/* #undef HAVE_DCGETTEXT */

/* Define to 1 if you have the declaration of `CLOCK_MONOTONIC', and to 0 if
   you don't. */
#if !defined(__APPLE__)
#define HAVE_DECL_CLOCK_MONOTONIC 1
#endif

/* Define to 1 if you have the declaration of `program_invocation_name', and
   to 0 if you don't. */
#if defined(__ANDROID__) || defined(__APPLE__)
#define HAVE_DECL_PROGRAM_INVOCATION_NAME 0
#else
#define HAVE_DECL_PROGRAM_INVOCATION_NAME 1
#endif

/* Define to 1 if any of HAVE_DECODER_foo have been defined. */
#define HAVE_DECODERS 1

/* Define to 1 if arm decoder is enabled. */
#define HAVE_DECODER_ARM 1

/* Define to 1 if armthumb decoder is enabled. */
#define HAVE_DECODER_ARMTHUMB 1

/* Define to 1 if delta decoder is enabled. */
#define HAVE_DECODER_DELTA 1

/* Define to 1 if ia64 decoder is enabled. */
#define HAVE_DECODER_IA64 1

/* Define to 1 if lzma1 decoder is enabled. */
#define HAVE_DECODER_LZMA1 1

/* Define to 1 if lzma2 decoder is enabled. */
#define HAVE_DECODER_LZMA2 1

/* Define to 1 if powerpc decoder is enabled. */
#define HAVE_DECODER_POWERPC 1

/* Define to 1 if sparc decoder is enabled. */
#define HAVE_DECODER_SPARC 1

/* Define to 1 if x86 decoder is enabled. */
#define HAVE_DECODER_X86 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if any of HAVE_ENCODER_foo have been defined. */
#define HAVE_ENCODERS 1

/* Define to 1 if arm encoder is enabled. */
#define HAVE_ENCODER_ARM 1

/* Define to 1 if armthumb encoder is enabled. */
#define HAVE_ENCODER_ARMTHUMB 1

/* Define to 1 if delta encoder is enabled. */
#define HAVE_ENCODER_DELTA 1

/* Define to 1 if ia64 encoder is enabled. */
#define HAVE_ENCODER_IA64 1

/* Define to 1 if lzma1 encoder is enabled. */
#define HAVE_ENCODER_LZMA1 1

/* Define to 1 if lzma2 encoder is enabled. */
#define HAVE_ENCODER_LZMA2 1

/* Define to 1 if powerpc encoder is enabled. */
#define HAVE_ENCODER_POWERPC 1

/* Define to 1 if sparc encoder is enabled. */
#define HAVE_ENCODER_SPARC 1

/* Define to 1 if x86 encoder is enabled. */
#define HAVE_ENCODER_X86 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the `futimens' function. */
#define HAVE_FUTIMENS 1

/* Define to 1 if you have the `futimes' function. */
/* #undef HAVE_FUTIMES */

/* Define to 1 if you have the `futimesat' function. */
/* #undef HAVE_FUTIMESAT */

/* Define to 1 if you have the <getopt.h> header file. */
#define HAVE_GETOPT_H 1

/* Define to 1 if you have the `getopt_long' function. */
#define HAVE_GETOPT_LONG 1

/* Define if the GNU gettext() function is already present or preinstalled. */
/* #undef HAVE_GETTEXT */

/* Define if you have the iconv() function and it works. */
/* #undef HAVE_ICONV */

/* Define to 1 if you have the <immintrin.h> header file. */
/* #undef HAVE_IMMINTRIN_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if mbrtowc and mbstate_t are properly declared. */
#define HAVE_MBRTOWC 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 to enable bt2 match finder. */
#define HAVE_MF_BT2 1

/* Define to 1 to enable bt3 match finder. */
#define HAVE_MF_BT3 1

/* Define to 1 to enable bt4 match finder. */
#define HAVE_MF_BT4 1

/* Define to 1 to enable hc3 match finder. */
#define HAVE_MF_HC3 1

/* Define to 1 to enable hc4 match finder. */
#define HAVE_MF_HC4 1

/* Define to 1 if you have the <minix/sha2.h> header file. */
/* #undef HAVE_MINIX_SHA2_H */

/* Define to 1 if getopt.h declares extern int optreset. */
/* #undef HAVE_OPTRESET */

/* Define to 1 if you have the `posix_fadvise' function. */
#if !defined(__APPLE__)
#define HAVE_POSIX_FADVISE 1
#endif

/* Define to 1 if you have the `pthread_condattr_setclock' function. */
#if !defined(__APPLE__) && !defined(__ANDROID__)
#define HAVE_PTHREAD_CONDATTR_SETCLOCK 1
#endif

/* Have PTHREAD_PRIO_INHERIT. */
#define HAVE_PTHREAD_PRIO_INHERIT 1

/* Define to 1 if you have the `SHA256Init' function. */
/* #undef HAVE_SHA256INIT */

/* Define to 1 if the system has the type `SHA256_CTX'. */
/* #undef HAVE_SHA256_CTX */

/* Define to 1 if you have the <sha256.h> header file. */
/* #undef HAVE_SHA256_H */

/* Define to 1 if you have the `SHA256_Init' function. */
/* #undef HAVE_SHA256_INIT */

/* Define to 1 if the system has the type `SHA2_CTX'. */
/* #undef HAVE_SHA2_CTX */

/* Define to 1 if you have the <sha2.h> header file. */
/* #undef HAVE_SHA2_H */

/* Define to 1 if optimizing for size. */
/* #undef HAVE_SMALL */

/* Define to 1 if stdbool.h conforms to C99. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if `st_atimensec' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_ATIMENSEC */

/* Define to 1 if `st_atimespec.tv_nsec' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_ATIMESPEC_TV_NSEC */

/* Define to 1 if `st_atim.st__tim.tv_nsec' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_ATIM_ST__TIM_TV_NSEC */

/* Define to 1 if `st_atim.tv_nsec' is a member of `struct stat'. */
#define HAVE_STRUCT_STAT_ST_ATIM_TV_NSEC 1

/* Define to 1 if `st_uatime' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_UATIME */

/* Define to 1 if you have the <sys/byteorder.h> header file. */
/* #undef HAVE_SYS_BYTEORDER_H */

/* Define to 1 if you have the <sys/endian.h> header file. */
/* #undef HAVE_SYS_ENDIAN_H */

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if the system has the type `uintptr_t'. */
#define HAVE_UINTPTR_T 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `utime' function. */
/* #undef HAVE_UTIME */

/* Define to 1 if you have the `utimes' function. */
/* #undef HAVE_UTIMES */

/* Define to 1 or 0, depending whether the compiler supports simple visibility
   declarations. */
#if defined(__ANDROID__) || defined(__APPLE__)
/*
 * When compiling on embedded platforms, do not override
 * any default -fvisibility setting; linking to liblzma does not mean we want
 * to export all of its functions to the library users.
 */
#define HAVE_VISIBILITY 0
#else
#define HAVE_VISIBILITY 1
#endif

/* Define to 1 if you have the `wcwidth' function. */
#define HAVE_WCWIDTH 1

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* Define to 1 if _mm_movemask_epi8 is available. */
/* #undef HAVE__MM_MOVEMASK_EPI8 */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Define to 1 when using POSIX threads (pthreads). */
#define MYTHREAD_POSIX 1

/* Define to 1 when using Windows Vista compatible threads. This uses features
   that are not available on Windows XP. */
/* #undef MYTHREAD_VISTA */

/* Define to 1 when using Windows 95 (and thus XP) compatible threads. This
   avoids use of features that were added in Windows Vista. */
/* #undef MYTHREAD_WIN95 */

/* Define to 1 to disable debugging code. */
#define NDEBUG 1

/* Name of package */
#define PACKAGE "xz"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "lasse.collin@tukaani.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "XZ Utils"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "XZ Utils 5.2.2"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "xz"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://tukaani.org/xz/"

/* Define to the version of this package. */
#define PACKAGE_VERSION "5.2.2"

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if the number of available CPU cores can be detected with
   cpuset(2). */
/* #undef TUKLIB_CPUCORES_CPUSET */

/* Define to 1 if the number of available CPU cores can be detected with
   pstat_getdynamic(). */
/* #undef TUKLIB_CPUCORES_PSTAT_GETDYNAMIC */

/* Define to 1 if the number of available CPU cores can be detected with
   sysconf(_SC_NPROCESSORS_ONLN) or sysconf(_SC_NPROC_ONLN). */
#define TUKLIB_CPUCORES_SYSCONF 1

/* Define to 1 if the number of available CPU cores can be detected with
   sysctl(). */
/* #undef TUKLIB_CPUCORES_SYSCTL */

/* Define to 1 if the system supports fast unaligned access to 16-bit and
   32-bit integers. */
#define TUKLIB_FAST_UNALIGNED_ACCESS 1

/* Define to 1 if the amount of physical memory can be detected with
   _system_configuration.physmem. */
/* #undef TUKLIB_PHYSMEM_AIX */

/* Define to 1 if the amount of physical memory can be detected with
   getinvent_r(). */
/* #undef TUKLIB_PHYSMEM_GETINVENT_R */

/* Define to 1 if the amount of physical memory can be detected with
   getsysinfo(). */
/* #undef TUKLIB_PHYSMEM_GETSYSINFO */

/* Define to 1 if the amount of physical memory can be detected with
   pstat_getstatic(). */
/* #undef TUKLIB_PHYSMEM_PSTAT_GETSTATIC */

/* Define to 1 if the amount of physical memory can be detected with
   sysconf(_SC_PAGESIZE) and sysconf(_SC_PHYS_PAGES). */
#if defined(__APPLE__)
// Apple SDKs (either iOS or OSX) do not define _SC_PHYS_PAGES.
// This #define must be entirely eliminated (not 0), as usages are checking
// for it being defined.
#undef TUKLIB_PHYSMEM_SYSCONF
#else
#define TUKLIB_PHYSMEM_SYSCONF 1
#endif

/* Define to 1 if the amount of physical memory can be detected with sysctl().
   */
/* #undef TUKLIB_PHYSMEM_SYSCTL */

/* Define to 1 if the amount of physical memory can be detected with Linux
   sysinfo(). */
/* #undef TUKLIB_PHYSMEM_SYSINFO */

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable threading extensions on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif


/* Version number of package */
#define VERSION "5.2.2"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Enable large inode numbers on Mac OS X 10.5.  */
#ifndef _DARWIN_USE_64_BIT_INODE
# define _DARWIN_USE_64_BIT_INODE 1
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define to 1 if on MINIX. */
/* #undef _MINIX */

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
/* #undef _POSIX_1_SOURCE */

/* Define to 1 if you need to in order for `stat' and other things to work. */
/* #undef _POSIX_SOURCE */

/* Define for Solaris 2.5.1 so the uint32_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT32_T */

/* Define for Solaris 2.5.1 so the uint64_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT64_T */

/* Define for Solaris 2.5.1 so the uint8_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT8_T */

/* Define to rpl_ if the getopt replacement functions and variables should be
   used. */
/* #undef __GETOPT_PREFIX */

/* Define to the type of a signed integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int32_t */

/* Define to the type of a signed integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int64_t */

/* Define to the type of an unsigned integer type of width exactly 16 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint16_t */

/* Define to the type of an unsigned integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint32_t */

/* Define to the type of an unsigned integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint64_t */

/* Define to the type of an unsigned integer type of width exactly 8 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint8_t */

/* Define to the type of an unsigned integer type wide enough to hold a
   pointer, if such a type exists, and if the system does not define it. */
/* #undef uintptr_t */
EOF""",
)
