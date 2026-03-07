use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=ffi/");
    println!("cargo:rerun-if-env-changed=LIBJXL_DIR");
    println!("cargo:rerun-if-env-changed=LIBJXL_BUILD_DIR");

    if let Err(msg) = try_build_cpp_ffi() {
        println!("cargo:warning=Skipping C++ FFI benchmarks: {msg}");
    }
}

fn try_build_cpp_ffi() -> Result<(), String> {
    let (libjxl_dir, build_dir) = locate_libjxl()?;

    if !build_dir.join("lib/libjxl-internal.a").exists() {
        return Err(format!(
            "libjxl-internal.a not found in {}",
            build_dir.display()
        ));
    }

    compile_ffi_shims(&libjxl_dir, &build_dir);
    link_libjxl(&build_dir)?;

    println!("cargo:rustc-cfg=has_cpp_ssimulacra2");
    println!("cargo:rustc-cfg=has_cpp_butteraugli");
    Ok(())
}

/// Find libjxl source + build directories.
///
/// Priority:
/// 1. `LIBJXL_DIR` / `LIBJXL_BUILD_DIR` env vars
/// 2. Common local path (`~/work/jxl-efforts/libjxl`)
/// 3. `vendor/libjxl` (auto-cloned + cmake-built if needed)
fn locate_libjxl() -> Result<(PathBuf, PathBuf), String> {
    // 1. Explicit env vars
    if let Ok(dir) = std::env::var("LIBJXL_DIR") {
        let dir = PathBuf::from(dir);
        if dir.exists() {
            let build = std::env::var("LIBJXL_BUILD_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| dir.join("build"));
            if build.join("lib/libjxl-internal.a").exists() {
                return Ok((dir, build));
            }
            // Source exists, build missing — try cmake
            cmake_build(&dir, &build)?;
            return Ok((dir, build));
        }
    }

    // 2. Common local path
    if let Ok(home) = std::env::var("HOME") {
        let dir = PathBuf::from(&home).join("work/jxl-efforts/libjxl");
        if dir.exists() {
            let build = dir.join("build");
            if build.join("lib/libjxl-internal.a").exists() {
                return Ok((dir, build));
            }
        }
    }

    // 3. vendor/libjxl — clone if absent, build if needed
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor_dir = manifest_dir.join("vendor/libjxl");
    let build_dir = vendor_dir.join("build");

    if !vendor_dir.join("CMakeLists.txt").exists() {
        clone_libjxl(&vendor_dir)?;
    }

    if !build_dir.join("lib/libjxl-internal.a").exists() {
        cmake_build(&vendor_dir, &build_dir)?;
    }

    Ok((vendor_dir, build_dir))
}

fn clone_libjxl(target: &Path) -> Result<(), String> {
    eprintln!("cloning libjxl into {}", target.display());

    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "https://github.com/libjxl/libjxl.git",
        ])
        .arg(target)
        .status()
        .map_err(|e| format!("git clone failed: {e}"))?;

    if !status.success() {
        return Err("git clone libjxl failed".into());
    }

    // Only init the submodules we need (highway for SIMD, skcms for CMS)
    let status = Command::new("git")
        .args([
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "third_party/highway",
            "third_party/skcms",
        ])
        .current_dir(target)
        .status()
        .map_err(|e| format!("git submodule update failed: {e}"))?;

    if !status.success() {
        return Err("git submodule update failed".into());
    }

    Ok(())
}

fn cmake_build(source_dir: &Path, build_dir: &Path) -> Result<(), String> {
    eprintln!(
        "building libjxl: {} -> {}",
        source_dir.display(),
        build_dir.display()
    );

    std::fs::create_dir_all(build_dir).map_err(|e| format!("mkdir failed: {e}"))?;

    let status = Command::new("cmake")
        .arg(format!("-S{}", source_dir.display()))
        .arg(format!("-B{}", build_dir.display()))
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DJPEGXL_STATIC=ON",
            "-DJPEGXL_ENABLE_TOOLS=ON",
            "-DJPEGXL_ENABLE_DOXYGEN=OFF",
            "-DJPEGXL_ENABLE_MANPAGES=OFF",
            "-DJPEGXL_ENABLE_BENCHMARK=OFF",
            "-DJPEGXL_ENABLE_EXAMPLES=OFF",
            "-DJPEGXL_ENABLE_JNI=OFF",
            "-DJPEGXL_ENABLE_SJPEG=OFF",
            "-DJPEGXL_ENABLE_OPENEXR=OFF",
            "-DJPEGXL_ENABLE_JPEGLI=OFF",
            "-DJPEGXL_ENABLE_TCMALLOC=OFF",
            "-DJPEGXL_BUNDLE_LIBPNG=OFF",
            "-DBUILD_TESTING=OFF",
        ])
        .status()
        .map_err(|e| format!("cmake configure failed: {e}"))?;

    if !status.success() {
        return Err("cmake configure failed".into());
    }

    let nproc = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".into());

    let status = Command::new("cmake")
        .arg("--build")
        .arg(build_dir)
        .args(["--parallel", &nproc])
        .status()
        .map_err(|e| format!("cmake build failed: {e}"))?;

    if !status.success() {
        return Err("cmake build failed".into());
    }

    Ok(())
}

fn compile_ffi_shims(libjxl_dir: &Path, build_dir: &Path) {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("ffi/ssimulacra2_ffi.cpp")
        .file("ffi/butteraugli_ffi.cpp")
        .include(libjxl_dir)
        .include(libjxl_dir.join("lib/include"))
        .include(build_dir.join("lib/include"))
        .include("ffi")
        .opt_level(3)
        .compile("libjxl_ffi");
}

fn link_libjxl(build_dir: &Path) -> Result<(), String> {
    // ssimulacra2.cc.o — the ssimulacra2 algorithm implementation
    let ssim2_obj = build_dir.join("tools/CMakeFiles/ssimulacra2.dir/ssimulacra2.cc.o");
    if !ssim2_obj.exists() {
        return Err(format!(
            "ssimulacra2.cc.o not found at {}",
            ssim2_obj.display()
        ));
    }
    println!("cargo:rustc-link-arg-benches={}", ssim2_obj.display());

    // Search paths
    for subdir in ["lib", "tools", "third_party/highway", "third_party/brotli"] {
        let p = build_dir.join(subdir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }

    // Static libs
    println!("cargo:rustc-link-lib=static=jxl-internal");
    println!("cargo:rustc-link-lib=static=jxl_tool");
    println!("cargo:rustc-link-lib=static=jxl_gauss_blur");
    println!("cargo:rustc-link-lib=static=hwy");

    // CMS — static when available (vendor build), dynamic otherwise (system build)
    if build_dir.join("lib/libjxl_cms.a").exists() {
        println!("cargo:rustc-link-lib=static=jxl_cms");
    } else {
        println!("cargo:rustc-link-lib=dylib=jxl_cms");
    }

    // Brotli — needed by jxl-internal for ICC profile handling
    for lib in ["brotlienc", "brotlidec", "brotlicommon"] {
        let static_name = format!("lib{lib}-static.a");
        if build_dir
            .join("third_party/brotli")
            .join(&static_name)
            .exists()
        {
            println!("cargo:rustc-link-lib=static={lib}-static");
        } else if build_dir.join("lib").join(format!("lib{lib}.a")).exists() {
            println!("cargo:rustc-link-lib=static={lib}");
        }
        // If neither exists, hope it's bundled into jxl-internal or system-provided
    }

    // C++ runtime
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    Ok(())
}
