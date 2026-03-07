use std::path::PathBuf;

fn main() {
    // Look for libjxl build directory.
    // Set LIBJXL_BUILD_DIR to override, or LIBJXL_DIR for the repo root.
    let libjxl_dir = std::env::var("LIBJXL_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            // Common local path
            let home = std::env::var("HOME").ok()?;
            let p = PathBuf::from(home).join("work/jxl-efforts/libjxl");
            p.exists().then_some(p)
        });

    let build_dir = std::env::var("LIBJXL_BUILD_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| libjxl_dir.as_ref().map(|d| d.join("build")));

    let (Some(libjxl_dir), Some(build_dir)) = (libjxl_dir.as_ref(), build_dir.as_ref()) else {
        // No libjxl found — skip FFI build, benchmark will use subprocess fallback
        println!(
            "cargo:warning=libjxl not found, skipping C++ ssimulacra2 FFI. Set LIBJXL_DIR or LIBJXL_BUILD_DIR."
        );
        return;
    };

    if !build_dir.join("lib/libjxl-internal.a").exists() {
        println!("cargo:warning=libjxl build artifacts not found at {build_dir:?}, skipping FFI.");
        return;
    }

    // Compile the FFI shims
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

    // Link the pre-built ssimulacra2.cc.o
    let ssim2_obj = build_dir.join("tools/CMakeFiles/ssimulacra2.dir/ssimulacra2.cc.o");
    if ssim2_obj.exists() {
        println!("cargo:rustc-link-arg-benches={}", ssim2_obj.display());
    } else {
        println!("cargo:warning=ssimulacra2.cc.o not found at {ssim2_obj:?}");
        return;
    }

    // Static libs
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("tools").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("third_party/highway").display()
    );

    println!("cargo:rustc-link-lib=static=jxl-internal");
    println!("cargo:rustc-link-lib=static=jxl_tool");
    println!("cargo:rustc-link-lib=static=jxl_gauss_blur");
    println!("cargo:rustc-link-lib=static=hwy");

    // Dynamic libs
    println!("cargo:rustc-link-lib=dylib=jxl_cms");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Rerun triggers
    println!("cargo:rerun-if-changed=ffi/ssimulacra2_ffi.cpp");
    println!("cargo:rerun-if-changed=ffi/ssimulacra2_ffi.h");
    println!("cargo:rerun-if-changed=ffi/butteraugli_ffi.cpp");
    println!("cargo:rerun-if-changed=ffi/butteraugli_ffi.h");
    println!("cargo:rerun-if-env-changed=LIBJXL_DIR");
    println!("cargo:rerun-if-env-changed=LIBJXL_BUILD_DIR");

    // Tell the benchmark code the FFI is available
    println!("cargo:rustc-cfg=has_cpp_ssimulacra2");
    println!("cargo:rustc-cfg=has_cpp_butteraugli");
}
