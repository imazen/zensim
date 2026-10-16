//! Optional numerical-owner build. Ordinary builds require no VCS executable.
use std::{env, fs, path::PathBuf, process::Command};
const OWNER: &str = "81a545a3ff3457c9fabae93bc9d262b19b50952e";
fn main() {
    println!("cargo:rustc-check-cfg=cfg(e34_owner_oracle)");
    println!("cargo:rerun-if-env-changed=E34_OWNER_ORACLE");
    if env::var("E34_OWNER_ORACLE").as_deref() != Ok("1") {
        return;
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let snapshot = output.join("e34-owner");
    fs::create_dir_all(&snapshot).unwrap();
    let mut pins = Vec::new();
    for name in ["mod.rs", "reports.rs"] {
        let path = format!("zensim-validate/src/bin/panel_e34/{name}");
        let result = Command::new("jj")
            .args(["--ignore-working-copy", "file", "show", "-r", OWNER, &path])
            .current_dir(
                PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
                    .parent()
                    .unwrap(),
            )
            .output()
            .expect("jj required for explicit owner parity build");
        assert!(
            result.status.success(),
            "pinned E34 owner unavailable: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let target = snapshot.join(name);
        fs::write(&target, &result.stdout).unwrap();
        let hash = Command::new("sha256sum")
            .arg(&target)
            .output()
            .expect("sha256sum required for source receipt");
        assert!(hash.status.success(), "owner source hashing failed");
        let hash = String::from_utf8(hash.stdout).unwrap();
        pins.push(format!(
            "\"{name}\":\"{}\"",
            hash.split_whitespace().next().unwrap()
        ));
    }
    let receipt = format!(
        "{{\"owner_commit\":\"{OWNER}\",\"files\":{{{}}}}}",
        pins.join(",")
    );
    let receipt_path = output.join("e34-owner-source.json");
    fs::write(&receipt_path, receipt).unwrap();
    // Compile the original module as a module file, preserving its inner docs
    // and relative reports.rs path. Only this small entry wrapper is generated.
    let entry = format!(
        "#[path = {:?}] mod owner;\nfn main()->std::process::ExitCode {{\nif std::env::args().nth(1).as_deref()==Some(\"source-pins\") {{ println!(\"{{}}\",include_str!({:?})); std::process::ExitCode::SUCCESS }} else {{ owner::main() }}\n}}\n",
        snapshot.join("mod.rs"),
        receipt_path
    );
    fs::write(output.join("e34-owner-entry.rs"), entry).unwrap();
    println!("cargo:rustc-cfg=e34_owner_oracle");
}
