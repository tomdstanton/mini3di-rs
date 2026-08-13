//! Integration tests for the `mini3di-rs` CLI binary.

use std::path::PathBuf;
use std::process::Command;

fn test_data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(filename)
}

#[test]
fn test_cli_1xso_stdout() {
    let output = Command::new(env!("CARGO_BIN_EXE_mini3di-rs"))
        .arg(test_data_path("1xso.pdb"))
        .output()
        .expect("Failed to execute mini3di-rs binary");

    assert!(
        output.status.success(),
        "CLI command should exit with status 0. Stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8(output.stdout).expect("Valid UTF-8 stdout");
    assert!(
        stdout_str.contains(">1xso:A"),
        "Output should contain header >1xso:A"
    );
    assert!(
        stdout_str.contains(">1xso:B"),
        "Output should contain header >1xso:B"
    );
    assert!(
        stdout_str.contains("DKKKWWKDFPDPKTKIKIWDDDDLFKIKIWMKIFQADFDKKWKWWACAQDCPV"),
        "Output sequence should contain expected 3Di prefix"
    );
}

#[test]
fn test_cli_8crb_chain_filter() {
    let output = Command::new(env!("CARGO_BIN_EXE_mini3di-rs"))
        .arg(test_data_path("8crb.pdb"))
        .arg("--chain")
        .arg("B")
        .output()
        .expect("Failed to execute mini3di-rs binary");

    assert!(
        output.status.success(),
        "CLI command should exit with status 0. Stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8(output.stdout).expect("Valid UTF-8 stdout");
    assert!(
        stdout_str.contains(">8crb:B"),
        "Output should contain header >8crb:B"
    );
    assert!(
        !stdout_str.contains(">8crb:A"),
        "Output should NOT contain chain A"
    );
    assert!(
        !stdout_str.contains(">8crb:C"),
        "Output should NOT contain chain C"
    );
}

#[test]
fn test_cli_output_file() {
    let temp_dir = std::env::temp_dir();
    let out_file = temp_dir.join("mini3di_test_output.fasta");

    let output = Command::new(env!("CARGO_BIN_EXE_mini3di-rs"))
        .arg(test_data_path("1xso.pdb"))
        .arg("-o")
        .arg(&out_file)
        .output()
        .expect("Failed to execute mini3di-rs binary");

    assert!(
        output.status.success(),
        "CLI command should exit with status 0. Stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out_file.exists(), "Output fasta file should exist");

    let content = std::fs::read_to_string(&out_file).expect("Read output fasta file");
    assert!(
        content.contains(">1xso:A"),
        "Output file should contain header >1xso:A"
    );

    let _ = std::fs::remove_file(out_file);
}

#[test]
fn test_cli_missing_chain_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_mini3di-rs"))
        .arg(test_data_path("1xso.pdb"))
        .arg("--chain")
        .arg("NONEXISTENT")
        .output()
        .expect("Failed to execute mini3di-rs binary");

    assert!(
        !output.status.success(),
        "CLI command should fail on missing chain"
    );
    assert_eq!(output.status.code(), Some(1), "Exit code should be 1");

    let stderr_str = String::from_utf8(output.stderr).expect("Valid UTF-8 stderr");
    assert!(
        stderr_str.contains("Chain 'NONEXISTENT' not found"),
        "Stderr should report missing chain error message"
    );
}

#[test]
fn test_cli_nonexistent_file_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_mini3di-rs"))
        .arg("non_existent_file_12345.pdb")
        .output()
        .expect("Failed to execute mini3di-rs binary");

    assert!(
        !output.status.success(),
        "CLI command should fail on non-existent file"
    );
    assert_eq!(output.status.code(), Some(1), "Exit code should be 1");

    let stderr_str = String::from_utf8(output.stderr).expect("Valid UTF-8 stderr");
    assert!(
        stderr_str.contains("Input file does not exist"),
        "Stderr should report missing file error message"
    );
}

#[test]
fn test_cli_disordered_atoms_last_conformer() {
    let temp_dir = std::env::temp_dir();
    let pdb_path = temp_dir.join(format!("mini3di_altloc_test_{}.pdb", std::process::id()));

    // PDB structure with 4 residues where residue 2 contains alternate location conformers (AALA and BALA)
    let pdb_content = "\
ATOM      1  N   ALA A   1      -0.785   0.722   0.507  1.00 10.00           N  
ATOM      2  CA  ALA A   1       0.563   0.211   0.278  1.00 10.00           C  
ATOM      3  C   ALA A   1       0.772  -1.121   0.984  1.00 10.00           C  
ATOM      4  CB  ALA A   1       1.621   1.218   0.730  1.00 10.00           C  
ATOM      5  N  AALA A   2       2.000  -1.500   1.000  0.50 10.00           N  
ATOM      6  CA AALA A   2       2.500  -2.500   1.500  0.50 10.00           C  
ATOM      7  C  AALA A   2       3.500  -2.000   2.500  0.50 10.00           C  
ATOM      8  CB AALA A   2       2.800  -3.500   0.500  0.50 10.00           C  
ATOM      9  N  BALA A   2       2.100  -1.600   1.100  0.50 10.00           N  
ATOM     10  CA BALA A   2       2.600  -2.600   1.600  0.50 10.00           C  
ATOM     11  C  BALA A   2       3.600  -2.100   2.600  0.50 10.00           C  
ATOM     12  CB BALA A   2       2.900  -3.600   0.600  0.50 10.00           C  
ATOM     13  N   ALA A   3       4.500  -1.000   3.000  1.00 10.00           N  
ATOM     14  CA  ALA A   3       5.500   0.000   3.500  1.00 10.00           C  
ATOM     15  C   ALA A   3       5.000   1.200   4.200  1.00 10.00           C  
ATOM     16  CB  ALA A   3       6.500  -0.500   4.500  1.00 10.00           C  
ATOM     17  N   ALA A   4       4.000   2.000   3.500  1.00 10.00           N  
ATOM     18  CA  ALA A   4       3.500   3.000   4.200  1.00 10.00           C  
ATOM     19  C   ALA A   4       2.200   2.500   4.800  1.00 10.00           C  
ATOM     20  CB  ALA A   4       4.500   3.500   5.200  1.00 10.00           C  
END
";
    std::fs::write(&pdb_path, pdb_content).expect("Failed to write altloc test PDB file");

    let output = Command::new(env!("CARGO_BIN_EXE_mini3di-rs"))
        .arg(&pdb_path)
        .output()
        .expect("Failed to execute mini3di-rs binary");

    assert!(
        output.status.success(),
        "CLI command should exit with status 0. Stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8(output.stdout).expect("Valid UTF-8 stdout");
    let expected_header = format!(">{}:A", pdb_path.file_stem().unwrap().to_str().unwrap());
    assert!(
        stdout_str.contains(&expected_header),
        "Output header should match structure name and chain ID"
    );
    let lines: Vec<&str> = stdout_str.lines().collect();
    assert!(
        lines.len() >= 2,
        "Output should contain header and sequence line"
    );
    assert_eq!(
        lines[1].len(),
        4,
        "Output 3Di sequence length should equal 4 residues"
    );

    let _ = std::fs::remove_file(pdb_path);
}
