use clap::Parser;
use pdbtbx::StrictnessLevel;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

/// CLI tool to extract 3Di structural alphabet sequences from protein coordinates (PDB/mmCIF).
#[derive(Parser, Debug)]
#[command(
    name = "mini3di-rs",
    version,
    about = "Extract 3Di structural alphabet sequences from 3D protein coordinates",
    long_about = None
)]
pub struct Cli {
    /// Path to input PDB (.pdb) or mmCIF (.cif) structure file
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Optional chain identifier to filter (e.g., "A"). If omitted, processes all chains.
    #[arg(short, long, value_name = "CHAIN")]
    pub chain: Option<String>,

    /// Optional output file path. If omitted, outputs FASTA to stdout.
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    if let Err(err) = run(cli) {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    if !cli.input.exists() {
        return Err(format!(
            "Input file does not exist: {}",
            cli.input.display()
        ));
    }

    let input_str = cli
        .input
        .to_str()
        .ok_or_else(|| format!("Invalid UTF-8 path: {:?}", cli.input))?;

    // Open structure file using pdbtbx (loose strictness for real-world PDB/CIF compatibility)
    let (pdb, _warnings) = match pdbtbx::open(input_str, StrictnessLevel::Loose) {
        Ok(res) => res,
        Err(_) => pdbtbx::open_mmcif(input_str, StrictnessLevel::Loose).map_err(|errors| {
            format!(
                "Failed to open or parse structure file '{}': {:?}",
                cli.input.display(),
                errors
            )
        })?,
    };

    let struct_name = cli
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("structure");

    let model = match pdb.models().next() {
        Some(m) => m,
        None => {
            return Err(format!(
                "No 3D models found in structure file '{}'",
                cli.input.display()
            ))
        }
    };

    let mut fasta_entries = Vec::new();
    let mut found_target_chain = false;

    for chain in model.chains() {
        let chain_id = chain.id();

        if let Some(ref target_chain) = cli.chain {
            if chain_id != target_chain {
                continue;
            }
        }
        found_target_chain = true;

        let mut ca_list = Vec::new();
        let mut cb_list = Vec::new();
        let mut n_list = Vec::new();
        let mut c_list = Vec::new();

        for residue in chain.residues() {
            let mut ca: Option<[f32; 3]> = None;
            let mut cb: Option<[f32; 3]> = None;
            let mut n: Option<[f32; 3]> = None;
            let mut c: Option<[f32; 3]> = None;

            for atom in residue.atoms() {
                let name = atom.name().trim();
                let coord = [atom.x() as f32, atom.y() as f32, atom.z() as f32];
                match name {
                    "CA" => ca = Some(coord),
                    "CB" => cb = Some(coord),
                    "N" => n = Some(coord),
                    "C" => c = Some(coord),
                    _ => {}
                }
            }

            if let Some(ca_coord) = ca {
                ca_list.push(ca_coord);
                cb_list.push(cb.unwrap_or([f32::NAN, f32::NAN, f32::NAN]));
                n_list.push(n.unwrap_or([f32::NAN, f32::NAN, f32::NAN]));
                c_list.push(c.unwrap_or([f32::NAN, f32::NAN, f32::NAN]));
            }
        }

        if ca_list.is_empty() {
            continue;
        }

        let sequence = mini3di_rs::encode_atoms(&ca_list, &cb_list, &n_list, &c_list);
        fasta_entries.push(format!(">{}:{}\n{}", struct_name, chain_id, sequence));
    }

    if let Some(ref target_chain) = cli.chain {
        if !found_target_chain {
            return Err(format!(
                "Chain '{}' not found in structure file '{}'",
                target_chain,
                cli.input.display()
            ));
        }
    }

    if fasta_entries.is_empty() {
        return Err(format!(
            "No valid protein chain coordinates found in '{}'",
            cli.input.display()
        ));
    }

    let mut fasta_output = fasta_entries.join("\n");
    fasta_output.push('\n');

    if let Some(ref out_path) = cli.output {
        let mut file = File::create(out_path).map_err(|e| {
            format!(
                "Failed to create output file '{}': {}",
                out_path.display(),
                e
            )
        })?;
        file.write_all(fasta_output.as_bytes()).map_err(|e| {
            format!(
                "Failed to write output file '{}': {}",
                out_path.display(),
                e
            )
        })?;
    } else {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        handle
            .write_all(fasta_output.as_bytes())
            .map_err(|e| format!("Failed to write to stdout: {}", e))?;
    }

    Ok(())
}
