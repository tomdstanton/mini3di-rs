//! Integration test suite for `mini3di-rs` 3Di encoder.
//!
//! Covers Tiers 1 through 4:
//! - Tier 1: Component Unit Tests (math, partner lookup, descriptors, VAE, alphabet)
//! - Tier 2: Boundary & Corner Cases (short chains, terminal masking, NaN coordinates, distance ties)
//! - Tier 3: Cross-Feature Combinations (masked & glycine mix)
//! - Tier 4: Real-World PDB Structure Validation (1xso, 3bww, 3bww.masked, 8crb)

use mini3di_rs::{
    build_sequence, calc_conformation_descriptors, compute_virtual_centers, encode_atoms,
    encode_descriptors, find_partner_indices,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// ============================================================================
// Tier 1 — Component Unit Tests
// ============================================================================

/// Test virtual center computation against known reference numerical values.
#[test]
fn test_virtual_center_calc() {
    // Reference sample 1:
    // CA=[34.826, 19.254, 17.339], CB=[35.285, 18.694, 15.994], N=[35.805, 19.041, 18.426]
    // -> VC ~ [32.2276, 20.2157, 16.0518] (tolerance 1e-3).
    // Reference sample 2:
    // CA=[21.056, 18.27, 0.063], CB=[21.428, 19.604, 0.838], N=[21.789, 17.734, -1.084]
    // -> VC ~ [18.5941, 17.8221, 2.01565] (tolerance 1e-3).

    let ca = vec![[34.826, 19.254, 17.339], [21.056, 18.27, 0.063]];
    let cb = vec![[35.285, 18.694, 15.994], [21.428, 19.604, 0.838]];
    let n = vec![[35.805, 19.041, 18.426], [21.789, 17.734, -1.084]];
    // Arbitrary C coordinates since virtual center only depends on CA, CB, N
    let c = vec![[34.826, 19.254, 17.339], [21.056, 18.27, 0.063]];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert_eq!(vc.len(), 2);
    assert!(valid[0] && valid[1], "Both virtual centers must be valid");

    let expected_vc0 = [32.2276, 20.2157, 16.0518];
    let expected_vc1 = [18.5941, 17.8221, 2.01565];

    for dim in 0..3 {
        assert!(
            (vc[0][dim] - expected_vc0[dim]).abs() < 1e-3,
            "VC[0][{}] = {}, expected {}",
            dim,
            vc[0][dim],
            expected_vc0[dim]
        );
        assert!(
            (vc[1][dim] - expected_vc1[dim]).abs() < 1e-3,
            "VC[1][{}] = {}, expected {}",
            dim,
            vc[1][dim],
            expected_vc1[dim]
        );
    }
}

/// Test Cb reconstruction for Glycine (where Cb equals Ca or is missing/NaN).
#[test]
fn test_cb_approximation() {
    let ca = vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    // Case 0: missing Cb (NaN), Case 1: Glycine Cb == Ca, Case 2: normal Cb
    let cb = vec![
        [f32::NAN, f32::NAN, f32::NAN],
        [1.0, 2.0, 3.0],
        [4.5, 5.5, 6.5],
    ];
    let n = vec![[-1.0, 0.0, 0.0], [0.0, 2.0, 3.0], [3.0, 5.0, 6.0]];
    let c = vec![[0.0, 1.0, 0.0], [1.0, 3.0, 3.0], [4.0, 6.0, 6.0]];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert_eq!(vc.len(), 3);
    assert!(
        valid[0],
        "VC[0] should be valid via Cb reconstruction from NaN"
    );
    assert!(
        valid[1],
        "VC[1] should be valid via Cb reconstruction from Glycine (Cb==Ca)"
    );
    assert!(valid[2], "VC[2] should be valid");
    assert!(!vc[0][0].is_nan(), "VC[0] x must not be NaN");
    assert!(!vc[1][0].is_nan(), "VC[1] x must not be NaN");
}

/// Test partner selection logic, verifying terminal positions 0 and N-1 are masked
/// and finding min distance partner indices.
#[test]
fn test_partner_index_lookup() {
    let vc = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0], // closest to residue 1
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ];
    let valid_mask = vec![true; 5];

    let partners = find_partner_indices(&vc, &valid_mask);
    assert_eq!(partners.len(), 5);

    // Residue 1 (interior) nearest neighbor should be residue 2 (dist = 0.1)
    assert_eq!(partners[1], 2, "Partner of residue 1 should be residue 2");
    // Residue 2 (interior) nearest neighbor should be residue 1 (dist = 0.1)
    assert_eq!(partners[2], 1, "Partner of residue 2 should be residue 1");
}

/// Test 10D geometric descriptor generation, normalization/clipping, and log-distance features.
#[test]
fn test_feature_encoder_descriptors() {
    let ca = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let partner_indices = vec![0, 2, 1, 4, 3];
    let valid_mask = vec![true; 5];

    let (desc, mask) = calc_conformation_descriptors(&ca, &partner_indices, &valid_mask);
    assert_eq!(desc.len(), 5);
    assert_eq!(mask.len(), 5);

    // Terminal residues (0 and N-1 = 4) must be masked
    assert!(mask[0], "Residue 0 descriptor must be masked");
    assert!(mask[4], "Residue 4 descriptor must be masked");

    // For interior residue 1 (partner = 2):
    // Index diff J - I = 2 - 1 = 1.0
    // desc[1][8] is clipped diff in [-4, 4]
    // desc[1][9] is copysign(log(|J - I| + 1), J - I) = ln(2) ~ 0.693147
    assert_eq!(desc[1][8], 1.0, "Descriptor 8 should be clipped index diff");
    let expected_log = (1.0f32 + 1.0).ln();
    assert!(
        (desc[1][9] - expected_log).abs() < 1e-5,
        "Descriptor 9 should be log-distance feature"
    );
}

/// Test 3-layer Dense forward pass and centroid quantization against static weights.
#[test]
fn test_vae_dense_layers() {
    let descriptors = vec![
        [0.0f32; 10],
        [
            0.5,
            -0.2,
            0.1,
            0.8,
            -0.4,
            0.3,
            0.9,
            5.0,
            1.0,
            std::f32::consts::LN_2,
        ],
        [0.0f32; 10],
    ];
    let mask = vec![true, false, true];

    let states = encode_descriptors(&descriptors, &mask);
    assert_eq!(states.len(), 3);
    assert_eq!(
        states[0], 2,
        "Masked position 0 must map to invalid state index 2 ('D')"
    );
    assert_eq!(
        states[2], 2,
        "Masked position 2 must map to invalid state index 2 ('D')"
    );
    assert!(
        states[1] < 20,
        "Unmasked state index must be in range 0..20"
    );
}

/// Test mapping centroid state indices (0..19) to 3Di characters ('A'..='V'), and masked states mapping to 'D'.
#[test]
fn test_alphabet_mapping() {
    let states: Vec<usize> = (0..20).collect();
    let mask = vec![false; 20];
    let seq = build_sequence(&states, &mask);
    assert_eq!(seq, "ACDEFGHIKLMNPQRSTVWY");

    let states_with_mask = vec![0, 1, 5, 3];
    let mask_flags = vec![false, true, false, false];
    let seq_masked = build_sequence(&states_with_mask, &mask_flags);
    assert_eq!(seq_masked, "ADGE");
}

// ============================================================================
// Tier 2 — Boundary & Corner Cases
// ============================================================================

/// Test behavior on chains with N < 3 residues (should produce sequence of 'D's of length N).
#[test]
fn test_short_chains() {
    let seq0 = encode_atoms(&[], &[], &[], &[]);
    assert_eq!(seq0, "", "Empty chain should produce empty string");

    let seq1 = encode_atoms(
        &[[0.0, 0.0, 0.0]],
        &[[1.0, 0.0, 0.0]],
        &[[-1.0, 0.0, 0.0]],
        &[[0.0, 1.0, 0.0]],
    );
    assert_eq!(seq1, "D", "Length 1 chain should produce 'D'");

    let seq2 = encode_atoms(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        &[[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]],
        &[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        &[[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
    );
    assert_eq!(seq2, "DD", "Length 2 chain should produce 'DD'");
}

/// Verify first (index 0) and last (index N-1) residues always map to state index 2 ('D').
#[test]
fn test_terminal_masking() {
    let ca = vec![
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ];
    let cb = vec![
        [0.0, 1.0, 0.0],
        [1.5, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [4.5, 1.0, 0.0],
        [6.0, 1.0, 0.0],
    ];
    let n = vec![
        [-0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.5, 0.0, 0.0],
    ];
    let c = vec![
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [6.5, 0.0, 0.0],
    ];

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 5);
    assert!(seq.starts_with('D'), "First residue (index 0) must be 'D'");
    assert!(seq.ends_with('D'), "Last residue (index N-1) must be 'D'");
}

/// Test handling of NaN or missing backbone coordinates resulting in masked residues ('D').
#[test]
fn test_missing_atoms_nan() {
    let ca = vec![
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [f32::NAN, f32::NAN, f32::NAN], // missing CA at index 2
        [4.5, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ];
    let cb = vec![
        [0.0, 1.0, 0.0],
        [1.5, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [4.5, 1.0, 0.0],
        [6.0, 1.0, 0.0],
    ];
    let n = vec![
        [-0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.5, 0.0, 0.0],
    ];
    let c = vec![
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [6.5, 0.0, 0.0],
    ];

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 5);
    assert_eq!(&seq[2..3], "D", "Residue with NaN CA must map to 'D'");
}

/// Test partner selection behavior when distances are identical or equal.
#[test]
fn test_distance_ties() {
    let vc = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], // equidistant to 1 (dist=1) and 3 (dist=1)
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let valid_mask = vec![true; 5];

    let partners = find_partner_indices(&vc, &valid_mask);
    assert_eq!(partners.len(), 5);
    assert!(
        partners[2] == 1 || partners[2] == 3,
        "Equidistant partner selection for index 2 must choose either 1 or 3, got {}",
        partners[2]
    );
}

// ============================================================================
// Tier 3 — Cross-Feature Combinations
// ============================================================================

/// Test sequence encoding with a mixture of Glycine residues (reconstructed Cb)
/// and masked/missing residues within the same chain.
#[test]
fn test_masked_and_glycine_combination() {
    let ca = vec![
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [7.5, 0.0, 0.0],
    ];
    // Residue 1 has Glycine Cb = Ca, Residue 3 has missing Cb (NaN), Residue 4 has missing N (NaN)
    let cb = vec![
        [0.0, 1.0, 0.0],
        [1.5, 0.0, 0.0], // Glycine (Cb == Ca)
        [3.0, 1.0, 0.0],
        [f32::NAN, f32::NAN, f32::NAN], // missing Cb
        [6.0, 1.0, 0.0],
        [7.5, 1.0, 0.0],
    ];
    let n = vec![
        [-0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [f32::NAN, f32::NAN, f32::NAN], // missing N at residue 4
        [7.0, 0.0, 0.0],
    ];
    let c = vec![
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [6.5, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    ];

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 6);
    assert_eq!(&seq[0..1], "D", "Residue 0 must be 'D'");
    assert_eq!(&seq[4..5], "D", "Residue 4 with missing N must be 'D'");
    assert_eq!(&seq[5..6], "D", "Residue 5 (terminal) must be 'D'");
}

// ============================================================================
// Tier 4 — Real-World PDB Structure Validation
// ============================================================================

type BackboneCoords = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>);

/// Helper function to parse PDB structure files from `tests/data/`
/// using `pdbtbx` (or raw PDB ATOM parsing) and extract backbone coordinates
/// for a specified chain ID.
fn parse_pdb_chain_coords(filename: &str, target_chain: &str) -> BackboneCoords {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("data");
    path.push(filename);

    parse_pdb_file(&path, target_chain)
}

fn parse_pdb_file(path: &Path, target_chain: &str) -> BackboneCoords {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open PDB test file {:?}: {}", path, e));
    let reader = BufReader::new(file);

    struct ResidueData {
        res_num: i32,
        icode: char,
        ca: Option<[f32; 3]>,
        cb: Option<[f32; 3]>,
        n: Option<[f32; 3]>,
        c: Option<[f32; 3]>,
    }

    let mut residues: Vec<ResidueData> = Vec::new();

    for line in reader.lines().map_while(Result::ok) {
        if line.starts_with("ATOM  ") || line.starts_with("HETATM") {
            if line.len() < 54 {
                continue;
            }
            let chain_id = &line[21..22];
            if chain_id != target_chain {
                continue;
            }

            let alt_loc = line.chars().nth(16).unwrap_or(' ');
            // Retain standard atoms or 'A' / last alt location
            if alt_loc != ' ' && alt_loc != 'A' && alt_loc != 'B' {
                // If another altloc is present, we still keep if no primary exists
            }

            let res_num: i32 = line[22..26].trim().parse().unwrap_or(i32::MIN);
            let icode = line.chars().nth(26).unwrap_or(' ');

            let x: f32 = line[30..38].trim().parse().unwrap_or(f32::NAN);
            let y: f32 = line[38..46].trim().parse().unwrap_or(f32::NAN);
            let z: f32 = line[46..54].trim().parse().unwrap_or(f32::NAN);

            let last_res = residues.last_mut();
            let res = match last_res {
                Some(r) if r.res_num == res_num && r.icode == icode => r,
                _ => {
                    residues.push(ResidueData {
                        res_num,
                        icode,
                        ca: None,
                        cb: None,
                        n: None,
                        c: None,
                    });
                    residues.last_mut().unwrap()
                }
            };

            let coord = [x, y, z];
            let atom_trimmed = line[12..16].trim();
            let is_ca = atom_trimmed == "CA" || atom_trimmed == "CA A" || atom_trimmed == "CA B";
            let is_cb = atom_trimmed == "CB" || atom_trimmed == "CB A" || atom_trimmed == "CB B";
            let is_n = atom_trimmed == "N" || atom_trimmed == "N A" || atom_trimmed == "N B";
            let is_c = atom_trimmed == "C" || atom_trimmed == "C A" || atom_trimmed == "C B";

            if is_ca {
                if res.ca.is_none() || alt_loc == 'A' || alt_loc == ' ' {
                    res.ca = Some(coord);
                }
            } else if is_cb {
                if res.cb.is_none() || alt_loc == 'A' || alt_loc == ' ' {
                    res.cb = Some(coord);
                }
            } else if is_n {
                if res.n.is_none() || alt_loc == 'A' || alt_loc == ' ' {
                    res.n = Some(coord);
                }
            } else if is_c && (res.c.is_none() || alt_loc == 'A' || alt_loc == ' ') {
                res.c = Some(coord);
            }
        }
    }

    let mut ca_list = Vec::new();
    let mut cb_list = Vec::new();
    let mut n_list = Vec::new();
    let mut c_list = Vec::new();

    for res in residues {
        if let Some(ca) = res.ca {
            ca_list.push(ca);
            cb_list.push(res.cb.unwrap_or([f32::NAN, f32::NAN, f32::NAN]));
            n_list.push(res.n.unwrap_or([f32::NAN, f32::NAN, f32::NAN]));
            c_list.push(res.c.unwrap_or([f32::NAN, f32::NAN, f32::NAN]));
        }
    }

    (ca_list, cb_list, n_list, c_list)
}

/// Test 1xso (chain A): expected 3Di sequence validation.
#[test]
fn test_pdb_1xso_chain_a() {
    let (ca, cb, n, c) = parse_pdb_chain_coords("1xso.pdb", "A");
    let seq = encode_atoms(&ca, &cb, &n, &c);
    let expected = "DKKKWWKDFPDPKTKIKIWDDDDLFKIKIWMKIFQADFDKKWKWWACAQDCPVTVVVSHFGAAPPDFWDFAQPDPRHGLTGDFIFGDDPRMTTDMDIHNSAGCDDPNRQQRIKMFIANAGQCGLPPPDPVSRGTSPRDDTRIMTGMHGDD";
    assert_eq!(
        seq, expected,
        "3Di encoding for 1xso chain A does not match expected output"
    );
}

/// Test 3bww (chain A): expected 3Di sequence validation.
#[test]
fn test_pdb_3bww_chain_a() {
    let (ca, cb, n, c) = parse_pdb_chain_coords("3bww.pdb", "A");
    let seq = encode_atoms(&ca, &cb, &n, &c);
    let expected = "DKDFFEAAEDDLVCLVVLLPPPACPQRQAYEDALVVQVPDDPVSVVSVVNSLVHHAYAYEYEAQQLLDDPQGDVVSLVSVLVCCVVSVPQEYEYENDPPDADALDVVSLVSSLVSQLVSCVSSVGAYAYEDAADQDHDPRHPDDVLVSRQSNCVSNVHAHAYELVRLVRCCVRPVPDDSLVSLVRHPLQRHQHYEYQVVSVVSVLVNLVDHQAHHYYYHDYPDDVVVNSVVRVVSRVSNVVSCVVVVHYIDMD";
    assert_eq!(
        seq, expected,
        "3Di encoding for 3bww chain A does not match expected output"
    );
}

/// Test 3bww.masked (chain A): expected 3Di sequence validation with missing regions.
#[test]
fn test_pdb_3bww_masked_chain_a() {
    let (ca, cb, n, c) = parse_pdb_chain_coords("3bww.masked.pdb", "A");
    let seq = encode_atoms(&ca, &cb, &n, &c);
    let expected = "DKDFFEAAEDDLVCLVVLLPPPACPQRQAYEDALVVQVPDDPVSVVSVVNSLVHHAYAYEYEAQQLDDDPQGDVVSLVSVLVCCVVSVPQEYEYENDPPDADALDPVDDDSSLVSQLVSCVSSVGAYAYEDAADQDHDPRHPDDVLVSRQVSCVSNVHAHAYELVRLVRCCVRPVPDDSLVSLVRHPLQRHQHYEYQVVSVVSVLVNLVDHQAHHYYYHDYPDDVVVNSVVRVVSRVSNVVSCVVVVHYIDMD";
    assert_eq!(
        seq, expected,
        "3Di encoding for 3bww.masked chain A does not match expected output"
    );
}

/// Test 8crb (chain A): expected 3Di sequence validation.
#[test]
fn test_pdb_8crb_chain_a() {
    let (ca, cb, n, c) = parse_pdb_chain_coords("8crb.pdb", "A");
    let seq = encode_atoms(&ca, &cb, &n, &c);
    let expected = "DWAKDKDWADEDAAQAKTKIKMATPPDLLQDFFKFKWFDAPPDDIDGQAPGACPSPPLADDVHHHHGKGWHDDSVRRMIMIMGGNDDQVVFGKMKMFTADDADPQVVVPDGDDTDDMHDIDTYGHPPDDFFAWDKDKDQDDPVPCPVQKPKIKMKTDDGDDDDKDKAWLVNPGDPQKDDFDWDADPVRGIIDMIIGMDGNVCFQVGFTKIWMAGVVVRDIDIDGGHD";
    assert_eq!(
        seq, expected,
        "3Di encoding for 8crb chain A does not match expected output"
    );
}

/// Test 8crb (chain B): expected 3Di sequence validation.
#[test]
fn test_pdb_8crb_chain_b() {
    let (ca, cb, n, c) = parse_pdb_chain_coords("8crb.pdb", "B");
    let seq = encode_atoms(&ca, &cb, &n, &c);
    let expected = "DAAKDFDQQEEEAAQAKDKGWIFAADVPPVPDAFWKWWDAPPDDIDTAADPNQAGDPVDHSQKGWDADHGITIIMGGRDDNSRQGFIWRAQPDDPDHNGHTDDTHGYYHCPDDQDDKDKDWDDAAVVVLVVLFGKTKIKIDDGDDPPKDKFKDLQNHTDDAQWDWDDWDLDPVRTIMTMIIRRDGVVSCVVSQKMKMWIDDDVHTDIDMDGNVVHD";
    assert_eq!(
        seq, expected,
        "3Di encoding for 8crb chain B does not match expected output"
    );
}

/// Test 8crb (chain C): expected 3Di sequence validation.
#[test]
fn test_pdb_8crb_chain_c() {
    let (ca, cb, n, c) = parse_pdb_chain_coords("8crb.pdb", "C");
    let seq = encode_atoms(&ca, &cb, &n, &c);
    let expected = "DPCVLVVLVLQLVLVVLLLVVVVVVLVVCVVVLFKDWQDPVHDWQLACVSPDHDCPDCCSVPGSNNVQQCPKPLDDVTATNQSVQQIDDGDLDHDDDDDTIQGCPPPVRCSVVVVVVSVVSVVVSVVSCVVSVVVVVVD";
    assert_eq!(
        seq, expected,
        "3Di encoding for 8crb chain C does not match expected output"
    );
}
