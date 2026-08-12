//! Adversarial stress test suite for `mini3di-rs` core math and VAE encoder.
//!
//! Tests extreme coordinate inputs:
//! - Collinear atoms (CA, CB, N, C on a straight line)
//! - Zero vectors and coincident atom coordinates
//! - Huge floating point values (1e30, 1e38, f32::MAX, subnormals)
//! - Infinite values (f32::INFINITY, f32::NEG_INFINITY)
//! - All-NaN inputs
//! - Single residue and short residue chains (N = 0, 1, 2, 3, 4)
//! - Mismatched atom slice lengths
//! - Large protein chain stress test (N = 10,000 residues)
//! - Pseudo-random generative property test (1,000 iterations)

use mini3di_rs::{
    calc_conformation_descriptors, compute_virtual_centers, encode_atoms, encode_descriptors,
    find_partner_indices, ALPHABET,
};

/// 1. Collinear Atoms: Test behavior when backbone atoms lie on a straight line.
#[test]
fn test_adv_collinear_atoms() {
    // CA, N, C perfectly collinear along the X-axis
    let ca = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let cb = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let n = vec![
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [3.5, 0.0, 0.0],
    ];
    let c = vec![
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert_eq!(vc.len(), 5);
    for (i, v) in vc.iter().enumerate() {
        assert!(
            !valid[i],
            "Virtual center {} must be invalid due to collinearity degeneracy",
            i
        );
        assert!(
            v[0].is_nan() && v[1].is_nan() && v[2].is_nan(),
            "VC {} must contain NaN",
            i
        );
    }

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq, "DDDDD");
}

/// 2. Zero Vectors / Coincident Atoms: Test when all atom positions are at origin [0, 0, 0].
#[test]
fn test_adv_coincident_zero_vectors() {
    let ca = vec![[0.0, 0.0, 0.0]; 6];
    let cb = vec![[0.0, 0.0, 0.0]; 6];
    let n = vec![[0.0, 0.0, 0.0]; 6];
    let c = vec![[0.0, 0.0, 0.0]; 6];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert_eq!(vc.len(), 6);
    for (i, v) in vc.iter().enumerate() {
        assert!(!valid[i]);
        assert!(v[0].is_nan() && v[1].is_nan() && v[2].is_nan());
    }

    let partners = find_partner_indices(&vc, &valid);
    assert_eq!(partners.len(), 6);

    let (desc, _mask) = calc_conformation_descriptors(&ca, &partners, &valid);
    assert_eq!(desc.len(), 6);

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq, "DDDDDD");
}

/// 3. Huge Floating Point Values & Subnormals.
#[test]
fn test_adv_huge_floating_point_values() {
    let huge = 1e30f32;
    let ca = vec![
        [0.0, 0.0, 0.0],
        [huge, 0.0, 0.0],
        [huge * 2.0, huge, 0.0],
        [huge * 3.0, 0.0, huge],
        [huge * 4.0, huge, huge],
    ];
    let cb = ca.clone();
    let n = ca.clone();
    let c = ca.clone();

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 5);
    assert!(seq.chars().all(|ch| ALPHABET.contains(&ch)));

    // Subnormal values
    let tiny = 1e-38f32;
    let ca_tiny = vec![
        [0.0, 0.0, 0.0],
        [tiny, 0.0, 0.0],
        [tiny * 2.0, tiny, 0.0],
        [tiny * 3.0, 0.0, tiny],
        [tiny * 4.0, tiny, tiny],
    ];
    let seq_tiny = encode_atoms(&ca_tiny, &ca_tiny, &ca_tiny, &ca_tiny);
    assert_eq!(seq_tiny.len(), 5);
    assert!(seq_tiny.chars().all(|ch| ALPHABET.contains(&ch)));
}

/// 4. Infinite Values (f32::INFINITY and f32::NEG_INFINITY).
#[test]
fn test_adv_infinity_values() {
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    let ca = vec![
        [0.0, 0.0, 0.0],
        [inf, 0.0, 0.0],
        [0.0, neg_inf, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ];
    let cb = ca.clone();
    let n = ca.clone();
    let c = ca.clone();

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 5);
    assert!(seq.chars().all(|ch| ALPHABET.contains(&ch)));
}

/// 5. All-NaN Inputs.
#[test]
fn test_adv_all_nan_inputs() {
    let nan = f32::NAN;
    let ca = vec![[nan, nan, nan]; 5];
    let cb = vec![[nan, nan, nan]; 5];
    let n = vec![[nan, nan, nan]; 5];
    let c = vec![[nan, nan, nan]; 5];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert_eq!(vc.len(), 5);
    assert!(
        valid.iter().all(|&v| !v),
        "All validity flags must be false for all-NaN input"
    );

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq, "DDDDD", "All-NaN input must yield all 'D's");
}

/// 6. Single Residue and Short Residue Chains (N = 0, 1, 2, 3, 4).
#[test]
fn test_adv_short_chains_boundaries() {
    let p0 = encode_atoms(&[], &[], &[], &[]);
    assert_eq!(p0, "");

    let p1 = encode_atoms(
        &[[0.0, 0.0, 0.0]],
        &[[0.0, 1.0, 0.0]],
        &[[-1.0, 0.0, 0.0]],
        &[[1.0, 0.0, 0.0]],
    );
    assert_eq!(p1, "D");

    let p2 = encode_atoms(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        &[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        &[[-1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        &[[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
    );
    assert_eq!(p2, "DD");

    let ca3 = vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let cb3 = vec![[0.0, 1.0, 0.0], [1.5, 1.0, 0.0], [3.0, 1.0, 0.0]];
    let n3 = vec![[-0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]];
    let c3 = vec![[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.0, 0.0]];
    let p3 = encode_atoms(&ca3, &cb3, &n3, &c3);
    assert_eq!(p3.len(), 3);
    assert!(p3.starts_with('D') && p3.ends_with('D'));

    let ca4 = vec![
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ];
    let cb4 = vec![
        [0.0, 1.0, 0.0],
        [1.5, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [4.5, 1.0, 0.0],
    ];
    let n4 = vec![
        [-0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let c4 = vec![
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ];
    let p4 = encode_atoms(&ca4, &cb4, &n4, &c4);
    assert_eq!(p4.len(), 4);
    assert!(p4.starts_with('D') && p4.ends_with('D'));
}

/// 7. Large Protein Chain Stress Test (N = 10,000 residues).
#[test]
fn test_adv_large_protein_chain_stress() {
    let n_residues = 10_000;
    let mut ca = Vec::with_capacity(n_residues);
    let mut cb = Vec::with_capacity(n_residues);
    let mut n = Vec::with_capacity(n_residues);
    let mut c = Vec::with_capacity(n_residues);

    for i in 0..n_residues {
        let t = i as f32 * 0.1;
        let x = t.cos() * 10.0;
        let y = t.sin() * 10.0;
        let z = t * 1.5;

        ca.push([x, y, z]);
        cb.push([x + 1.0, y + 1.0, z + 0.5]);
        n.push([x - 0.5, y, z]);
        c.push([x + 0.5, y, z]);
    }

    let start = std::time::Instant::now();
    let seq = encode_atoms(&ca, &cb, &n, &c);
    let elapsed = start.elapsed();

    assert_eq!(seq.len(), n_residues);
    assert!(seq.starts_with('D') && seq.ends_with('D'));
    println!("Encoded {} residues in {:?}", n_residues, elapsed);
    assert!(
        elapsed.as_secs() < 5,
        "10k residue encoding took too long: {:?}",
        elapsed
    );
}

/// 8. VAE Extreme Descriptors: Test encode_descriptors with NaN, Inf, and Max float descriptors.
#[test]
fn test_adv_vae_extreme_descriptors() {
    let extreme_desc = vec![
        [f32::NAN; 10],
        [f32::INFINITY; 10],
        [f32::NEG_INFINITY; 10],
        [f32::MAX; 10],
        [f32::MIN; 10],
    ];
    let mask = vec![false; 5]; // unmasked

    let states = encode_descriptors(&extreme_desc, &mask);
    assert_eq!(states.len(), 5);
    // All extreme/non-finite descriptors should quantify safely (or return INVALID_STATE)
    for &st in &states {
        assert!(
            st < ALPHABET.len(),
            "State index {} must be within ALPHABET bounds",
            st
        );
    }
}

/// 9. Pseudo-Random Generative Property Test (Fuzzing 1,000 iterations).
#[test]
fn test_adv_generative_fuzz_property() {
    // Simple LCG pseudo-random generator
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn next_f32(&mut self) -> f32 {
            let u = self.next_u32();
            let mode = u % 10;
            match mode {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => 0.0,
                4 => 1e30,
                5 => -1e30,
                6 => 1e-38,
                _ => (u as f32 / u32::MAX as f32) * 200.0 - 100.0,
            }
        }
        fn next_coord(&mut self) -> [f32; 3] {
            [self.next_f32(), self.next_f32(), self.next_f32()]
        }
    }

    let mut rng = Lcg(0x123456789ABCDEF);

    for iter in 0..1000 {
        let len = (rng.next_u32() % 50) as usize; // chain length 0..49
        let ca: Vec<[f32; 3]> = (0..len).map(|_| rng.next_coord()).collect();
        let cb: Vec<[f32; 3]> = (0..len).map(|_| rng.next_coord()).collect();
        let n: Vec<[f32; 3]> = (0..len).map(|_| rng.next_coord()).collect();
        let c: Vec<[f32; 3]> = (0..len).map(|_| rng.next_coord()).collect();

        let seq = encode_atoms(&ca, &cb, &n, &c);
        assert_eq!(
            seq.len(),
            len,
            "Iteration {}: Output sequence length {} must equal input length {}",
            iter,
            seq.len(),
            len
        );

        for ch in seq.chars() {
            assert!(
                ALPHABET.contains(&ch),
                "Iteration {}: Invalid character '{}' in output sequence",
                iter,
                ch
            );
        }
    }
}

/// 10. Mismatched Slice Lengths: Test catch_unwind on mismatched atom array lengths.
#[test]
fn test_adv_mismatched_slice_lengths() {
    let ca = vec![[0.0, 0.0, 0.0]; 5];
    let cb = vec![[0.0, 1.0, 0.0]; 3]; // shorter length
    let n = vec![[-0.5, 0.0, 0.0]; 5];
    let c = vec![[0.5, 0.0, 0.0]; 5];

    let result = std::panic::catch_unwind(|| {
        compute_virtual_centers(&ca, &cb, &n, &c);
    });

    // Document whether mismatched slice lengths trigger a panic or fail gracefully.
    assert!(
        result.is_err(),
        "Mismatched slice lengths (cb.len < ca.len) should trigger an index out of bounds panic"
    );
}

/// 11. Exact Distance Ties: Verify partner selection deterministically chooses lowest index j when distances are equal.
#[test]
fn test_adv_exact_distance_ties_lowest_index_selection() {
    let vc = vec![
        [0.0, 0.0, -10.0], // 0 (terminal)
        [0.0, 0.0, 0.0],   // 1
        [0.0, 1.0, 0.0],   // 2
        [0.0, 2.0, 0.0],   // 3 (equidistant to 2 as 1 is)
        [0.0, 0.0, 10.0],  // 4 (terminal)
    ];
    let valid = vec![true; 5];

    let partners = find_partner_indices(&vc, &valid);
    assert_eq!(
        partners[2], 1,
        "Distance tie must select lowest index candidate (1)"
    );
}

/// 12. 6-Residue Mask Propagation: Verify NaN in residue i masks residues i-1, i, i+1 and partner j-1, j, j+1.
#[test]
fn test_adv_mask_propagation_6_residues() {
    let mut ca = vec![
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
        [11.4, 0.0, 0.0],
        [15.2, 0.0, 0.0],
        [19.0, 0.0, 0.0],
        [22.8, 0.0, 0.0],
    ];
    let cb = vec![[f32::NAN; 3]; 7];
    let n = ca
        .iter()
        .map(|p| [p[0] - 1.0, p[1], p[2]])
        .collect::<Vec<_>>();
    let c = ca
        .iter()
        .map(|p| [p[0] + 1.0, p[1], p[2]])
        .collect::<Vec<_>>();

    ca[3] = [f32::NAN, f32::NAN, f32::NAN];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert!(!valid[3], "Residue 3 must be marked invalid");

    let partners = find_partner_indices(&vc, &valid);
    let (_descriptors, mask) = calc_conformation_descriptors(&ca, &partners, &valid);

    assert!(
        mask[2],
        "Residue 2 must be masked due to neighbor 3 being NaN"
    );
    assert!(mask[3], "Residue 3 must be masked due to being NaN");
    assert!(
        mask[4],
        "Residue 4 must be masked due to neighbor 3 being NaN"
    );

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 7);
    assert_eq!(&seq[2..5], "DDD", "Residues 2..5 must be 'DDD'");
}

/// 13. Feature Descriptor Clamping and Log Distance verification.
#[test]
fn test_adv_feature_clamping_and_log_dist() {
    let n = 60;
    let mut ca = Vec::with_capacity(n);
    for i in 0..n {
        ca.push([i as f32 * 3.8, 0.0, 0.0]);
    }
    let valid = vec![true; n];
    let mut partner_indices = vec![0; n];
    partner_indices[5] = 55;

    let (desc, mask) = calc_conformation_descriptors(&ca, &partner_indices, &valid);
    assert!(!mask[5], "Residue 5 should be unmasked");

    let d8_clamp = desc[5][8];
    let d9_log = desc[5][9];

    assert_eq!(d8_clamp, 4.0, "d8 must be clamped to 4.0 for delta = 50");

    let expected_d9 = (50.0f32 + 1.0).ln();
    assert!(
        (d9_log - expected_d9).abs() < 1e-5,
        "d9 log distance must equal ln(abs(delta) + 1)"
    );
}

type BackboneCoords = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>);

/// 14. Synthetic Backbone Generator: Helix and Sheet Coordinate Verification.
#[test]
fn test_adv_synthetic_backbone_generator_helix_sheet() {
    fn generate_alpha_helix(n_res: usize) -> BackboneCoords {
        let mut ca = Vec::with_capacity(n_res);
        let mut cb = Vec::with_capacity(n_res);
        let mut n = Vec::with_capacity(n_res);
        let mut c = Vec::with_capacity(n_res);

        let radius = 2.3;
        for i in 0..n_res {
            let angle = i as f32 * 1.745;
            let z = i as f32 * 1.5;
            let x = radius * angle.cos();
            let y = radius * angle.sin();

            ca.push([x, y, z]);
            cb.push([x + 1.2 * angle.cos(), y + 1.2 * angle.sin(), z + 0.5]);
            n.push([x - 0.7, y, z - 0.5]);
            c.push([x + 0.7, y, z + 0.5]);
        }
        (ca, cb, n, c)
    }

    let (ca, cb, n, c) = generate_alpha_helix(30);
    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 30);
    assert_eq!(seq.chars().next(), Some('D'));
    assert_eq!(seq.chars().last(), Some('D'));
    assert!(seq.chars().all(|ch| ALPHABET.contains(&ch)));
}

/// 15. Non-Standard Amino Acids and Missing Cbeta Handling.
#[test]
fn test_adv_non_standard_amino_acids_and_missing_cb() {
    let n_res = 10;
    let mut ca = Vec::with_capacity(n_res);
    let mut cb = Vec::with_capacity(n_res);
    let mut n = Vec::with_capacity(n_res);
    let mut c = Vec::with_capacity(n_res);

    for i in 0..n_res {
        let x = i as f32 * 3.8;
        ca.push([x, 0.0, 0.0]);
        if i == 2 || i == 7 {
            cb.push([f32::NAN, f32::NAN, f32::NAN]);
        } else {
            cb.push([x, 1.5, 0.0]);
        }
        n.push([x - 1.2, 1.2, 0.0]);
        c.push([x + 1.2, -1.2, 0.0]);
    }

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert_eq!(vc.len(), n_res);
    for (i, &is_valid) in valid.iter().enumerate() {
        assert!(
            is_valid,
            "Residue {} virtual center should be valid via Cb approximation",
            i
        );
    }

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), n_res);
    assert!(seq.chars().all(|ch| ALPHABET.contains(&ch)));
}

/// 16. Isolated Valid Residue Surrounded by NaNs.
#[test]
fn test_adv_isolated_valid_residue_surrounded_by_nans() {
    let nan = f32::NAN;
    let mut ca = vec![[nan, nan, nan]; 7];
    let mut cb = vec![[nan, nan, nan]; 7];
    let mut n = vec![[nan, nan, nan]; 7];
    let mut c = vec![[nan, nan, nan]; 7];

    // Only residue 3 has valid backbone coordinates
    ca[3] = [10.0, 10.0, 10.0];
    cb[3] = [10.0, 11.5, 10.0];
    n[3] = [9.0, 10.0, 10.0];
    c[3] = [11.0, 10.0, 10.0];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert!(valid[3], "Residue 3 virtual center should be valid");
    assert!(
        !valid[2] && !valid[4],
        "Neighboring residues must be invalid"
    );

    let partners = find_partner_indices(&vc, &valid);
    assert_eq!(
        partners[3], 3,
        "Isolated residue partner index should default to itself"
    );

    let (_descriptors, mask) = calc_conformation_descriptors(&ca, &partners, &valid);
    assert!(
        mask[3],
        "Residue 3 must be masked due to invalid neighbors (i-1, i+1)"
    );

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(
        seq, "DDDDDDD",
        "All residues in chain with isolated valid residue must yield 'D'"
    );
}

/// 17. All Intermediate Residues Invalid.
#[test]
fn test_adv_all_intermediate_residues_invalid() {
    let nan = f32::NAN;
    let mut ca = vec![[nan, nan, nan]; 6];
    let mut cb = vec![[nan, nan, nan]; 6];
    let mut n = vec![[nan, nan, nan]; 6];
    let mut c = vec![[nan, nan, nan]; 6];

    // Residues 0 and 5 valid, residues 1..4 invalid
    ca[0] = [0.0, 0.0, 0.0];
    cb[0] = [0.0, 1.5, 0.0];
    n[0] = [-1.0, 0.0, 0.0];
    c[0] = [1.0, 0.0, 0.0];
    ca[5] = [20.0, 0.0, 0.0];
    cb[5] = [20.0, 1.5, 0.0];
    n[5] = [19.0, 0.0, 0.0];
    c[5] = [21.0, 0.0, 0.0];

    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert!(valid[0] && valid[5]);
    assert!(!valid[1] && !valid[2] && !valid[3] && !valid[4]);

    let partners = find_partner_indices(&vc, &valid);
    assert_eq!(partners.len(), 6);

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq, "DDDDDD");
}

/// 18. Degenerate Collinear Glycine Residue.
#[test]
fn test_adv_degenerate_collinear_glycine() {
    // Residue 2 is Glycine (CB == CA) and CA, N, C are collinear
    let ca = vec![
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
        [11.4, 0.0, 0.0],
        [15.2, 0.0, 0.0],
    ];
    let cb = vec![
        [0.0, 1.5, 0.0],
        [3.8, 1.5, 0.0],
        [7.6, 0.0, 0.0],
        [11.4, 1.5, 0.0],
        [15.2, 1.5, 0.0],
    ]; // cb[2] == ca[2]
    let n = vec![
        [-1.0, 0.0, 0.0],
        [2.8, 0.0, 0.0],
        [6.6, 0.0, 0.0],
        [10.4, 0.0, 0.0],
        [14.2, 0.0, 0.0],
    ];
    let c = vec![
        [1.0, 0.0, 0.0],
        [4.8, 0.0, 0.0],
        [8.6, 0.0, 0.0],
        [12.4, 0.0, 0.0],
        [16.2, 0.0, 0.0],
    ];

    // Make residue 2 collinear (N=[6.6, 0, 0], CA=[7.6, 0, 0], C=[8.6, 0, 0])
    let (vc, valid) = compute_virtual_centers(&ca, &cb, &n, &c);
    assert!(
        !valid[2],
        "Collinear Glycine residue 2 virtual center must be invalid"
    );
    assert!(vc[2][0].is_nan(), "VC 2 x component must be NaN");

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 5);
    assert_eq!(
        &seq[1..4],
        "DDD",
        "Residues 1..4 must be masked due to collinear Glycine"
    );
}

/// 19. Subnormal Coordinate Differences.
#[test]
fn test_adv_subnormal_coordinate_differences() {
    let tiny = 1e-37f32;
    let ca = vec![
        [0.0, 0.0, 0.0],
        [tiny, tiny, tiny],
        [tiny * 2.0, tiny * 2.0, tiny * 2.0],
        [tiny * 3.0, tiny * 3.0, tiny * 3.0],
        [tiny * 4.0, tiny * 4.0, tiny * 4.0],
    ];
    let cb = ca.clone();
    let n = ca.clone();
    let c = ca.clone();

    let seq = encode_atoms(&ca, &cb, &n, &c);
    assert_eq!(seq.len(), 5);
    assert!(seq.chars().all(|ch| ALPHABET.contains(&ch)));
}

/// 20. VAE Centroid Quantization Tie Breaking.
#[test]
fn test_adv_vae_centroid_quantization_tie_breaking() {
    // Create descriptors that produce identical distances to centroids or extreme 0 descriptors
    let desc = vec![[0.0f32; 10]; 5];
    let mask = vec![false; 5];

    let states = encode_descriptors(&desc, &mask);
    assert_eq!(states.len(), 5);
    // All identical zero descriptors must map deterministically to the same centroid
    assert!(
        states.windows(2).all(|w| w[0] == w[1]),
        "Identical descriptors must map to identical centroid states"
    );
}
