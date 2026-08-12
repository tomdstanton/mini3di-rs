//! High-level 3Di encoder pipeline and sequence generation functions.

use crate::feature_encoder::calc_conformation_descriptors;
use crate::partner_index::find_partner_indices;
use crate::vae::{encode_descriptors, ALPHABET, INVALID_STATE};
use crate::virtual_center::compute_virtual_centers;

/// High-level API to encode backbone 3D atom coordinates (CA, CB, N, C) into a 3Di sequence string.
///
/// # Arguments
/// * `ca` - Calpha atom coordinates `[[x, y, z]; N]`
/// * `cb` - Cbeta atom coordinates `[[x, y, z]; N]` (use NaN for missing CB / Glycine)
/// * `n`  - Backbone Nitrogen atom coordinates `[[x, y, z]; N]`
/// * `c`  - Backbone Carbon atom coordinates `[[x, y, z]; N]`
///
/// # Returns
/// 3Di alphabet sequence string of length `N`. Empty string if `ca` is empty.
pub fn encode_atoms(ca: &[[f32; 3]], cb: &[[f32; 3]], n: &[[f32; 3]], c: &[[f32; 3]]) -> String {
    let len = ca.len();
    if len == 0 {
        return String::new();
    }
    if len < 3 {
        return "D".repeat(len);
    }

    // Step 1: Compute Virtual Centers (approximating missing CB if necessary)
    let (vc, vc_valid) = compute_virtual_centers(ca, cb, n, c);

    // Step 2: Find Nearest Partner Residue Indices
    let partner_indices = find_partner_indices(&vc, &vc_valid);

    // Step 3: Calculate 10D Geometric Conformation Descriptors & Residue Mask
    let (descriptors, mask) = calc_conformation_descriptors(ca, &partner_indices, &vc_valid);

    // Step 4: Pass through 3-Layer VAE Neural Network & Quantize to Centroid States
    let states = encode_descriptors(&descriptors, &mask);

    // Step 5: Convert Centroid States to 3Di Alphabet String
    build_sequence(&states, &mask)
}

/// Converts centroid state indices and mask array to a 3Di sequence string.
///
/// # Arguments
/// * `states` - Array of centroid state indices (0..19).
/// * `mask` - Array of boolean flags indicating masked/invalid residues.
///
/// # Returns
/// 3Di sequence string where unmasked valid states map to ALPHABET characters,
/// and masked positions map to 'D' (INVALID_STATE = 2).
pub fn build_sequence(states: &[usize], mask: &[bool]) -> String {
    let mut result = String::with_capacity(states.len());
    for (i, &state) in states.iter().enumerate() {
        let is_masked = mask.get(i).copied().unwrap_or(true);
        if is_masked || state >= ALPHABET.len() {
            result.push(ALPHABET[INVALID_STATE]);
        } else {
            result.push(ALPHABET[state]);
        }
    }
    result
}
