//! Feature encoder module for `mini3di-rs`.
//!
//! Calculates 10-dimensional geometric conformation descriptors for residue pairs
//! and handles 6-residue mask propagation across neighbor and partner positions.

use nalgebra::Vector3;

/// Normalizes a 3D vector.
///
/// If the norm is zero, NaN, or smaller than 1e-12, returns a zero vector
/// to avoid division-by-zero or NaN propagation.
#[inline]
pub fn normalize(v: Vector3<f32>) -> Vector3<f32> {
    let norm_sq = v.norm_squared();
    if norm_sq > 1e-24 && norm_sq.is_finite() {
        v / norm_sq.sqrt()
    } else {
        Vector3::zeros()
    }
}

/// Calculates 10D conformation descriptors and 6-residue descriptor mask.
///
/// # Arguments
/// * `ca` - Backbone Calpha coordinates for each residue (length N).
/// * `partner_indices` - Nearest partner residue index J_i for each residue i (length N).
/// * `valid_mask` - Validity flag for each residue's virtual center / backbone (length N).
///
/// # Returns
/// * `(descriptors, mask)`:
///   - `descriptors`: Vec<[f32; 10]> of length N.
///   - `mask`: Vec<bool> of length N, where `true` indicates the residue is MASKED (invalid/'D').
pub fn calc_conformation_descriptors(
    ca: &[[f32; 3]],
    partner_indices: &[usize],
    valid_mask: &[bool],
) -> (Vec<[f32; 10]>, Vec<bool>) {
    let n = ca.len();
    let mut descriptors = vec![[0.0f32; 10]; n];
    let mut mask = vec![true; n];

    if n < 3 {
        return (descriptors, mask);
    }

    let ca_v: Vec<Vector3<f32>> = ca.iter().map(|p| Vector3::new(p[0], p[1], p[2])).collect();

    // Helper closure matching Python NumPy array indexing with wrap-around (e.g. -1 -> n-1)
    let is_nan = |k: i64| -> bool {
        if n == 0 {
            return true;
        }
        let idx = k.rem_euclid(n as i64) as usize;
        !valid_mask.get(idx).copied().unwrap_or(false)
            || ca[idx][0].is_nan()
            || ca[idx][1].is_nan()
            || ca[idx][2].is_nan()
    };

    for i in 1..(n - 1) {
        let j = partner_indices[i];
        let i_i64 = i as i64;
        let j_i64 = j as i64;

        // 6-residue mask propagation matching Python _create_descriptor_mask:
        let any_nan = is_nan(i_i64 - 1)
            || is_nan(i_i64)
            || is_nan(i_i64 + 1)
            || is_nan(j_i64 - 1)
            || is_nan(j_i64)
            || is_nan(j_i64 + 1);

        if any_nan {
            mask[i] = true;
            descriptors[i] = [0.0; 10];
            continue;
        }

        // Unmasked residue i
        mask[i] = false;

        let j_prev = (j_i64 - 1).rem_euclid(n as i64) as usize;
        let j_next = (j_i64 + 1).rem_euclid(n as i64) as usize;

        // Compute 5 direction vectors (u1..u5)
        let u1 = normalize(ca_v[i] - ca_v[i - 1]);
        let u2 = normalize(ca_v[i + 1] - ca_v[i]);
        let u3 = normalize(ca_v[j] - ca_v[j_prev]);
        let u4 = normalize(ca_v[j_next] - ca_v[j]);
        let u5 = normalize(ca_v[j] - ca_v[i]);

        // Compute 10D descriptor components:
        let d0 = u1.dot(&u2);
        let d1 = u3.dot(&u4);
        let d2 = u1.dot(&u5);
        let d3 = u3.dot(&u5);
        let d4 = u1.dot(&u4);
        let d5 = u2.dot(&u3);
        let d6 = u1.dot(&u3);

        let d7 = (ca_v[i] - ca_v[j]).norm();

        let delta = j_i64 as f32 - i_i64 as f32;
        let d8 = delta.clamp(-4.0, 4.0);

        let d9 = (delta.abs() + 1.0).ln().copysign(delta);

        descriptors[i] = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9];
    }

    // Terminal residues (0 and n-1) are always masked
    mask[0] = true;
    mask[n - 1] = true;

    (descriptors, mask)
}
