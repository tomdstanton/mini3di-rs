//! Virtual center calculation and Cb position approximation.

use nalgebra::Vector3;

/// Scaling factor for Cb approximation bond geometry (in Angstroms).
pub const DISTANCE_ALPHA_BETA: f32 = 1.5336;

/// Scaling factor for Virtual Center vector from Ca (in Angstroms).
pub const DISTANCE_ALPHA_V: f32 = 2.0;

/// Normal rotation angle theta in radians (270 degrees = 3 * pi / 2).
pub const THETA_RAD: f32 = std::f32::consts::FRAC_PI_2 * 3.0;

/// Dihedral rotation angle tau in radians (0 degrees).
pub const TAU_RAD: f32 = 0.0;

/// Safely normalizes a 3D vector. Returns a NAN vector if the norm is 0 or non-finite.
#[inline]
fn normalize(v: Vector3<f32>) -> Vector3<f32> {
    let norm = v.norm();
    if norm > 0.0 && norm.is_finite() {
        v / norm
    } else {
        Vector3::new(f32::NAN, f32::NAN, f32::NAN)
    }
}

/// Approximates the position of the Cb atom from backbone Ca, N, and C atom coordinates.
///
/// # Arguments
/// * `ca` - C-alpha position vector
/// * `n`  - Nitrogen position vector
/// * `c`  - Carbon position vector
///
/// # Returns
/// Approximated Cb position vector.
pub fn approximate_cb_position(
    ca: &Vector3<f32>,
    n: &Vector3<f32>,
    c: &Vector3<f32>,
) -> Vector3<f32> {
    let v1 = normalize(c - ca);
    let v2 = normalize(n - ca);
    let v3 = v1 / 3.0;

    let b1 = v2 + v3;
    let b2 = v1.cross(&b1);
    let u1 = normalize(b1);
    let u2 = normalize(b2);

    let factor = (8.0f32).sqrt() / 3.0;
    let dir = factor * ((-u1 / 2.0) - (u2 * (3.0f32).sqrt() / 2.0)) - v3;
    ca + dir * DISTANCE_ALPHA_BETA
}

/// Computes the 3D virtual center coordinate for a single residue.
///
/// Applies Rodrigues' rotation formula around normal axis (theta = 270 deg)
/// and dihedral axis (tau = 0 deg), scaled by 2.0 from Ca.
pub fn compute_single_virtual_center(
    ca: &Vector3<f32>,
    cb: &Vector3<f32>,
    n: &Vector3<f32>,
) -> Vector3<f32> {
    let mut v = cb - ca;
    let a = cb - ca;
    let b = n - ca;

    // Normal angle rotation (theta = 270 deg)
    let theta_rad = 270.0f64.to_radians();
    let cos_theta = theta_rad.cos() as f32;
    let sin_theta = theta_rad.sin() as f32;
    let k_norm = normalize(a.cross(&b));
    v = v * cos_theta + k_norm.cross(&v) * sin_theta + k_norm * k_norm.dot(&v) * (1.0 - cos_theta);

    // Dihedral angle rotation (tau = 0 deg)
    let tau_rad = 0.0f64.to_radians();
    let cos_tau = tau_rad.cos() as f32;
    let sin_tau = tau_rad.sin() as f32;
    let k_dih = normalize(n - ca);
    v = v * cos_tau + k_dih.cross(&v) * sin_tau + k_dih * k_dih.dot(&v) * (1.0 - cos_tau);

    // Apply final scale factor (2.0) and shift to Ca
    ca + v * DISTANCE_ALPHA_V
}

/// Computes virtual centers and validity mask for a sequence of backbone atom coordinates.
///
/// # Arguments
/// * `ca` - C-alpha atom coordinates [N, 3]
/// * `cb` - C-beta atom coordinates [N, 3]
/// * `n`  - Nitrogen atom coordinates [N, 3]
/// * `c`  - Carbon atom coordinates [N, 3]
///
/// # Returns
/// A tuple `(virtual_centers, valid_mask)`:
/// - `virtual_centers`: Vec of 3D coordinates `[f32; 3]` for each residue.
/// - `valid_mask`: Vec of booleans indicating whether each residue has valid backbone coordinates.
pub fn compute_virtual_centers(
    ca: &[[f32; 3]],
    cb: &[[f32; 3]],
    n: &[[f32; 3]],
    c: &[[f32; 3]],
) -> (Vec<[f32; 3]>, Vec<bool>) {
    let len = ca.len();
    let mut virtual_centers = Vec::with_capacity(len);
    let mut valid_mask = Vec::with_capacity(len);

    for i in 0..len {
        let ca_arr = ca[i];
        let cb_arr = cb[i];
        let n_arr = n[i];
        let c_arr = c[i];

        let ca_valid = !ca_arr[0].is_nan() && !ca_arr[1].is_nan() && !ca_arr[2].is_nan();
        let n_valid = !n_arr[0].is_nan() && !n_arr[1].is_nan() && !n_arr[2].is_nan();
        let c_valid = !c_arr[0].is_nan() && !c_arr[1].is_nan() && !c_arr[2].is_nan();

        if !ca_valid || !n_valid || !c_valid {
            virtual_centers.push([f32::NAN, f32::NAN, f32::NAN]);
            valid_mask.push(false);
            continue;
        }

        let ca_vec = Vector3::new(ca_arr[0], ca_arr[1], ca_arr[2]);
        let n_vec = Vector3::new(n_arr[0], n_arr[1], n_arr[2]);
        let c_vec = Vector3::new(c_arr[0], c_arr[1], c_arr[2]);

        let cb_is_nan = cb_arr[0].is_nan() || cb_arr[1].is_nan() || cb_arr[2].is_nan();
        let cb_is_gly = cb_arr == ca_arr;

        let cb_vec = if cb_is_nan || cb_is_gly {
            approximate_cb_position(&ca_vec, &n_vec, &c_vec)
        } else {
            Vector3::new(cb_arr[0], cb_arr[1], cb_arr[2])
        };

        let vc = compute_single_virtual_center(&ca_vec, &cb_vec, &n_vec);
        let is_vc_valid = !vc.x.is_nan()
            && vc.x.is_finite()
            && !vc.y.is_nan()
            && vc.y.is_finite()
            && !vc.z.is_nan()
            && vc.z.is_finite();

        if is_vc_valid {
            virtual_centers.push([vc.x, vc.y, vc.z]);
            valid_mask.push(true);
        } else {
            virtual_centers.push([f32::NAN, f32::NAN, f32::NAN]);
            valid_mask.push(false);
        }
    }

    (virtual_centers, valid_mask)
}
