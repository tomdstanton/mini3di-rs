//! Pairwise distance matrix calculation and nearest partner residue resolution.

/// Finds the nearest partner residue index J_i for each residue i based on virtual centers.
///
/// # Arguments
/// * `vc` - Virtual center coordinates [N, 3]
/// * `valid_mask` - Validity mask indicating which virtual centers are valid
///
/// # Returns
/// Vector of partner residue indices `J_i` for i in 0..N.
pub fn find_partner_indices(vc: &[[f32; 3]], valid_mask: &[bool]) -> Vec<usize> {
    let n = vc.len();
    let mut partners = vec![0; n];
    if n == 0 {
        return partners;
    }

    for i in 0..n {
        let is_i_terminal = i == 0 || i == n - 1;
        let is_i_valid =
            valid_mask.get(i).copied().unwrap_or(false) && !is_i_terminal && !vc[i][0].is_nan();

        if !is_i_valid {
            partners[i] = i;
            continue;
        }

        let vc_i = vc[i];
        let mut min_dist = f32::INFINITY;
        let mut best_j = i;

        for (j, vc_j) in vc.iter().enumerate().take(n - 1).skip(1) {
            if i == j || !valid_mask.get(j).copied().unwrap_or(false) || vc_j[0].is_nan() {
                continue;
            }

            let dx = vc_i[0] - vc_j[0];
            let dy = vc_i[1] - vc_j[1];
            let dz = vc_i[2] - vc_j[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < min_dist {
                min_dist = dist_sq;
                best_j = j;
            }
        }

        partners[i] = best_j;
    }

    partners
}
