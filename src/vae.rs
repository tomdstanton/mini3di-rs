//! VQ-VAE neural network inference and nearest-centroid state quantization.

use crate::weights::{
    CENTROIDS, LAYER1_BIASES, LAYER1_WEIGHTS, LAYER2_BIASES, LAYER2_WEIGHTS, LAYER3_BIASES,
    LAYER3_WEIGHTS,
};

/// 3Di structural alphabet characters (20 standard states + 1 unknown/masked state).
pub const ALPHABET: [char; 21] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y', 'X',
];

/// Index of invalid/masked state in ALPHABET (maps to 'D').
pub const INVALID_STATE: usize = 2;

/// Evaluates the 3-layer Dense feedforward VAE model and assigns centroid state indices.
///
/// # Arguments
/// * `descriptors` - 10D geometric conformation feature descriptors per residue.
/// * `mask` - Boolean mask array where `true` indicates a masked/invalid residue.
///
/// # Returns
/// Vector of state indices (0..19) corresponding to nearest 2D centroids in `ALPHABET`.
/// Masked residues return `INVALID_STATE` (2).
pub fn encode_descriptors(descriptors: &[[f32; 10]], mask: &[bool]) -> Vec<usize> {
    let n = descriptors.len();
    let mut states = Vec::with_capacity(n);

    for (i, desc) in descriptors.iter().enumerate() {
        if mask.get(i).copied().unwrap_or(true) {
            states.push(INVALID_STATE);
            continue;
        }

        // Layer 1: 10 -> 10 Dense + ReLU
        let mut h1 = [0.0f32; 10];
        for j in 0..10 {
            let mut sum = LAYER1_BIASES[j];
            for k in 0..10 {
                sum += desc[k] * LAYER1_WEIGHTS[k][j];
            }
            h1[j] = sum.max(0.0);
        }

        // Layer 2: 10 -> 10 Dense + ReLU
        let mut h2 = [0.0f32; 10];
        for j in 0..10 {
            let mut sum = LAYER2_BIASES[j];
            for k in 0..10 {
                sum += h1[k] * LAYER2_WEIGHTS[k][j];
            }
            h2[j] = sum.max(0.0);
        }

        // Layer 3: 10 -> 2 Dense (Linear activation)
        let mut z = [0.0f32; 2];
        for j in 0..2 {
            let mut sum = LAYER3_BIASES[j];
            for k in 0..10 {
                sum += h2[k] * LAYER3_WEIGHTS[k][j];
            }
            z[j] = sum;
        }

        // Quantization: Nearest centroid lookup in 2D space
        let mut min_dist_sq = f32::INFINITY;
        let mut best_state = INVALID_STATE;

        for (centroid_idx, centroid) in CENTROIDS.iter().enumerate() {
            let dx = z[0] - centroid[0];
            let dy = z[1] - centroid[1];
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                best_state = centroid_idx;
            }
        }

        states.push(best_state);
    }

    states
}
