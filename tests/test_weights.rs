//! Integration test for `src/weights.rs` constants accessibility, typing, and values.

use mini3di_rs::weights::{
    CENTROIDS, LAYER1_BIASES, LAYER1_WEIGHTS, LAYER2_BIASES, LAYER2_WEIGHTS, LAYER3_BIASES,
    LAYER3_WEIGHTS,
};

#[test]
fn test_weights_constants_access_and_types() {
    // Verify types and dimensions of all 7 constants
    let _: &[[f32; 10]; 10] = &LAYER1_WEIGHTS;
    let _: &[f32; 10] = &LAYER1_BIASES;
    let _: &[[f32; 10]; 10] = &LAYER2_WEIGHTS;
    let _: &[f32; 10] = &LAYER2_BIASES;
    let _: &[[f32; 2]; 10] = &LAYER3_WEIGHTS;
    let _: &[f32; 2] = &LAYER3_BIASES;
    let _: &[[f32; 2]; 20] = &CENTROIDS;

    // Print lengths to ensure access without unused warnings
    assert_eq!(LAYER1_WEIGHTS.len(), 10);
    assert_eq!(LAYER1_BIASES.len(), 10);
    assert_eq!(LAYER2_WEIGHTS.len(), 10);
    assert_eq!(LAYER2_BIASES.len(), 10);
    assert_eq!(LAYER3_WEIGHTS.len(), 10);
    assert_eq!(LAYER3_BIASES.len(), 2);
    assert_eq!(CENTROIDS.len(), 20);
}

#[test]
fn test_centroids_values() {
    // CENTROIDS[0] must equal [-1.0729, -0.3600]
    let c0 = CENTROIDS[0];
    assert_eq!(c0[0], -1.0729f32, "CENTROIDS[0][0] mismatch");
    assert_eq!(c0[1], -0.3600f32, "CENTROIDS[0][1] mismatch");

    // CENTROIDS[19] must equal [1.0290, 0.8772]
    let c19 = CENTROIDS[19];
    assert_eq!(c19[0], 1.0290f32, "CENTROIDS[19][0] mismatch");
    assert_eq!(c19[1], 0.8772f32, "CENTROIDS[19][1] mismatch");
}
