//! mini3di-rs: Pure Rust implementation for encoding 3D protein coordinates into 3Di structural alphabet sequences.

pub mod encoder;
pub mod feature_encoder;
pub mod partner_index;
pub mod vae;
pub mod virtual_center;
pub mod weights;

pub use encoder::{build_sequence, encode_atoms};
pub use feature_encoder::calc_conformation_descriptors;
pub use partner_index::find_partner_indices;
pub use vae::{encode_descriptors, ALPHABET, INVALID_STATE};
pub use virtual_center::compute_virtual_centers;
