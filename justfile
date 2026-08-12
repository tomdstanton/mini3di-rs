set default-target := "build"

# Run tests using cargo nextest
test:
    cargo nextest run

# Run clippy and format checks
lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

# Build the project in release mode
build:
    cargo build --release

# Publish to crates.io (dry run)
publish-dry-run:
    cargo publish --dry-run

# Publish to crates.io
publish:
    cargo publish
