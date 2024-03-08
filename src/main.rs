mod deep_titanic;
mod titanic;
mod digit_recognizer;
mod image_classifications;
mod playground_series_s3e24;
use rayon::prelude::*;
fn main() {
    // deep_titanic::model::main().unwrap();
    digit_recognizer::model::main().unwrap();
}
