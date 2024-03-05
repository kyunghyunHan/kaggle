mod deep_titanic;
mod titanic;
mod digit_recognizer;
mod image_classifications;
mod playground_series_s3e24;
use rayon::prelude::*;
fn main() {
    playground_series_s3e24::model::main().unwrap();
}
