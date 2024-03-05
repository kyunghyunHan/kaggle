mod deep_titanic;
mod titanic;
mod digit_recognizer;
mod image_classifications;
use rayon::prelude::*;
fn main() {
    image_classifications::model::main().unwrap();
}
