mod deep_titanic;
mod titanic;
mod digit_recognizer;
use rayon::prelude::*;
fn main() {
    deep_titanic::model::main().unwrap();
}
