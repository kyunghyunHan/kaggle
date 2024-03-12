mod deep_titanic;
mod titanic;
mod digit_recognizer;
mod image_classifications;
mod playground_series_s3e24;
mod binary_classification_with_a_bank_churn_dataset;
use rayon::prelude::*;

fn main() {
    binary_classification_with_a_bank_churn_dataset::model::main().unwrap();
    // deep_titanic::model::main().unwrap();
    // playground_series_s3e24::model::main().unwrap();
}
