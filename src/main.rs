mod projects;
mod utils;
use crate::projects::binary_classification_with_a_bank_churn_dataset as model;
fn main() {
    model::main().unwrap();
}
