mod deep_titanic;
mod titanic;
use rayon::prelude::*;
fn main() {
    deep_titanic::model::main();
}
