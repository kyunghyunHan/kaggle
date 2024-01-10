use polars::prelude::*;
use std::fs::File;
use xgboost::{parameters, Booster, DMatrix};
pub fn main() {
    /*===============data============= */
    let train_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s3e19/train.csv")
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();
  


    /*==============결과============= */

}
