use ndarray::prelude::*;
use polars::prelude::*;
use plotters::prelude::*;
use plotters::{prelude::*, style::full_palette::PURPLE};
use polars_lazy::dsl::col;
use std::fs::File;
use xgboost::{parameters, Booster, DMatrix};

pub fn main() {
    /*===================data 불러오기========================= */
    let train_df: DataFrame =
        CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/train.csv")
            .unwrap()
            .has_header(true)
            .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
            .finish()
            .unwrap();

    let test_df: DataFrame =
        CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/test.csv")
            .unwrap()
            .has_header(true)
            .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
            .finish()
            .unwrap();
    println!("{:?}", train_df.head(None));
    println!("{:?}", train_df.shape());
    println!("{:?}", train_df.schema());

    
    
}    
