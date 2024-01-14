use polars::prelude::*;
use plotters::prelude::*;



pub fn main(){
     let train_df: DataFrame = CsvReader::from_path("./datasets/icr-identify-age-related-conditions/train.csv")
     .unwrap()
     .finish()
     .unwrap();
    println!("{}",train_df);

}