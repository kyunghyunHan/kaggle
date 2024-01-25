use polars::prelude::*;

pub fn main(){
    let train_df= CsvReader::from_path("./datasets/playground-series-s3e24/train.csv").unwrap().finish().unwrap();
    println!("null_data:{}",train_df.null_count());
    println!("{:?}",train_df.schema())
}