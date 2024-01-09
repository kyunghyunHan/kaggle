use polars::prelude::*;

pub fn main(){
     /*===========data========= */
     let train_df: DataFrame = CsvReader::from_path("./datasets/used-car-price-prediction/used_cars_train.csv")
     .unwrap()
     .finish()
     .unwrap();
    println!("{:?}",train_df.shape());

        
}