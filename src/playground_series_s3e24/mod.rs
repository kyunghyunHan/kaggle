pub mod model{

    use candle_core::{Tensor,Result as CRS};
    use polars::prelude::*;
    struct  Dataset {
        x_train:Tensor,
        y_train:Tensor,
        x_test:Tensor,
        y_test:Tensor,
    }
    impl Dataset{
        fn new()->CRS<()>{
            
        Ok(())
        }

    }
    pub fn main(){
        let  train_df=  CsvReader::from_path("./datasets/playground-series-s3e24/train.csv").unwrap().finish().unwrap();
        
    }
}