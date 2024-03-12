pub mod model {

    use polars::prelude::*;
    use candle_core::{Tensor,D};
    use std::fs::File;
    #[derive(Debug)]
    struct Dataset{
        x_train:Tensor,
        y_train:Tensor,
        x_test:Tensor,
        y_test:Tensor,
    }
    impl Dataset{
        fn new(){
            let train_df=CsvReader::from_path("./datasets/playground-series-s4e1/train.csv").unwrap().finish().unwrap();
            let test_df=CsvReader::from_path("./datasets/playground-series-s4e1/test.csv").unwrap().finish().unwrap();
            let submussion_df=CsvReader::from_path("./datasets/playground-series-s4e1/sample_submission.csv").unwrap().finish().unwrap();
            
            //NUll check
            println!("{}",train_df.null_count());
            println!("{}",test_df.null_count());
            println!("{}",submussion_df.null_count());

        }
    }
    pub fn main()->anyhow::Result<()> {
        Dataset::new();
        Ok(())
    }
}
