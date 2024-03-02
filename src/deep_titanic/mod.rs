pub mod model {
    use polars::prelude::*;
    use candle_core::{DType, Device, Result, Tensor, D};

    struct model {}
    impl model {}

    struct Dataset {
        pub train_images: Tensor, //train data
        pub train_labels: Tensor,
        pub test_images: Tensor, //test data
        pub test_labels: Tensor,
    }
    impl Dataset {
        fn new(){
            //데이터 불러오기
            let train_df = CsvReader::from_path("./datasets/titanic/train.csv")
                .unwrap()
                .finish()
                .unwrap();
            let test_df = CsvReader::from_path("./datasets/titanic/test.csv")
                .unwrap()
                .finish()
                .unwrap();
            let submisstion_df = CsvReader::from_path("./datasets/titanic/gender_submission.csv")
                .unwrap()
                .finish()
                .unwrap();
            
             //Tensor로 변경
             //train label
             //train tensor
             //test label
             //testtensor

             println!("{:?}",train_df.shape());
            
        }
    }
    pub fn main() {
       Dataset::new();
    }
}
