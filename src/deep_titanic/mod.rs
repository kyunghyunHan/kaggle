pub mod model {
    use candle_core::{DType, Device, Result, Tensor, D};
    use polars::prelude::*;

    struct model {}
    impl model {}

    struct Dataset {
        pub train_images: Tensor, //train data
        pub train_labels: Tensor,
        pub test_images: Tensor, //test data
        pub test_labels: Tensor,
    }
    impl Dataset {
        fn new() {
            let train_samples = 42_000;
            let test_samples = 28_000;
            //데이터 불러오기
            let train_df = CsvReader::from_path("./datasets/titanic/train.csv")
                .unwrap()
                .finish()
                .unwrap();
            let test_df = CsvReader::from_path("./datasets/titanic/test.csv")
                .unwrap()
                .finish()
                .unwrap();
            let submission_df = CsvReader::from_path("./datasets/titanic/gender_submission.csv")
                .unwrap()
                .finish()
                .unwrap();
            let train_lebels = train_df
                .column("Survived")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|x| x as u8)
                .collect::<Vec<u8>>();
            println!("{}",train_df.null_count());
            println!("{}",test_df.null_count());
            println!("{}",submission_df.null_count());


            //Tensor로 변경
            //train label
            //train tensor
            //test label
            //testtensor

            println!("{:?}", train_df.shape());
        }
    }
    pub fn main() {
        Dataset::new();
    }
}
