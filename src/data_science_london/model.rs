use plotters::prelude::*;
use polars::prelude::*;
use smartcore::linalg::basic::arrays::Array;
use xgboost::DMatrix;
use xgboost::parameters;
use xgboost::Booster;
use std::fs::File;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::metrics::Metrics;
use smartcore::metrics::ClassificationMetricsOrd;
pub fn main() {
    let train_df: DataFrame =
        CsvReader::from_path("./datasets/data-science-london-scikit-learn/train.csv")
            .unwrap()
            .has_header(false) // 헤더가 없음을 명시
            .finish()
            .unwrap();
    let test_df: DataFrame =
        CsvReader::from_path("./datasets/data-science-london-scikit-learn/test.csv")
            .unwrap()
            .has_header(false) // 헤더가 없음을 명시

            .finish()
            .unwrap();

    let train_labels_df: DataFrame =
        CsvReader::from_path("./datasets/data-science-london-scikit-learn/trainLabels.csv")
            .unwrap()
            .has_header(false) // 헤더가 없음을 명시

            .finish()
            .unwrap();

    

   println!("{}",train_labels_df);
   /*======model====== */
    
    let x_train= train_df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();
    let x_test= test_df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();

    let mut x_train_vec: Vec<Vec<_>> = Vec::new();
    for row in x_train.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        x_train_vec.push(row_vec);
    }

    let mut x_test_vec: Vec<Vec<_>> = Vec::new();
    for row in x_test.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        x_test_vec.push(row_vec);
    }
    let y_train= train_labels_df.column("column_1").unwrap();
    let y_train:Vec<i32>= y_train.i64().unwrap().into_no_null_iter().map(|x|x as i32).collect();
    let x_train= DenseMatrix::from_2d_vec(&x_train_vec);
    let x_test= DenseMatrix::from_2d_vec(&x_test_vec);






/*===================model test======================== */
    /*랜덤포레스트 */
    let random_forest = RandomForestClassifier::fit(
        &x_train,
        &y_train,
        RandomForestClassifierParameters::default().with_n_trees(100),
    )
    .unwrap();
    let y_pred: Vec<i32> = random_forest.predict(&x_test).unwrap();
    // let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&y_test, &y_pred);
    // println!("{:?}", acc);

    /*===================제출용 파일========================= */
    //random forest 가 가장 빠르기 때문에
    let survived_series = Series::new("Solution", y_pred.clone().into_iter().collect::<Vec<i32>>());
    let passenger_id_series = Series::new("Id", (1i32..y_pred.len() as i32 +1).collect::<Vec<i32>>());

    let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
    let mut output_file: File = File::create("./datasets/data-science-london-scikit-learn/out.csv").unwrap();
    CsvWriter::new(&mut output_file).finish(&mut df).unwrap();
 
}
