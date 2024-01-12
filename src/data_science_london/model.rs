use plotters::prelude::*;
use polars::prelude::*;
use smartcore::linalg::basic::arrays::Array;
use xgboost::DMatrix;
use xgboost::parameters;
use xgboost::Booster;
use std::fs::File;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::basic::matrix::DenseMatrix;
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

    println!("{:?}",train_df.shape());
    println!("{:?}",test_df.shape());
    println!("{:?}",train_labels_df.shape());


   /*======model====== */
    
    let x_train= train_df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();
    let mut x_train_vec: Vec<Vec<_>> = Vec::new();
    for row in x_train.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        x_train_vec.push(row_vec);
    }
    let y_train= train_labels_df.column("column_1").unwrap();
    let y:Vec<i32>= y_train.i64().unwrap().into_no_null_iter().map(|x|x as i32).collect();
    let x= DenseMatrix::from_2d_vec(&x_train_vec);
    // let x_data = x_data.into_shape(1000 * 40).unwrap();
    // let x_data: Vec<f32> = x_data.into_iter().collect();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25, true, None);
    

    println!("{:?}",x_train.shape());
    println!("{:?}",x_test.shape());
    println!("{:?}",y_train.shape());
    println!("{:?}",y_test.shape());

 
}
