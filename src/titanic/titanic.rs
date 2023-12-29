
use ndarray::prelude::*;
use polars::prelude::*;

use std::fs::File;
use smartcore::neighbors::knn_classifier::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::model_selection::train_test_split;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::*;
use polars::prelude::{CsvWriter, DataFrame, NamedFrom, SerWriter, Series};
use smartcore::svm::svc::{SVC,SVCParameters};
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier,RandomForestClassifierParameters};


pub fn main(){
   let train_df: DataFrame = CsvReader::from_path("./datasets/titanic/train.csv")
    .unwrap()
    .finish().unwrap().drop_many(&["PassengerId","Name", "Ticket"]);
   let test_df = CsvReader::from_path("./datasets/titanic/test.csv")
   .unwrap()
   .finish().unwrap().drop_many(&["Name", "Ticket"]);

   let result_df = CsvReader::from_path("./datasets/titanic/gender_submission.csv")
   .unwrap()
   .finish().unwrap();
   //데이터 미리보기
   println!("데이터 미리보기:{}",train_df.head(None));
   println!("데이터 정보 확인:{:?}",train_df.schema());
  
 
   
/*===================processing========================= */
/*Pclass */

//value_count 확인
println!("{:?}", train_df.column("Pclass").unwrap().value_counts(false,false).unwrap());
//dummies만들기
let pclass_train_dummies: DataFrame= train_df.select(["Pclass"]).unwrap().to_dummies(None, false).unwrap();
let pclass_test_dummies: DataFrame= test_df.select(["Pclass"]).unwrap().to_dummies(None, false).unwrap();

//더미 확인
println!("{:?}", pclass_train_dummies);
//join
let train_df = train_df.hstack(pclass_train_dummies.get_columns()).unwrap().drop(&"Pclass").unwrap();
let test_df = test_df.hstack(pclass_test_dummies.get_columns()).unwrap().drop(&"Pclass").unwrap();

println!("{:?}", train_df);
println!("{:?}", test_df);

/*sex */
let sex_train_dummies: DataFrame= train_df.select(["Sex"]).unwrap().to_dummies(None, false).unwrap();
let sex_test_dummies: DataFrame= test_df.select(["Sex"]).unwrap().to_dummies(None, false).unwrap();
println!("{:?}", sex_train_dummies);

let mut train_df: DataFrame = train_df.hstack(sex_train_dummies.get_columns()).unwrap().drop(&"Sex").unwrap();
let mut test_df: DataFrame = test_df.hstack(sex_test_dummies.get_columns()).unwrap().drop(&"Sex").unwrap();
println!("{:?}", train_df);

/*Age */

let  train_df: &mut DataFrame= train_df.with_column(  train_df.column("Age").unwrap().fill_null(FillNullStrategy::Mean).unwrap()).unwrap();
let  test_df: &mut DataFrame= test_df.with_column(  test_df.column("Age").unwrap().fill_null(FillNullStrategy::Mean).unwrap()).unwrap();
println!("{:?}", train_df);

/*SibSp & Panch */

/*Fare */
let  test_df: &mut DataFrame= test_df.with_column(  test_df.column("Fare").unwrap().fill_null(FillNullStrategy::Zero).unwrap()).unwrap();

/*Cabin */

let  mut train_df = train_df.drop("Cabin").unwrap();
let  test_df= test_df.drop("Cabin").unwrap();
println!("train_df:{:?}",train_df);
println!("test_df:{:?}",test_df);


/*Embarked */

println!("{}",train_df.column("Embarked").unwrap().value_counts(true,false).unwrap());
/*채우기 */
let mut  train_df: &mut DataFrame= train_df.with_column(  train_df.column("Embarked").unwrap().fill_null(FillNullStrategy::Backward(None)).unwrap()).unwrap();
println!("{}",train_df.column("Embarked").unwrap().value_counts(true,false).unwrap());


let embarked_train_dummies: DataFrame= train_df.select(["Embarked"]).unwrap().to_dummies(None, false).unwrap();
let embarked_test_dummies: DataFrame= test_df.select(["Embarked"]).unwrap().to_dummies(None, false).unwrap();



let mut train_df: DataFrame = train_df.hstack(embarked_train_dummies.get_columns()).unwrap().drop(&"Embarked").unwrap();
let mut test_df: DataFrame = test_df.hstack(embarked_test_dummies.get_columns()).unwrap().drop(&"Embarked").unwrap();

/*data 변환 */
let y_train= train_df.column("Survived").unwrap();
let x_train= train_df.drop("Survived").unwrap();
let x_test= test_df.drop("PassengerId").unwrap().clone();
//y_train변환
let y_train: Vec<i64> = y_train.i64().unwrap().into_no_null_iter().collect();
let y_train= y_train.iter().map(|x|*x as i32).collect();

//x_train변환
let x_data= x_train.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
let mut x_train: Vec<Vec<_>> = Vec::new();
for row in x_data.outer_iter() {
    let row_vec: Vec<_> = row.iter().cloned().collect();
    x_train.push(row_vec);
}
let x_train: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_train);
//x_변환
let test_data= x_test.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
let mut x_test: Vec<Vec<_>> = Vec::new();
for row in test_data.outer_iter() {
    let row_vec: Vec<_> = row.iter().cloned().collect();
    x_test.push(row_vec);
}
let x_test: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_test);
//result_data

let result_df: &Series= result_df.column(&"Survived").unwrap();

let result_train: Vec<i64> = result_df.i64().unwrap().into_no_null_iter().collect();
let result_train:Vec<i32>= result_train.iter().map(|x|*x as i32).collect();

/*===================processing========================= */
/*===================model test======================== */
/*knn */
let knn: KNNClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>, distance::euclidian::Euclidian<f64>>= KNNClassifier::fit(&x_train, &y_train, KNNClassifierParameters::default().with_k(3)).unwrap();
let y_pred: Vec<i32> = knn.predict(&x_test).unwrap();
let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&result_train,&y_pred);
println!("{:?}",acc);

// /*로지스틱 */
// let logreg= LogisticRegression::fit(&train_input, &tarin_target, Default::default()).unwrap();
// let y_pred: Vec<i32> = logreg.predict(&test_input).unwrap();
// let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&test_target, &y_pred);
// println!("{:?}",acc);
// /*랜덤포레스트 */
let random_forest= RandomForestClassifier::fit(&x_train, &y_train,RandomForestClassifierParameters::default().with_n_trees(100)).unwrap();
let y_pred: Vec<i32> = random_forest.predict(&x_test).unwrap();
let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&result_train,&y_pred);
println!("{:?}",acc);
/*===================model test======================== */
/*===================제출용 파일========================= */

//random forest 가 가장 빠르기 때문에
let random_forest= RandomForestClassifier::fit(&x_train, &y_train,RandomForestClassifierParameters::default().with_n_trees(100)).unwrap();
let y_pred: Vec<i32> = random_forest.predict(&x_test).unwrap();

let survived_series = Series::new("Survived", y_pred.into_iter().collect::<Vec<i32>>());
let passenger_id_series = test_df.column("PassengerId").unwrap().clone();

let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();


let mut output_file: File = File::create("./datasets/titanic/out.csv").unwrap();

CsvWriter::new(&mut output_file)
    .has_header(true)
    .finish(&mut df)
    .unwrap();
/*===================제출용 파일========================= */

}
