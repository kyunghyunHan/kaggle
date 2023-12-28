
use ndarray::{Axis, iter::Axes,concatenate,stack};
use polars::lazy::dsl::col;
use polars::prelude::*;
use polars_core::prelude::*;
use polars_io::prelude::*;
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
   let mut train_df: DataFrame = CsvReader::from_path("./datasets/titanic/train.csv")
    .unwrap()
    .finish().unwrap().drop_many(&["PassengerId","Name", "Ticket"]);
   let test_df = CsvReader::from_path("./datasets/titanic/test.csv")
   .unwrap()
   .finish().unwrap().drop_many(&["Name", "Ticket"]);
   //데이터 미리보기
   println!("데이터 미리보기:{}",train_df.head(None));
   println!("데이터 정보 확인:{:?}",train_df.schema());
  
 
 

/*Pclass */
//value_count 확인
println!("{:?}", train_df.column("Pclass").unwrap().value_counts(false,false).unwrap());
let mut pclass_test_dummies: DataFrame= train_df.select(["Pclass"]).unwrap().to_dummies(None, false).unwrap();
println!("{:?}", pclass_test_dummies);
// let train: DataFrame= pclass_test_dummies.join(&pclass_test_dummies, ["Pclass_1"], ["Pclass_1"], JoinArgs::new(JoinType::Outer)).unwrap();
// // println!("{:?}", train);
// println!("{:?}", pclass_test_dummies);
let train_df = train_df.hstack(pclass_test_dummies.get_columns()).unwrap();
println!("{:?}", train_df);

// let  pclass_train_dummies= pclass_train_dummies.rename("Pclass_1","Pclass_1").unwrap();
// let  pclass_test_dummies= pclass_test_dummies.rename("Pclass_1","Pclass_1").unwrap();
// println!("pclass_test_dummies:{:?}",pclass_train_dummies.shape());
// let train_df = train_df.inner_join(&pclass_train_dummies, ["Pclass_1"], ["Age"]).unwrap();




// let train_df = train_df.join(&pclass_test_dummies, None, None,  JoinArgs::new(JoinType::Inner)).unwrap();
// println!("train_dummies:{:?}",pclass_train_dummies);
// println!("test_dummies{:?}",pclass_test_dummies);

// let  train_df: DataFrame = train_df.drop("Pclass").unwrap();
// let  test_df= test_df.drop("Pclass").unwrap();
// println!("train:{:?}",train_df);
// println!("test:{:?}",test_df);

// for series in pclass_train_dummies.iter(){
//     train_df = train_df.hstack(&[series.clone()]).unwrap();
// }

// for series in pclass_test_dummies.iter(){
//     test_df = test_df.hstack(&[series.clone()]).unwrap();
// }
// println!("train:{:?}",train_df);
// println!("test:{:?}",test_df);

// /*Sex */

// let  mut sex_train_dummies= train_df.column("Sex").unwrap().to_dummies(None, false).unwrap();
// let  mut sex_test_dummies= test_df.column("Sex").unwrap().to_dummies(None, false).unwrap();


// let  sex_train_dummies= sex_train_dummies.rename("Sex_female","Female").unwrap().rename("Sex_male","male").unwrap();
// let  sex_test_dummies= sex_test_dummies.rename("Sex_female","Female").unwrap().rename("Sex_male","male").unwrap();
// println!("train_dummies:{:?}",sex_train_dummies);
// println!("test_dummies{:?}",sex_test_dummies);

// let  mut train_df = train_df.drop("Sex").unwrap();
// let  mut test_df = test_df.drop("Sex").unwrap();
// println!("train:{:?}",train_df);
// println!("test:{:?}",test_df);
// //join
// for series in sex_train_dummies.iter(){
//     train_df = train_df.hstack(&[series.clone()]).unwrap();
// }
// for series in sex_test_dummies.iter(){
//     test_df = test_df.hstack(&[series.clone()]).unwrap();
// }
// println!("train:{:?}",train_df);
// println!("test:{:?}",test_df);
// /*Age */

// println!("train_df:{:?}",train_df);

// let  train_df: &mut DataFrame= train_df.with_column(  train_df.column("Age").unwrap().fill_null(FillNullStrategy::Mean).unwrap()).unwrap();
// let  test_df: &mut DataFrame= test_df.with_column(  test_df.column("Age").unwrap().fill_null(FillNullStrategy::Mean).unwrap()).unwrap();

// println!("train_df:{:?}",train_df);
// println!("test_df:{:?}",test_df);


// /*SibSp & Panch 
// 바꿀필요 x
// */


// /*Fare 
// 무단탑승으로 생각하고 0으로 입력
// */
// println!("Fare:{}",test_df.column("Fare").unwrap().value_counts(true,false).unwrap());

// let mut test_df: &mut DataFrame= test_df.with_column(  test_df.column("Fare").unwrap().fill_null(FillNullStrategy::Zero).unwrap()).unwrap();
// println!("Fare:{}",test_df.column("Fare").unwrap().value_counts(true,false).unwrap());

// println!("train_df:{:?}",train_df);
// println!("test_df:{:?}",test_df);
// /*Cabin */
// let  mut train_df = train_df.drop("Cabin").unwrap();
// let mut test_df= test_df.drop("Cabin").unwrap();
// println!("train_df:{:?}",train_df);
// println!("test_df:{:?}",test_df);
/*Embarked
탑승구를 의미
*/

// println!("{}",train_df.column("Embarked").unwrap().value_counts(true,false).unwrap());
// let mut  train_df: &mut DataFrame= train_df.with_column(  train_df.column("Embarked").unwrap().fill_null(FillNullStrategy::Backward(None)).unwrap()).unwrap();
// println!("{}",train_df.column("Embarked").unwrap().value_counts(true,false).unwrap());
// let  mut embarked_train_dummies= train_df.column("Embarked").unwrap().to_dummies(None, false).unwrap();
// let  mut embarked_test_dummies= test_df.column("Embarked").unwrap().to_dummies(None, false).unwrap();


// let  embarked_train_dummies= embarked_train_dummies.rename("Embarked_C","C").unwrap().rename("Embarked_Q","Q").unwrap().rename("Embarked_S","S").unwrap();
// let  embarked_test_dummies= embarked_test_dummies.rename("Embarked_C","C").unwrap().rename("Embarked_Q","Q").unwrap().rename("Embarked_S","S").unwrap();
// let  mut train_df = train_df.drop("Embarked").unwrap();
// let  mut test_df = test_df.drop("Embarked").unwrap();

// for series in embarked_train_dummies.iter(){
//     train_df = train_df.hstack(&[series.clone()]).unwrap();
// }
// for series in embarked_test_dummies.iter(){
//     test_df = test_df.hstack(&[series.clone()]).unwrap();
// }
// let x_train= train_df.drop("Survived").unwrap();//훈련데이터
// let  y_train= train_df.column("Survived").unwrap();//타겟데이터
// let x_test= test_df.drop("PassengerId").unwrap();


// let  x_train = x_train.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

// let  x_test = x_test.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

// let mut data: Vec<Vec<_>> = Vec::new();
// for row in x_train.outer_iter() {
//     let row_vec: Vec<_> = row.iter().cloned().collect();
//     data.push(row_vec);
// }


// let mut test_data: Vec<Vec<_>> = Vec::new();
// for row in x_test.outer_iter() {
//     let row_vec: Vec<_> = row.iter().cloned().collect();
//     test_data.push(row_vec);
// }
// let x_train: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&data);
// println!("{}",x_train);
// let x_test: DenseMatrix<f64>= DenseMatrix::from_2d_vec(&test_data);
// let y_train: Vec<i64> = y_train.i64().unwrap().into_no_null_iter().collect();
// let y_train= y_train.iter().map(|x|*x as i32).collect();
// let (train_input, test_input, tarin_target, test_target) = train_test_split(&x_train, &y_train, 0.2,true,None);
    
/*데이터 나누기 */

/*알고리즘 적용 
target= vec<i32>
input= denseMartrix
*/
// let knn: KNNClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>, distance::euclidian::Euclidian<f64>>= KNNClassifier::fit(&train_input, &tarin_target, KNNClassifierParameters::default().with_k(4)).unwrap();

// let y_pred: Vec<i32> = knn.predict(&test_input).unwrap();

// let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&test_target, &y_pred);
// println!("{:?}",acc);



// /*로지스틱 */
// let logreg= LogisticRegression::fit(&train_input, &tarin_target, Default::default()).unwrap();
// let y_pred: Vec<i32> = logreg.predict(&test_input).unwrap();
// let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&test_target, &y_pred);
// println!("{:?}",acc);

// // /*SVC */
// // let y_pred =  SVC::fit(&x_train, &y_train, &SVCParameters::default().with_c(10.0))
// //     .and_then(|svm| svm.predict(&x_test))
// //     .unwrap();
// // println!("AUC SVM: {}", roc_auc_score(&test_target, &y_pred));

// /*랜덤포레스트 */
// let random_forest= RandomForestClassifier::fit(&train_input, &tarin_target, RandomForestClassifierParameters::default().with_n_trees(100)).unwrap();
// let y_pred: Vec<i32> = random_forest.predict(&test_input).unwrap();
// let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&test_target, &y_pred);
// println!("{:?}",acc);

/*제출용 파일 */

// let random_forest= RandomForestClassifier::fit(&train_input, &tarin_target, RandomForestClassifierParameters::default().with_n_trees(100)).unwrap();
// let random_forest_y_pred: Vec<i32> = random_forest.predict(&test_input).unwrap();

// let survived_series = Series::new("Survived", random_forest_y_pred.into_iter().collect::<Vec<i32>>());
// let passenger_id_series = test_df.column("PassengerId").unwrap().clone();

// let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();


// let mut output_file: File = File::create("./datasets/titanic/out.csv").unwrap();

// CsvWriter::new(&mut output_file)
//     .has_header(true)
//     .finish(&mut df)
//     .unwrap();

}
