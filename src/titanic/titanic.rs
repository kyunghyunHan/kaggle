use polars::prelude::*;
use polars_lazy::dsl::col;
use std::fs::File;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::*;
use polars::prelude::{CsvWriter, DataFrame, NamedFrom, SerWriter, Series};
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier,RandomForestClassifierParameters};
use smartcore::{decomposition::pca::{PCA,PCAParameters}, linalg::basic::arrays::Array};

pub fn main(){
    /*===================data========================= */
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
   println!("null확인:{:?}",train_df.null_count());
  /*
   범주형 : Survived, Sex, Embarkd / PClass
   수치형 : SibSp, Parch / Age, Fare 
   */
  /*===================data========================= */

 
   
/*===================processing========================= */
/*null처리 */

let train_df= train_df.clone().lazy().with_columns([
    col("Age").fill_null((col("Age").mean())),
    col("Embarked").fill_null(lit("S")),

]).collect().unwrap().drop_many(&["Cabin"]);

let test_df: DataFrame=test_df.clone().lazy().with_columns([
    col("Age").fill_null((col("Age").mean())),
    col("Embarked").fill_null(lit("S")),

]).collect().unwrap().drop_many(&["Cabin"]);
//범주형 먼저 처
/*Pclass */
println!("중간점검:{:?}",train_df.tail(Some(3)));
println!("null확인:{:?}",train_df.null_count());

//dummies만들기
let pclass_train_dummies: DataFrame= train_df.select(["Pclass"]).unwrap().to_dummies(None, false).unwrap();
let pclass_test_dummies: DataFrame= test_df.select(["Pclass"]).unwrap().to_dummies(None, false).unwrap();
//join
let train_df = train_df.hstack(pclass_train_dummies.get_columns()).unwrap().drop(&"Pclass").unwrap();
let test_df = test_df.hstack(pclass_test_dummies.get_columns()).unwrap().drop(&"Pclass").unwrap();
/*sex */
let sex_train_dummies: DataFrame= train_df.select(["Sex"]).unwrap().to_dummies(None, false).unwrap();
let sex_test_dummies: DataFrame= test_df.select(["Sex"]).unwrap().to_dummies(None, false).unwrap();
let  train_df: DataFrame = train_df.hstack(sex_train_dummies.get_columns()).unwrap().drop(&"Sex").unwrap();
let mut test_df: DataFrame = test_df.hstack(sex_test_dummies.get_columns()).unwrap().drop(&"Sex").unwrap();


/*Age */

let  test_df: &mut DataFrame= test_df.with_column(  test_df.column("Age").unwrap().fill_null(FillNullStrategy::Mean).unwrap()).unwrap();
println!("{:?}", train_df);

/*SibSp & Panch */

/*Fare */
let  test_df: &mut DataFrame= test_df.with_column(  test_df.column("Fare").unwrap().fill_null(FillNullStrategy::Zero).unwrap()).unwrap();


/*Embarked */

/*채우기 */

let embarked_train_dummies: DataFrame= train_df.select(["Embarked"]).unwrap().to_dummies(None, false).unwrap();
let embarked_test_dummies: DataFrame= test_df.select(["Embarked"]).unwrap().to_dummies(None, false).unwrap();
let train_df: DataFrame = train_df.hstack(embarked_train_dummies.get_columns()).unwrap().drop(&"Embarked").unwrap();
let test_df: DataFrame = test_df.hstack(embarked_test_dummies.get_columns()).unwrap().drop(&"Embarked").unwrap();

// /*data 변환 */
let y_train= train_df.column("Survived").unwrap();
let x_train= train_df.drop("Survived").unwrap();
let x_test= test_df.drop("PassengerId").unwrap().clone();
// //y_train변환
let y_train: Vec<i64> = y_train.i64().unwrap().into_no_null_iter().collect();
let y_train:Vec<i32>= y_train.iter().map(|x|*x as i32).collect();

// //x_train변환
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
/*랜덤포레스트 */
let random_forest= RandomForestClassifier::fit(&x_train, &y_train,RandomForestClassifierParameters::default().with_n_trees(42)).unwrap();
let y_pred: Vec<i32> = random_forest.predict(&x_test).unwrap();
let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&result_train,&y_pred);
println!("{:?}",acc);
/*===================model test======================== */
/*===================제출용 파일========================= */
//random forest 가 가장 빠르기 때문에
let survived_series = Series::new("Survived", y_pred.into_iter().collect::<Vec<i32>>());
let passenger_id_series = test_df.column("PassengerId").unwrap().clone();

let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();


let mut output_file: File = File::create("./datasets/titanic/out.csv").unwrap();

CsvWriter::new(&mut output_file)
    .finish(&mut df)
    .unwrap();
/*===================제출용 파일========================= */
/*===================result========================= */
//0.8851674641148325
/*===================result========================= */

}
