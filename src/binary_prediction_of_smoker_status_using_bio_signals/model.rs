use polars::prelude::cov::pearson_corr;
use polars::prelude::*;
use ndarray::prelude::*;
use xgboost::DMatrix;
use xgboost::parameters;
use xgboost::Booster;
use std::fs::File;
pub fn main() {
    /*data */
    let train_df = CsvReader::from_path("./datasets/playground-series-s3e24/train.csv")
        .unwrap()
        .finish()
        .unwrap()
        .drop_many(&["id"]);

    println!("null_data:{}", train_df.null_count());
    println!("{:?}", train_df.schema());



    
    let test_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s3e24/test.csv")
    .unwrap()
    .finish()
    .unwrap().drop("id").unwrap();
     
    let submission_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s3e24/sample_submission.csv").unwrap().finish().unwrap();

    /*상관관계 확인 */
    let train_df = train_df
        .clone()
        .lazy()
        .with_columns([
            col("age").cast(DataType::Float64),
            col("height(cm)").cast(DataType::Float64),
            col("weight(kg)").cast(DataType::Float64),
            col("hearing(left)").cast(DataType::Float64),
            col("hearing(right)").cast(DataType::Float64),
            col("systolic").cast(DataType::Float64),
            col("relaxation").cast(DataType::Float64),
            col("fasting blood sugar").cast(DataType::Float64),
            col("Cholesterol").cast(DataType::Float64),
            col("triglyceride").cast(DataType::Float64),
            col("HDL").cast(DataType::Float64),
            col("LDL").cast(DataType::Float64),
            col("Urine protein").cast(DataType::Float64),
            col("AST").cast(DataType::Float64),
            col("ALT").cast(DataType::Float64),
            col("Gtp").cast(DataType::Float64),
            col("dental caries").cast(DataType::Float64),
            col("smoking").cast(DataType::Float64),
        ])
        .collect()
        .unwrap();


        let test_df: DataFrame = test_df
        .clone()
        .lazy()
        .with_columns([
            col("age").cast(DataType::Float64),
            col("height(cm)").cast(DataType::Float64),
            col("weight(kg)").cast(DataType::Float64),
            col("hearing(left)").cast(DataType::Float64),
            col("hearing(right)").cast(DataType::Float64),
            col("systolic").cast(DataType::Float64),
            col("relaxation").cast(DataType::Float64),
            col("fasting blood sugar").cast(DataType::Float64),
            col("Cholesterol").cast(DataType::Float64),
            col("triglyceride").cast(DataType::Float64),
            col("HDL").cast(DataType::Float64),
            col("LDL").cast(DataType::Float64),
            col("Urine protein").cast(DataType::Float64),
            col("AST").cast(DataType::Float64),
            col("ALT").cast(DataType::Float64),
            col("Gtp").cast(DataType::Float64),
            col("dental caries").cast(DataType::Float64),
            // col("smoking").cast(DataType::Float64),
        ])
        .collect()
        .unwrap();
    let age_corr: f64 = pearson_corr(
        train_df.column("age").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();
    let height_corr: f64 = pearson_corr(
        train_df.column("height(cm)").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();
    let weight_corr: f64 = pearson_corr(
        train_df.column("weight(kg)").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let waist_corr: f64 = pearson_corr(
        train_df.column("waist(cm)").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();


    let eyesight_left_corr: f64 = pearson_corr(
        train_df.column("eyesight(left)").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let eyesight_right_corr: f64 = pearson_corr(
        train_df.column("eyesight(right)").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();


    let hearing_left_corr: f64 = pearson_corr(
        train_df.column("hearing(left)").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let hearing_right_corr: f64 = pearson_corr(
        train_df.column("hearing(right)").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();


    let systolic_corr: f64 = pearson_corr(
        train_df.column("systolic").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();


    let relaxation_corr: f64 = pearson_corr(
        train_df.column("relaxation").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let fasting_blood_sugar_corr: f64 = pearson_corr(
        train_df.column("fasting blood sugar").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let cholesterol_corr: f64 = pearson_corr(
        train_df.column("Cholesterol").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let triglyceride_corr: f64 = pearson_corr(
        train_df.column("triglyceride").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let hdl_corr: f64 = pearson_corr(
        train_df.column("HDL").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let ldl_corr: f64 = pearson_corr(
        train_df.column("LDL").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();


    let hemoglobin_corr: f64 = pearson_corr(
        train_df.column("hemoglobin").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let urine_protein_corr: f64 = pearson_corr(
        train_df.column("Urine protein").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();

    let serum_creatinine_corr: f64 = pearson_corr(
        train_df.column("serum creatinine").unwrap().f64().unwrap(),
        train_df.column("smoking").unwrap().f64().unwrap(),
        1,
    )
    .unwrap();
let ast_corr: f64 = pearson_corr(
    train_df.column("AST").unwrap().f64().unwrap(),
    train_df.column("smoking").unwrap().f64().unwrap(),
    1,
)
.unwrap();

let alt_corr: f64 = pearson_corr(
    train_df.column("ALT").unwrap().f64().unwrap(),
    train_df.column("smoking").unwrap().f64().unwrap(),
    1,
)
.unwrap();

let gtp_corr: f64 = pearson_corr(
    train_df.column("Gtp").unwrap().f64().unwrap(),
    train_df.column("smoking").unwrap().f64().unwrap(),
    1,
)
.unwrap();

let dental_caries_corr: f64 = pearson_corr(
    train_df.column("dental caries").unwrap().f64().unwrap(),
    train_df.column("smoking").unwrap().f64().unwrap(),
    1,
)
.unwrap();

    println!("age_corr:{}", age_corr);
    println!("height_corr:{}", height_corr);
    println!("weight_corr:{}", weight_corr);
    println!("waist_corr:{}", waist_corr);
    println!("eyesight(left):{}", eyesight_left_corr);
    println!("eyesight(right):{}", eyesight_right_corr);
    println!("hearing(left):{}", hearing_left_corr);
    println!("hearing(right):{}", hearing_right_corr);
    println!("systolic_corr:{}", systolic_corr);
    println!("relaxation_corr:{}", relaxation_corr);
    println!("fasting_blood_sugar_corr:{}", fasting_blood_sugar_corr);
    println!("cholesterol_corr:{}", cholesterol_corr);
    println!("triglyceride_corr:{}", triglyceride_corr);
    println!("hdl_corr:{}", hdl_corr);
    println!("ldl_corr:{}", ldl_corr);
    println!("urine_protein_corr:{}", urine_protein_corr);
    println!("hemoglobin_corr:{}", hemoglobin_corr);
    println!("serum_creatinine_corr:{}", serum_creatinine_corr);
    println!("ast_corr:{}", ast_corr);
    println!("alt_corr:{}", alt_corr);
    println!("gtp_corr:{}", gtp_corr);
    println!("dental_caries_corr:{}", dental_caries_corr);

   /*상관계수가 0.2이상 것들만 파악  */
   let y_train:Vec<f32>= train_df.column("smoking").unwrap().f64().unwrap().into_no_null_iter().map(|x|x as f32).collect();
   let y_test:Vec<f32>= submission_df.column("smoking").unwrap().f64().unwrap().into_no_null_iter().map(|x|x as f32).collect();

   let  train_df=  train_df.select(["hemoglobin","weight(kg)","height(cm)","triglyceride","Gtp","serum creatinine"]).unwrap().to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();
   let  test_df=  test_df.select(["hemoglobin","weight(kg)","height(cm)","triglyceride","Gtp","serum creatinine"]).unwrap().to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();

   println!("{:?}", train_df.shape());
   println!("{:?}", test_df.shape());
   let x_train: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 1]>> = train_df.into_shape(159256 * 6).unwrap();
   let x_train: Vec<f32> = x_train.into_iter().collect();
 
   let x_test: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 1]>> = test_df.into_shape(106171 * 6).unwrap();
   let x_test: Vec<f32> = x_test.into_iter().collect();

   let mut dtrain = DMatrix::from_dense(&x_train, 159256).unwrap();
   dtrain.set_labels(&y_train).unwrap();
   let mut dtest = DMatrix::from_dense(&x_test, 106171).unwrap();
   dtest.set_labels(&y_test).unwrap();
   

   let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .evaluation_sets(Some(evaluation_sets))
        .build()
        .unwrap();

    let bst = Booster::train(&training_params).unwrap();
    let y = bst.predict(&dtest).unwrap();
    let y: Vec<f64> = y.iter().map(|x| *x as f64).collect();
    println!("{:?}", bst.predict(&dtest).unwrap());


    let survived_series = Series::new("smoking", &y);
    let passenger_id_series = submission_df.clone().column("id").unwrap().clone();

    let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
    let mut output_file: File =
        File::create("./datasets/playground-series-s3e24/out.csv").unwrap();
  
    CsvWriter::new(&mut output_file)
        // .has_header(true)
        .finish(&mut df)
        .unwrap();
}
