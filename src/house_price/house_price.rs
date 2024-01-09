use std::{ops::Add, result};

use linfa::correlation::{self, PearsonCorrelation};
use ndarray::prelude::*;
use polars::prelude::*;
// use polars_lazy::{prelude::*, dsl::{col, self}};
use plotters::{prelude::*, style::full_palette::PURPLE};
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use polars_lazy::dsl::col;
use xgboost::{parameters, DMatrix, Booster};
use ndarray_stats::CorrelationExt;
use plotters::prelude::*;
use std::fs::File;

pub fn main(){
    /*===================data 불러오기========================= */

    let train_df: DataFrame = CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/train.csv")
    .unwrap()
    .has_header(true)
    .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
    .finish()
    .unwrap();   

    let test_df: DataFrame = CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/test.csv")
    .unwrap()
    .has_header(true)
    .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
    .finish()
    .unwrap();  
    println!("{:?}",train_df.shape());
    println!("{:?}",train_df.head(None));
    println!("데이터 정보 확인:{:?}",train_df.schema());
    // println!("수치형 데이터 확인:{:?}",train_df.dtypes(None).unwrap());
    // println!("범주형 데이터 확인:{:?}",train_df.describe(Some(&[0f64])).unwrap());
    let results: DataFrame = CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/sample_submission.csv")
    .unwrap()
    .has_header(true)
    .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
    .finish()
    .unwrap();  
    
    /*===================sale_price log 정규화========================= */
    let sale_price_data:Vec<i64>= train_df.column("SalePrice").unwrap().i64().unwrap().into_no_null_iter().collect();
    let sale_price_data:Vec<f64>=sale_price_data.into_iter().map(|x|x as f64).collect();
    let sale_price_data: Vec<f64> = sale_price_data.iter().map(|&x| x.ln()).collect();
    println!("{:?}",sale_price_data);

    /*===================sale_price log 정규화========================= */
    /*===================data 불러오기========================= */


    /*TotalBsmtSF */
    //=>총 가용면적
    let total_bsmt_sf_data= train_df.select(&["TotalBsmtSF","SalePrice"]).unwrap();
    let total_bsmt_sf_data= total_bsmt_sf_data.to_ndarray::<Int64Type>(IndexOrder::Fortran).unwrap();
    let mut total_bsmt_sf_data_vec: Vec<Vec<_>> = Vec::new();
    for row in total_bsmt_sf_data.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        total_bsmt_sf_data_vec.push(row_vec);
    }
    let one_st_flr_sf_data= train_df.select(&["1stFlrSF","SalePrice"]).unwrap();
    let one_st_flr_sf_data= one_st_flr_sf_data.to_ndarray::<Int64Type>(IndexOrder::Fortran).unwrap();
    let mut one_st_flr_sf_data_vec: Vec<Vec<_>> = Vec::new();
    for row in one_st_flr_sf_data.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        one_st_flr_sf_data_vec.push(row_vec);
    }

    let two_st_flr_sf_data= train_df.select(&["2ndFlrSF","SalePrice"]).unwrap();
    let two_st_flr_sf_data= two_st_flr_sf_data.to_ndarray::<Int64Type>(IndexOrder::Fortran).unwrap();
    let mut two_st_flr_sf_data_vec: Vec<Vec<_>> = Vec::new();
    for row in two_st_flr_sf_data.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        two_st_flr_sf_data_vec.push(row_vec);
    }
    
    let train_df= train_df.clone().lazy().with_columns([
        col("TotalBsmtSF").add(col("1stFlrSF")).add(col("2ndFlrSF")).alias("TotalSF")
    
    ]).collect().unwrap();
     
    let total_data= train_df.select(&["TotalSF","SalePrice"]).unwrap();
    let total_data= total_data.to_ndarray::<Int64Type>(IndexOrder::Fortran).unwrap();
    let mut total_data_vec: Vec<Vec<_>> = Vec::new();
    for row in total_data.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        total_data_vec.push(row_vec);
    }
    
    let total_sf_root = BitMapBackend::new("./src/house_price/total_sf_avilble.png", (800, 600)).into_drawing_area();
    total_sf_root.fill(&WHITE).unwrap();
     let data= [
        ("total_bsmt_sf",total_bsmt_sf_data_vec.clone(),BLUE,0.0..6000.0,(0.0..800000.0)),
        ("1stFlrSF",one_st_flr_sf_data_vec,PURPLE,(0.0..4000.0),(0.0..700000.0)),
        ("two_st_flr_sf_data_vec",two_st_flr_sf_data_vec,GREEN,(0.0..2000.0),(0.0..700000.0)),
        ("total_data_vec",total_data_vec.clone(),RED,(0.0..12000.0),(0.0..1000000.0)),

        ];

    let total_sf_drawing_areas = total_sf_root.split_evenly((2, 2));
    for (total_sf_drawing_area, idx) in total_sf_drawing_areas.iter().zip(1..) {
 
        let mut chart: ChartContext<'_, BitMapBackend<'_>, Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>> = ChartBuilder::on(&total_sf_drawing_area)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(data[idx as usize-1].3.clone(), data[idx as usize-1].4.clone())
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Length")
        .y_desc("Weight")
        .draw()
        .unwrap();

    chart
        .draw_series(
            data[idx as usize-1].1
                .iter()
                .map(|x| {
                    Circle::new((x[0]as f64, x[1]as f64), 5, Into::<ShapeStyle>::into(data[idx as usize -1].2))
                }),
        )
        .unwrap();

}
    /*Year_built_and_Remodeled*/
    let year_built= train_df.select(&["YearBuilt","SalePrice"]).unwrap().to_ndarray::<Int64Type>(IndexOrder::Fortran).unwrap();
    let mut year_built_vec: Vec<Vec<_>> = Vec::new();
    for row in year_built.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        year_built_vec.push(row_vec);
    }
    let year_remode_add= train_df.select(&["YearRemodAdd","SalePrice"]).unwrap().to_ndarray::<Int64Type>(IndexOrder::Fortran).unwrap();
    let mut year_remode_add_vec: Vec<Vec<_>> = Vec::new();
    for row in year_remode_add.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        year_remode_add_vec.push(row_vec);
    }

    let year_remod_root = BitMapBackend::new("./src/house_price/year_remod.png", (800, 600)).into_drawing_area();
    year_remod_root.fill(&WHITE).unwrap();
    let year_remode_data= [
        ("YearBuilt",year_built_vec.clone(),BLUE,1880.0..2000.0,(0.0..800000.0)),
        ("YearRemodAdd",year_built_vec,PURPLE,(1950.0..2010.0),(0.0..700000.0)),
        ];
        let year_remode_drawing_areas = year_remod_root.split_evenly((1, 2));
        for (year_remode_drewing_area, idx) in year_remode_drawing_areas.iter().zip(1..) {
     
            let mut year_remode_drewing_area_chart: ChartContext<'_, BitMapBackend<'_>, Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>> = ChartBuilder::on(&year_remode_drewing_area)
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(year_remode_data[idx as usize-1].3.clone(), year_remode_data[idx as usize-1].4.clone())
            .unwrap();
    
            year_remode_drewing_area_chart
            .configure_mesh()
            .x_desc("Length")
            .y_desc("Weight")
            .draw()
            .unwrap();
    
            year_remode_drewing_area_chart
            .draw_series(
                year_remode_data[idx as usize-1].1
                    .iter()
                    .map(|x| {
                        Circle::new((x[0]as f64, x[1]as f64), 5, Into::<ShapeStyle>::into(year_remode_data[idx as usize -1].2))
                    }),
            )
            .unwrap();
    
    }
    //YrBltAndRemoddk으로저장
    let train_df= train_df.clone().lazy().with_columns([
        col("YearBuilt").add(col("YearRemodAdd")).alias("YrBltAndRemod"),

    
    ]).collect().unwrap();
     

     
         // let x_data= train_data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
    let test_df: DataFrame= test_df.clone().lazy().with_columns([
            col("YearBuilt").add(col("YearRemodAdd")).alias("YrBltAndRemod"),
            col("TotalBsmtSF").add(col("1stFlrSF")).add(col("2ndFlrSF")).alias("TotalSF")

        
        ]).collect().unwrap();
  
             // let x_data= train_data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
    
    /*============ data============ */
    //train_data
    let result_df= train_df.select(&["YrBltAndRemod","TotalSF"]).unwrap();
    let result_df= result_df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();
   //train_target
   let target_data=train_df.column("SalePrice").unwrap();
   let target_data:Vec<f64>= target_data.i64().unwrap().into_no_null_iter().into_iter().map(|x|x.as_f64()).collect();
   let target_data= arr1(&target_data);
   //test_data
   let test_df = test_df.fill_null(FillNullStrategy::Backward(None)).unwrap();
   let test_result_df: DataFrame= test_df.select(&["YrBltAndRemod","TotalSF"]).unwrap();
   let test_result_df= test_result_df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();

   //test_test_target_data

   let results= results.clone().lazy().with_columns([
    col("SalePrice").fill_nan(lit(176041.882371574)).alias("SalePrice")
   ]).collect().unwrap();
   let test_target_data= results.column("SalePrice").unwrap();
   let test_target_data:Vec<f64>= test_target_data.f64().unwrap().into_no_null_iter().into_iter().map(|x|x.as_f64()).collect();
   let test_target_data: ArrayBase<ndarray::OwnedRepr<_>, Dim<[usize; 1]>>= arr1(&test_target_data);
    /*============ data============ */

    /*============ LinearRegression============ */
   
    // let a: DatasetBase<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>>: DatasetBase<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>>= Dataset::new(result_df, target_data);
     

   
    // let b= Dataset::new(test_result_df, test_target_data);

    // let model = LinearRegression::default().fit(&a).unwrap();
    // let  pred = model.predict(&b);

    // let mut pred_vec: Vec<f64> = Vec::new();
    // for row in pred.iter() {
    //     let rounded_val = format!("{:.9}", row).parse::<f64>().unwrap();

    //     pred_vec.push(rounded_val);
    // }

    // let r2 = pred.r2(&a).unwrap();
    
    /*===================LinearRegression========================= */
/*===================xgboost========================= */
println!("{:?}",result_df.shape());
let b = result_df.into_shape(1460* 2).unwrap();
let x_train:Vec<f32>= b.into_iter().collect();

let num_rows = 1460;
let y_train:Vec<f32>= target_data.iter().map(|x|*x as f32).collect();

// // convert training data into XGBoost's matrix format
let mut dtrain = DMatrix::from_dense(&x_train,num_rows).unwrap();

// // set ground truth labels for the training matrix
dtrain.set_labels(&y_train).unwrap();

// // test matrix with 1 row
println!("{:?}",test_result_df.shape());

let x_test = test_result_df.into_shape(1459* 2).unwrap();
let x_test:Vec<f32>= x_test.into_iter().collect();

let num_rows = 1459;
let result_train:Vec<f32>= test_target_data.iter().map(|x|*x as f32).collect();


let mut dtest = DMatrix::from_dense(&x_test, num_rows).unwrap();
dtest.set_labels(&result_train).unwrap();

// specify datasets to evaluate against during training
let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

// // specify overall training setup
let training_params = parameters::TrainingParametersBuilder::default()
    .dtrain(&dtrain)
    .evaluation_sets(Some(evaluation_sets))
    .build()
    .unwrap();

// // // train model, and print evaluation data
let bst = Booster::train(&training_params).unwrap();
let y= bst.predict(&dtest).unwrap();
let y:Vec<f64>= y.iter().map(|x|*x as f64).collect();
println!("{:?}", bst.predict(&dtest).unwrap());
/*===================xgboost========================= */
let survived_series = Series::new("SalePrice",&y);
let passenger_id_series =results.clone().column("Id").unwrap().clone();

let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
 println!("{}",df.null_count());

let mut output_file: File = File::create("./datasets/house-prices-advanced-regression-techniques/out.csv").unwrap();
let mut df: DataFrame= df.clone().lazy().with_columns([
    col("SalePrice").fill_nan(lit(285932012))


]).collect().unwrap();
CsvWriter::new(&mut output_file)
    // .has_header(true)
    .finish(&mut df)
    .unwrap();

}