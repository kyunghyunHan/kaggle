use std::fs::File;
use ndarray::prelude::*;
use polars::{prelude::*, lazy::dsl::{col, when}, error::constants::TRUE};
use plotters::prelude::*;
use smartcore::{decomposition::pca::{PCA,PCAParameters}, linalg::basic::arrays::Array, ensemble::random_forest_classifier::RandomForestClassifier};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::*;

use smartcore::ensemble::{random_forest_classifier::RandomForestClassifierSearchParameters};
use smartcore::tree::decision_tree_classifier::SplitCriterion;
pub fn main(){
    /*===================data 불러오기========================= */

    let  mut train_df: DataFrame = CsvReader::from_path("./datasets/spaceship-titanic/train.csv").unwrap()
    .finish().unwrap();

    let  test_df: DataFrame = CsvReader::from_path("./datasets/spaceship-titanic/test.csv").unwrap()
    .finish().unwrap();
let result_df = CsvReader::from_path("./datasets/spaceship-titanic/sample_submission.csv")
.unwrap()
.finish().unwrap();

    println!("데이터 미리보기:{}",train_df.head(None));
    println!("데이터 정보 확인:{:?}",train_df.schema());
    println!("데이터 미리보기:{}",test_df.head(None));
    println!("데이터 정보 확인:{:?}",test_df.schema());

    println!("결측치 확인:{:?}",train_df.null_count());
    // println!("수치형 데이터 확인:{:?}",train_df.describe(None).unwrap());

    // println!("범주형 데이터 확인:{:?}",train_df.describe(Some(&[0f64])).unwrap());
    let age_data:Vec<f64>= train_df.column("Age").unwrap().f64().unwrap().into_no_null_iter().collect();
    let room_service_data:Vec<f64>= train_df.column("RoomService").unwrap().f64().unwrap().into_no_null_iter().collect();
    let food_court_data:Vec<f64>= train_df.column("FoodCourt").unwrap().f64().unwrap().into_no_null_iter().collect();
    let shopping_mall_data:Vec<f64>= train_df.column("ShoppingMall").unwrap().f64().unwrap().into_no_null_iter().collect();
    let spa_data:Vec<f64>= train_df.column("Spa").unwrap().f64().unwrap().into_no_null_iter().collect();
    let vr_deck_data:Vec<f64>= train_df.column("VRDeck").unwrap().f64().unwrap().into_no_null_iter().collect();

    /*===================data 불러오기========================= */
    /*===================히스토그램 그리기========================= */
    let root = BitMapBackend::new("./src/spaceship_titanic/histogram.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let data= [ 
     ("Age", &age_data),
    ("RoomService", &room_service_data),
    ("FoodCourt", &food_court_data),
    ("ShoppingMall", &shopping_mall_data),
    ("Spa", &spa_data),
    ("VRDeck", &vr_deck_data)];
    let drawing_areas = root.split_evenly((3, 2));
    for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
        let mut chart_builder = ChartBuilder::on(&drawing_area);
    chart_builder.margin(5).set_left_and_bottom_label_area_size(20).caption(format!("{}",data[idx as usize -1].0), ("sans-serif", 40));
    let mut chart_context = chart_builder.build_cartesian_2d((0u32..80u32).step(1).into_segmented(), (0f64..2100f64).step(100f64)).unwrap();
    chart_context.configure_mesh().draw().unwrap();
    chart_context.draw_series(Histogram::vertical(&chart_context).style(BLUE.filled()).margin(10)
    .data(data[idx -1].1.iter().map(|v | (*v as u32,1f64 )))
   ).unwrap();
    }


    /*===================히스토그램 그리기========================= */

    /*===================processing========================= */

let train_df = train_df
    .clone()
    .lazy()
    .with_columns([
        col("CryoSleep").cast(DataType::Float64).fill_null(0),
        col("VIP").cast(DataType::Float64).fill_null(0),
        col("Cabin").fill_null(lit("G/734/S")),
        col("HomePlanet").fill_null(lit("Earth")),
        col("Destination").fill_null(lit("TRAPPIST-1e")),
        col("ShoppingMall").fill_null(col("ShoppingMall").median()),
        col("VRDeck").fill_null(col("VRDeck").median()),
        col("FoodCourt").fill_null(col("FoodCourt").median()),
        col("Spa").fill_null(col("Spa").median()),
        col("RoomService").fill_null(col("RoomService").median()),
        col("Age").fill_null(col("Age").median()),
    ])
    .collect().unwrap();
let  train_df= train_df.clone().lazy().with_columns(
[col("Cabin").str().split(lit("/")).list().get(lit(0)).alias("Cabin_1"),
       col("Cabin").str().split(lit("/")).list().get(lit(1)).alias("Cabin_2").cast(DataType::Float64),
        col("Cabin").str().split(lit("/")).list().get(lit(2)).alias("Cabin_3"),
        col("VIP").cast(DataType::Int64),
        col("CryoSleep").cast(DataType::Int64)

        ]
).collect().unwrap();
  let  train_df= train_df.drop_many(&["PassengerId", "Name", "Cabin"]);
  println!("{}",train_df.null_count());


  /*VIP */
    let vip_result = train_df
   .group_by(&["VIP"]).unwrap().select(["Transported"]).groups().unwrap();
   let  vip_result= vip_result.clone().lazy().with_columns(
    [
            col("groups").list().len()
            ]
    ).collect().unwrap();
    let vip_transported:Vec<u32>= vip_result.column("groups").unwrap().u32().unwrap().into_no_null_iter().collect();
    println!("{:?}",vip_transported);

    let root = BitMapBackend::new("./src/spaceship_titanic/vip.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let data= [ 
     ("a", &vip_transported[0]),
    ("b", &vip_transported[1]),
  ];
  let mut chart_builder = ChartBuilder::on(&root);
  chart_builder.margin(5).set_left_and_bottom_label_area_size(20);
  let mut chart_context = chart_builder.build_cartesian_2d(["a","b"].into_segmented(), (0u32..8500u32).step(100u32)).unwrap();
  chart_context.configure_mesh().draw().unwrap();
  chart_context.draw_series(Histogram::vertical(&chart_context).style(BLUE.filled()).margin(10)
  .data(data.iter().map(|v | (&v.0 ,*v.1 )))
 ).unwrap();
   /*CryoSleep */

   let cryosleep_result = train_df
   .group_by(&["CryoSleep"]).unwrap().select(["Transported"]).mean().unwrap();
let sorted_means = cryosleep_result.sort(&["Transported_mean"], vec![false, true], false).unwrap();
let cryosleep_arrived = train_df.clone().lazy().filter(col("Transported").eq(true)).collect().unwrap().column("CryoSleep").unwrap().value_counts(true,false).unwrap();
let cryosleep_not_arrived = train_df.clone().lazy().filter(col("Transported").eq(false)).collect().unwrap().column("CryoSleep").unwrap().value_counts(true,false).unwrap();
let cryosleep_arrived_vec:Vec<u32>= cryosleep_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
let cryosleep_not_arrived_vec:Vec<u32>= cryosleep_not_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();

let data1: [(&str, u32); 2]= [
    ("arrived-1",cryosleep_arrived_vec[1]),
    ("arrived-2",cryosleep_arrived_vec[0])
];
let data2: [(&str, u32); 2]= [
    ("not-arrived-1",cryosleep_not_arrived_vec[0]),
    ("not-arrived-2",cryosleep_not_arrived_vec[1])
];
let root = BitMapBackend::new("./src/spaceship_titanic/CryoSleep.png", (800, 600)).into_drawing_area();
root.fill(&WHITE).unwrap();
  let mut chart_builder = ChartBuilder::on(&root);
  chart_builder.margin(15).set_left_and_bottom_label_area_size(20);
  let mut chart_context = chart_builder.build_cartesian_2d(["arrived-1","arrived-2","","not-arrived-1","not-arrived-2"].into_segmented(), (0u32..4000u32).step(500u32)).unwrap();
  chart_context.configure_mesh().draw().unwrap();

  chart_context.draw_series(Histogram::vertical(&chart_context).style(BLUE.filled()).margin(0)
  .data(data1.iter().map(|v | (&v.0 ,v.1)))
 ).unwrap();

 chart_context.draw_series(Histogram::vertical(&chart_context).style(RED.filled()).margin(0)
  .data(data2.iter().map(|v | (&v.0 ,v.1)))
 ).unwrap();


/*HomePlanet */
let homeplanet_result = train_df
.group_by(&["HomePlanet"]).unwrap().select(["Transported"]).mean().unwrap().sort(&["Transported_mean"], vec![false, true], false).unwrap();
println!("{}",homeplanet_result);

let homeplanet_arrived = train_df.clone().lazy().filter(col("Transported").eq(true)).collect().unwrap().column("HomePlanet").unwrap().value_counts(true,false).unwrap();
let homeplanet_not_arrived = train_df.clone().lazy().filter(col("Transported").eq(false)).collect().unwrap().column("HomePlanet").unwrap().value_counts(true,false).unwrap();
let homeplanet_arrived_vec:Vec<u32>= homeplanet_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
let homeplanet_not_arrived_vec:Vec<u32>= homeplanet_not_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();


/* Destination*/
let destination_result = train_df
.group_by(&["Destination"]).unwrap().select(["Transported"]).mean().unwrap().sort(&["Transported_mean"], vec![false, true], false).unwrap();
println!("{}",destination_result);

let destination_arrived = train_df.clone().lazy().filter(col("Transported").eq(true)).collect().unwrap().column("Destination").unwrap().value_counts(true,false).unwrap();
let destination_not_arrived = train_df.clone().lazy().filter(col("Transported").eq(false)).collect().unwrap().column("Destination").unwrap().value_counts(true,false).unwrap();
let destination_arrived_vec:Vec<u32>= destination_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
let destination_not_arrived_vec:Vec<u32>= destination_not_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
println!("{}",destination_result);



/*Age */

let train_df = train_df
    .clone()
    .lazy()
    .with_columns([
        when(col("Age").lt(lit(10.0))).then(lit(0f64)).
        when(col("Age").lt(lit(20.0))).then(lit(10f64)).
        when(col("Age").lt(lit(30.0))).then(lit(20f64)).
        when(col("Age").lt(lit(40.0))).then(lit(30f64)).
        when(col("Age").lt(lit(50.0))).then(lit(40f64)).
        when(col("Age").lt(lit(60.0))).then(lit(50f64)).
        when(col("Age").lt(lit(70.0))).then(lit(60f64)).
        when(col("Age").lt(lit(80.0))).then(lit(70f64)).
        otherwise(19.9).alias("Age")
    ])
    .collect().unwrap();   
println!("{}",train_df); 
println!("{}",train_df.column("Age").unwrap());

let age_arrived = train_df.clone().lazy().filter(col("Transported").eq(true)).collect().unwrap().column("Age").unwrap().value_counts(true,false).unwrap();
let age_not_arrived = train_df.clone().lazy().filter(col("Transported").eq(false)).collect().unwrap().column("Age").unwrap().value_counts(true,false).unwrap();
let age_arrived_vec:Vec<u32>= age_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
let age_not_arrived_vec:Vec<u32>= age_not_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
println!("{}",age_arrived); 
println!("{}",age_not_arrived);

/*Cabin1 */
let cabin_1_arrived = train_df.clone().lazy().filter(col("Transported").eq(true)).collect().unwrap().column("Cabin_1").unwrap().value_counts(true,false).unwrap();
let cabin_1_not_arrived = train_df.clone().lazy().filter(col("Transported").eq(false)).collect().unwrap().column("Cabin_1").unwrap().value_counts(true,false).unwrap();
let cabin_1_arrived_vec:Vec<u32>= cabin_1_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
let cabin_1_not_arrived_vec:Vec<u32>= cabin_1_not_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
println!("{}",cabin_1_arrived); 
println!("{}",cabin_1_not_arrived);
/*Cabin3 */
let cabin_3_arrived = train_df.clone().lazy().filter(col("Transported").eq(true)).collect().unwrap().column("Cabin_3").unwrap().value_counts(true,false).unwrap();
let cabin_3_not_arrived = train_df.clone().lazy().filter(col("Transported").eq(false)).collect().unwrap().column("Cabin_3").unwrap().value_counts(true,false).unwrap();
let cabin_3_arrived_vec:Vec<u32>= cabin_3_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
let cabin_3_not_arrived_vec:Vec<u32>= cabin_3_not_arrived.column("counts").unwrap().u32().unwrap().into_no_null_iter().collect();
println!("{}",cabin_3_arrived); 
println!("{}",cabin_3_not_arrived);

/*One_hot_ecnoding */
let  input_data= train_df.drop_many(&["Transported"]);
let cabin_1_data=  input_data.select(["Cabin_1"]).unwrap().to_dummies(None, false).unwrap();
let cabin_3_data=  input_data.select(["Cabin_3"]).unwrap().to_dummies(None, false).unwrap();
let destination_data=  input_data.select(["Destination"]).unwrap().to_dummies(None, false).unwrap();
let homeplanet_data=  input_data.select(["HomePlanet"]).unwrap().to_dummies(None, false).unwrap();

println!("{}",input_data);
println!("{}",cabin_1_data);
println!("{}",cabin_3_data);
println!("{}",destination_data);
println!("{}",homeplanet_data);
let input_data= input_data.hstack(cabin_1_data.get_columns()).unwrap();
let input_data= input_data.hstack(cabin_3_data.get_columns()).unwrap();
let input_data= input_data.hstack(destination_data.get_columns()).unwrap();
let input_data= input_data.hstack(homeplanet_data.get_columns()).unwrap();

println!("{}",input_data);

/*===================processing========================= */

/*==========result_data================ */
let result_df = result_df
    .clone()
    .lazy()
    .with_columns([
      col("Transported").cast(DataType::Float64)
    ])
    .collect().unwrap();
let result_id_df: &Series= result_df.column(&"PassengerId").unwrap();

let result_df: &Series= result_df.column(&"Transported").unwrap();

let result_train: Vec<f64> = result_df.f64().unwrap().into_no_null_iter().collect();
let result_train:Vec<i32>= result_train.iter().map(|x|*x as i32).collect();

/*==========result_data================ */
/*===================model========================= */
let train_df = train_df
    .clone()
    .lazy()
    .with_columns([
      col("Transported").cast(DataType::Float64)
    ])
    .collect().unwrap();

    println!("{}",train_df);

let y_target :Vec<f64> = train_df.column("Transported").unwrap().f64().unwrap().into_no_null_iter().collect();
let y_target :Vec<i32> = y_target.into_iter().map(|x|x as i32).collect();

let x_data= input_data.drop_many(&["Cab_1_A", "Cab_1_G", "Cab_1_T", "Cab_1_D", "Cab_1_C", "VIP", "HomePlanet_Mars", "Destination_PSO J318.5-22","HomePlanet","Destination","Cabin_1","Cabin_3"]);
println!("{}",x_data);
println!("데이터 미리보기:{}",x_data.head(None));
println!("데이터 정보 확인:{:?}",x_data.schema());

println!("{}",x_data);
let x_data= x_data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
let mut x_train: Vec<Vec<_>> = Vec::new();
for row in x_data.outer_iter() {
    let row_vec: Vec<_> = row.iter().cloned().collect();
    x_train.push(row_vec);
}

let x_data= DenseMatrix::from_2d_vec(&x_train);

let pca: PCA<f64, DenseMatrix<f64>> = PCA::fit(&x_data, PCAParameters::default().with_n_components(10)).unwrap(); // Reduce number of features to 2

let pca_train_data= pca.transform(&x_data).unwrap();

// let (x_train, x_test, y_train, y_test) = train_test_split(&pca_train_data, &y_target, 0.2, true, None);
/*===================test========================= */
let test_df = test_df
    .clone()
    .lazy()
    .with_columns([
        col("CryoSleep").cast(DataType::Float64).fill_null(0),
        col("VIP").cast(DataType::Float64).fill_null(0),
        col("Cabin").fill_null(lit("G/734/S")),
        col("HomePlanet").fill_null(lit("Earth")),
        col("Destination").fill_null(lit("TRAPPIST-1e")),
        col("ShoppingMall").fill_null(col("ShoppingMall").median()),
        col("VRDeck").fill_null(col("VRDeck").median()),
        col("FoodCourt").fill_null(col("FoodCourt").median()),
        col("Spa").fill_null(col("Spa").median()),
        col("RoomService").fill_null(col("RoomService").median()),
        col("Age").fill_null(col("Age").median()),
    ])
    .collect().unwrap();
let  test_df= test_df.clone().lazy().with_columns(
    [col("Cabin").str().split(lit("/")).list().get(lit(0)).alias("Cabin_1"),
           col("Cabin").str().split(lit("/")).list().get(lit(1)).alias("Cabin_2").cast(DataType::Float64),
            col("Cabin").str().split(lit("/")).list().get(lit(2)).alias("Cabin_3"),
            col("VIP").cast(DataType::Int64),
            col("CryoSleep").cast(DataType::Int64)
    
            ]
    ).collect().unwrap();
let test_df= test_df.drop_many(&["PassengerId","Name","Cabin"]);
let  test_df= test_df.clone().lazy().with_columns(
    [col("VIP").cast(DataType::Int64),
           col("CryoSleep").cast(DataType::Int64),
            ]
    ).collect().unwrap();


let  input_data_test= test_df.drop_many(&["Transported"]);
let cabin_1_data_test=  input_data_test.select(["Cabin_1"]).unwrap().to_dummies(None, false).unwrap();
let cabin_3_data_test=  input_data_test.select(["Cabin_3"]).unwrap().to_dummies(None, false).unwrap();
let destination_data_test=  input_data_test.select(["Destination"]).unwrap().to_dummies(None, false).unwrap();
let homeplanet_data_test=  input_data_test.select(["HomePlanet"]).unwrap().to_dummies(None, false).unwrap();


let input_data_test= input_data_test.hstack(cabin_1_data_test.get_columns()).unwrap();
let input_data_test= input_data_test.hstack(cabin_3_data_test.get_columns()).unwrap();
let input_data_test= input_data_test.hstack(destination_data_test.get_columns()).unwrap();
let input_data_test= input_data_test.hstack(homeplanet_data_test.get_columns()).unwrap();

println!("{}",input_data_test);
let test_data= input_data_test.drop_many(&["Cab_1_A", "Cab_1_G", "Cab_1_T", "Cab_1_D", "Cab_1_C", "VIP", "HomePlanet_Mars", "Destination_PSO J318.5-22","HomePlanet","Destination","Cabin_1","Cabin_3"]);

let test_data= test_data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
let mut test_train: Vec<Vec<_>> = Vec::new();
for row in test_data.outer_iter() {
    let row_vec: Vec<_> = row.iter().cloned().collect();
    test_train.push(row_vec);
}

let test_data= DenseMatrix::from_2d_vec(&test_train);

let pca_test: PCA<f64, DenseMatrix<f64>> = PCA::fit(&test_data, PCAParameters::default().with_n_components(10)).unwrap(); // Reduce number of features to 2
let pca_test_data= pca_test.transform(&test_data).unwrap();


/*===================test========================= */



let pca_model = RandomForestClassifier::fit(&pca_train_data, &y_target, Default::default()).unwrap();
let y_hat: Vec<i32> = pca_model.predict(&pca_test_data).unwrap(); // use the same data for prediction
let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&result_train,&y_hat);


/*===================model========================= */
/*===================제출용 파일========================= */

// //random forest 가 가장 빠르기 때문에

let survived_series = Series::new("Transported", y_hat.into_iter().collect::<Vec<i32>>()).cast(&DataType::Boolean).unwrap();
let passenger_id_series =result_id_df.clone();

let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();

let mut df= df.clone().lazy().with_columns([
    col("Transported").cast(DataType::String)
]).collect().unwrap();

let mut df= df.clone().lazy().with_columns([
    when(col("Transported").eq(lit("true"))).then(lit("True")).otherwise(lit("False")).alias("Transported")
]).collect().unwrap();
let mut output_file: File = File::create("./datasets/spaceship-titanic/out.csv").unwrap();

CsvWriter::new(&mut output_file)
    // .has_header(true)
    .finish(&mut df)
    .unwrap();


/*===================result========================= */
}
