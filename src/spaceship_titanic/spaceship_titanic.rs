/*

제츨파일

PassengerId :PassengerId
HomePlanet- 승객이 출발한 행성, 일반적으로 영주권이 있는 행성입니다.

CryoSleep- 승객이 항해 기간 동안 애니메이션을 정지하도록 선택했는지 여부를 나타냅니다. 냉동 수면 중인 승객은 객실에 갇혀 있습니다.
Cabin- 승객이 머무르는 객실 번호. Port 또는 Starboarddeck/num/side 의 형식 을 취side 합니다 .PS
Destination- 승객이 내릴 행성.
Age- 승객의 나이.
VIP- 승객이 항해 중 특별 VIP 서비스 비용을 지불했는지 여부.
RoomService, FoodCourt, ShoppingMall, Spa, - 우주선 타이타닉VRDeck 의 다양한 고급 편의 시설 각각에 대해 승객이 청구한 금액입니다 .
Name- 승객의 이름과 성.
Transported : target
*/
use std::fs::File;
use ndarray::prelude::*;
use polars::{prelude::*, lazy::dsl::col};
use plotters::prelude::*;

pub fn main(){
    /*===================data 불러오기========================= */

    let  mut train_df: DataFrame = CsvReader::from_path("./datasets/spaceship-titanic/train.csv").unwrap()
    .finish().unwrap();

    let  test_df: DataFrame = CsvReader::from_path("./datasets/spaceship-titanic/test.csv").unwrap()
    .finish().unwrap();


    println!("데이터 미리보기:{}",train_df.head(None));
    println!("데이터 정보 확인:{:?}",train_df.schema());
    println!("데이터 미리보기:{}",test_df.head(None));
    println!("데이터 정보 확인:{:?}",test_df.schema());

    println!("결측치 확인:{:?}",train_df.null_count());
    println!("수치형 데이터 확인:{:?}",train_df.describe(None).unwrap());

    println!("범주형 데이터 확인:{:?}",train_df.describe(Some(&[0f64])).unwrap());
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
        col("HomePlanet").fill_null(lit("Eath")),
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

    /*===================processing========================= */


/*===================result========================= */
}
