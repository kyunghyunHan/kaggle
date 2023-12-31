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
    
    /*===================data 불러오기========================= */
    /*===================히스토그램 그리기========================= */
    let root = BitMapBackend::new("./src/spaceship_titanic/histogram.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let drawing_areas = root.split_evenly((2, 3));

    for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
        let mut chart_builder = ChartBuilder::on(&drawing_area);

    chart_builder.margin(5).set_left_and_bottom_label_area_size(20);
    let mut chart_context = chart_builder.build_cartesian_2d((1..10).into_segmented(), 0..9).unwrap();
    chart_context.configure_mesh().draw().unwrap();
    chart_context.draw_series(Histogram::vertical(&chart_context).style(BLUE.filled()).margin(10)
    .data((0..10).map(|x| (x, x)))).unwrap();
    }
    /*===================히스토그램 그리기========================= */

    /*===================processing========================= */
//   train_df.with_column(  train_df.column("CryoSleep").unwrap().fill_null(FillNullStrategy::Zero).unwrap()).unwrap();

//   let train_df= train_df.clone()
//       .lazy()
//       .select([
//           col("CryoSleep").cast(DataType::Float64),
//         //   col("floats").cast(DataType::Boolean),
//       ])
//       .collect().unwrap();
  let  train_df:&mut DataFrame= train_df.with_column(  train_df.column("CryoSleep").unwrap().cast(&DataType::Float64).unwrap().fill_null(FillNullStrategy::Zero).unwrap()).unwrap();

   println!("범주형 데이터 확인:{:?}",train_df.describe(Some(&[0f64])).unwrap());
   println!("수치형 데이터 확인:{:?}",train_df.describe(None).unwrap());
   
  println!("{:?}",train_df.select(["CryoSleep"]).unwrap().null_count());
  println!("{}",train_df.null_count())
    


    /*===================processing========================= */
    /*===================result========================= */
}
