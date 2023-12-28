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
use polars::prelude::*;
use polars_core::prelude::*;
use polars_io::prelude::*;
use plotters::prelude::*;

use std::str;
pub fn main(){
    /*데이터불러오기 */
    let mut train_df: DataFrame = CsvReader::from_path("./datasets/spaceship_titanic/train.csv")
    .unwrap()
    .finish().unwrap();

    let  test_df: DataFrame = CsvReader::from_path("./datasets/spaceship_titanic/test.csv")
    .unwrap()
    .finish().unwrap();


    println!("데이터 미리보기:{}",train_df.head(None));
    println!("데이터 정보 확인:{:?}",train_df.schema());

    /* 결측치 확인
    결측치가 5프로 내외
    */
    println!("{}",train_df.null_count());
    
    /*수치형 데이터 확인*/
    println!("{:?}",train_df.describe(None));

   let a=  train_df.column("PassengerId").unwrap();

   let y_train: Vec<u32> = a.u32().unwrap().into_no_null_iter().collect();
   let y_train:Vec<i32>= y_train.iter().map(|x|*x as i32).collect();
 

    let root_area = BitMapBackend::new("2.13.png", (600, 400))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Prime Distribution", ("sans-serif", 40))
        .build_cartesian_2d([0, 20,40,60,80].into_segmented(), 0..2000)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();


    ctx.draw_series(
        Histogram::vertical(&ctx)
        .margin(100)
        .data(y_train.iter().map(|x| (x, 1)))
    ).unwrap();
}
