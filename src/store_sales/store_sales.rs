use plotters::prelude::*;
use linfa::prelude::*;
use std::{fs::File, cmp::Ordering};
use polars::{prelude::*, series::SeriesTrait};
use chrono::{Utc, TimeZone};



pub fn main(){
   /*===================data========================= */
   //train
   let  mut train_df: DataFrame = CsvReader::from_path("./datasets/store-sales-time-series-forecasting/train.csv").unwrap()
   .finish().unwrap();//null 없음
     //test
    let  mut test_df: DataFrame = CsvReader::from_path("./datasets/store-sales-time-series-forecasting/test.csv").unwrap()
     .finish().unwrap();//null 없음
    //submission
    let  mut submission_df: DataFrame = CsvReader::from_path("./datasets/store-sales-time-series-forecasting/sample_submission.csv").unwrap()
    .finish().unwrap();//null 없음

    println!("{:?}",train_df.shape());
    println!("{}",train_df.head(None));
    println!("{}",train_df.head(None));
    println!("{:?}",train_df.schema());
    println!("{}",train_df.column("date").unwrap());
    /*===================data========================= */
    /*===================visualization========================= */
    //날자별 가격
    let date_data:Vec<&str>= train_df.column("date").unwrap().str().unwrap().into_iter().map(|x|x.unwrap()).collect();
    let sales_data:Vec<f64>= train_df.column("sales").unwrap().f64().unwrap().into_no_null_iter().collect();   
    
    let mut  data: Vec<(&str,f64)>= vec![];
    for i in 0..date_data.len(){
      data.push((date_data[i],sales_data[i]))
    }
    let date_root = BitMapBackend::new("./src/store_sales/date_series.png", (800, 600)).into_drawing_area();
    date_root.fill(&WHITE).unwrap();

     let start_date = Utc.ymd(2013, 1, 1);
     let end_date = Utc.ymd(2017, 8, 20);

    let mut date_ctx = ChartBuilder::on(&date_root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("date_time_series", ("sans-serif", 40))
        .build_cartesian_2d(start_date..end_date, (0.0..25000.0).step(5000f64))
        .unwrap();

    date_ctx.configure_mesh().draw().unwrap();
    date_ctx.draw_series(
        LineSeries::new(
            (0..).zip(data.iter()).map(|(idx, &price)| {                
                let date = Utc.ymd(price.0.split("-").collect::<Vec<&str>>()[0].to_owned().parse::<i32>().unwrap(),price.0.split("-").collect::<Vec<&str>>()[1].parse::<u32>().unwrap(), price.0.split("-").collect::<Vec<&str>>()[2].parse::<u32>().unwrap());
                (date, price.1)
            }),
            &BLUE,
        )
    ).unwrap();


    // let date_data= date_data.to_ndarray::<FromDataUtf8>(IndexOrder::Fortran).unwrap();
    /*===================visualization========================= */

}