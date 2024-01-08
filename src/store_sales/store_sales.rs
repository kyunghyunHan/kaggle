use plotters::prelude::*;
use linfa::prelude::*;
use polars::prelude::*;
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

    //store_nbr
    let store_nbr_data= train_df.select(["store_nbr","sales"]).unwrap();
     let store_nbr_data= store_nbr_data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
     let mut store_nbr_data_vec: Vec<Vec<_>> = Vec::new();
     for row in store_nbr_data.outer_iter() {
         let row_vec: Vec<_> = row.iter().cloned().collect();
         store_nbr_data_vec.push(row_vec);
     }
    
     let store_nbr_root = BitMapBackend::new("./src/store_sales/store_nbr.png", (800, 600)).into_drawing_area();
     store_nbr_root.fill(&WHITE).unwrap();
  
  
         let mut store_nbr_chart: ChartContext<'_, BitMapBackend<'_>, Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>> = ChartBuilder::on(&store_nbr_root)
         .margin(20)
         .x_label_area_size(40)
         .y_label_area_size(40)
         .build_cartesian_2d((0.0..60.0), (0.0..12500.0))
         .unwrap();
 
         store_nbr_chart
         .configure_mesh()
         .x_desc("Length")
         .y_desc("Weight")
         .draw()
         .unwrap();
        store_nbr_chart.draw_series(
            LineSeries::new(
                store_nbr_data_vec.iter().map(|x| {                
                    (x[0], x[1])
                }),
                &BLUE,
            )
        ).unwrap();
    
     //store_nbr
     let onpromotion_data= train_df.select(["onpromotion","sales"]).unwrap();
     let onpromotion_data= onpromotion_data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
     let mut onpromotion_data_vec: Vec<Vec<_>> = Vec::new();
     for row in onpromotion_data.outer_iter() {
         let row_vec: Vec<_> = row.iter().cloned().collect();
         onpromotion_data_vec.push(row_vec);
     }
    
     let onpromotion_root = BitMapBackend::new("./src/store_sales/onpromotion.png", (800, 600)).into_drawing_area();
     onpromotion_root.fill(&WHITE).unwrap();
  
  
         let mut onpromotion_chart: ChartContext<'_, BitMapBackend<'_>, Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>> = ChartBuilder::on(&onpromotion_root)
         .margin(20)
         .x_label_area_size(40)
         .y_label_area_size(40)
         .build_cartesian_2d((0.0..1000.0), (0.0..125000.0))
         .unwrap();
 
         onpromotion_chart
         .configure_mesh()
         .x_desc("Length")
         .y_desc("Weight")
         .draw()
         .unwrap();
        onpromotion_chart.draw_series(
            LineSeries::new(
                onpromotion_data_vec.iter().map(|x| {                
                    (x[0], x[1])
                }),
                &BLUE,
            )
        ).unwrap();
    
    
    // let date_data= date_data.to_ndarray::<FromDataUtf8>(IndexOrder::Fortran).unwrap();
    /*===================visualization========================= */

}