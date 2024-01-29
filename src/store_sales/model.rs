use chrono::{TimeZone, Utc};
use plotters::prelude::*;
use polars::prelude::*;
use std::fs::File;
use xgboost::parameters;
use xgboost::Booster;
use xgboost::DMatrix;
pub fn main() {
    /*===================data========================= */
    //train
    let train_df: DataFrame =
        CsvReader::from_path("./datasets/store-sales-time-series-forecasting/train.csv")
            .unwrap()
            .finish()
            .unwrap(); //null 없음
                       //test
    let test_df: DataFrame =
        CsvReader::from_path("./datasets/store-sales-time-series-forecasting/test.csv")
            .unwrap()
            .finish()
            .unwrap(); //null 없음
                       //submission
    let submission_df: DataFrame = CsvReader::from_path(
        "./datasets/store-sales-time-series-forecasting/sample_submission.csv",
    )
    .unwrap()
    .finish()
    .unwrap(); //null 없음

    println!("{:?}", train_df.shape());
    println!("{}", train_df.head(None));
    println!("{}", train_df.head(None));
    println!("{:?}", train_df.schema());
    println!("{}", train_df.column("date").unwrap());
    /*===================data========================= */
    /*===================visualization========================= */
    //날자별 가격
    let date_data: Vec<&str> = train_df
        .column("date")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap())
        .collect();
    let sales_data: Vec<f64> = train_df
        .column("sales")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    let mut data: Vec<(&str, f64)> = vec![];
    for i in 0..date_data.len() {
        data.push((date_data[i], sales_data[i]))
    }
    let date_root =
        BitMapBackend::new("./src/store_sales/date_series.png", (800, 600)).into_drawing_area();
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
    date_ctx
        .draw_series(LineSeries::new(
            (0..).zip(data.iter()).map(|(idx, &price)| {
                let date = Utc.ymd(
                    price.0.split("-").collect::<Vec<&str>>()[0]
                        .to_owned()
                        .parse::<i32>()
                        .unwrap(),
                    price.0.split("-").collect::<Vec<&str>>()[1]
                        .parse::<u32>()
                        .unwrap(),
                    price.0.split("-").collect::<Vec<&str>>()[2]
                        .parse::<u32>()
                        .unwrap(),
                );
                (date, price.1)
            }),
            &BLUE,
        ))
        .unwrap();

    //store_nbr
    let date_data: Vec<&str> = train_df
        .column("date")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap())
        .collect();
    let store_nbr_data: Vec<i64> = train_df
        .column("store_nbr")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let store_nbr_data: Vec<f64> = store_nbr_data.into_iter().map(|x| x as f64).collect();
    let mut store_nbr_data_vec: Vec<(&str, f64)> = vec![];
    for i in 0..store_nbr_data.len() {
        store_nbr_data_vec.push((date_data[i], store_nbr_data[i]))
    }

    let store_nbr_root =
        BitMapBackend::new("./src/store_sales/store_nbr.png", (800, 600)).into_drawing_area();
    store_nbr_root.fill(&WHITE).unwrap();

    let start_date = Utc.ymd(2013, 1, 1);
    let end_date = Utc.ymd(2017, 8, 20);

    let mut store_nbr_chart = ChartBuilder::on(&store_nbr_root)
        .margin(20)
        .caption("store_nbr", ("sans-serif", 40))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(start_date..end_date, (0.0..800.0))
        .unwrap();

    store_nbr_chart
        .configure_mesh()
        .x_desc("Length")
        .y_desc("Weight")
        .draw()
        .unwrap();
    store_nbr_chart
        .draw_series(LineSeries::new(
            (0..).zip(store_nbr_data_vec.iter()).map(|(idx, &price)| {
                let date = Utc.ymd(
                    price.0.split("-").collect::<Vec<&str>>()[0]
                        .to_owned()
                        .parse::<i32>()
                        .unwrap(),
                    price.0.split("-").collect::<Vec<&str>>()[1]
                        .parse::<u32>()
                        .unwrap(),
                    price.0.split("-").collect::<Vec<&str>>()[2]
                        .parse::<u32>()
                        .unwrap(),
                );
                (date, price.1)
            }),
            &BLUE,
        ))
        .unwrap();

    //store_nbr
    let date_data: Vec<&str> = train_df
        .column("date")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap())
        .collect();
    let onpromotion_data: Vec<i64> = train_df
        .column("onpromotion")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let onpromotion_data: Vec<f64> = onpromotion_data.into_iter().map(|x| x as f64).collect();
    let mut onpromotion_data_vec: Vec<(&str, f64)> = vec![];
    for i in 0..onpromotion_data.len() {
        onpromotion_data_vec.push((date_data[i], onpromotion_data[i]))
    }
    let onpromotion_root =
        BitMapBackend::new("./src/store_sales/onpromotion.png", (800, 600)).into_drawing_area();
    onpromotion_root.fill(&WHITE).unwrap();

    let start_date = Utc.ymd(2013, 1, 1);
    let end_date = Utc.ymd(2017, 8, 20);

    let mut onpromotion_chart = ChartBuilder::on(&onpromotion_root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("onpromotion", ("sans-serif", 40))
        .build_cartesian_2d(start_date..end_date, (0.0..800.0))
        .unwrap();

    onpromotion_chart
        .configure_mesh()
        .x_desc("Length")
        .y_desc("Weight")
        .draw()
        .unwrap();
    onpromotion_chart
        .draw_series(LineSeries::new(
            (0..).zip(onpromotion_data_vec.iter()).map(|(idx, &price)| {
                let date = Utc.ymd(
                    price.0.split("-").collect::<Vec<&str>>()[0]
                        .to_owned()
                        .parse::<i32>()
                        .unwrap(),
                    price.0.split("-").collect::<Vec<&str>>()[1]
                        .parse::<u32>()
                        .unwrap(),
                    price.0.split("-").collect::<Vec<&str>>()[2]
                        .parse::<u32>()
                        .unwrap(),
                );
                (date, price.1)
            }),
            &BLUE,
        ))
        .unwrap();

    /*===================visualization========================= */
    /*===================model========================= */

    /*===================xgboost========================= */
    let date_data: Vec<&str> = train_df
        .column("date")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap())
        .collect();
    let test_date_data: Vec<&str> = test_df
        .column("date")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap())
        .collect();

    let num_rows = 3000888;
    let test_sales_data: Vec<f64> = submission_df
        .column("sales")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let test_sales_data: Vec<f32> = test_sales_data.iter().map(|&x| x as f32).collect();

    let sales_data: Vec<f64> = train_df
        .column("sales")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let sales_data: Vec<f32> = sales_data.iter().map(|&x| x as f32).collect();

    let transformed_data: Vec<f32> = date_data
        .iter()
        .map(|&x| {
            let parts: Vec<&str> = x.split("-").collect(); // 날짜를 "-" 기준으로 분할하여 벡터에 저장합니다.
            let concatenated = format!("{}{}{}", parts[0], parts[1], parts[2]); // "-" 제거 후 문자열을 합칩니다.
            concatenated.parse::<f32>().unwrap_or(0.0) // f32 숫자로 변환합니다. 변환이 실패하면 0.0으로 반환합니다.
        })
        .collect();

    let test_transformed_data: Vec<f32> = test_date_data
        .iter()
        .map(|&x| {
            let parts: Vec<&str> = x.split("-").collect(); // 날짜를 "-" 기준으로 분할하여 벡터에 저장합니다.
            let concatenated = format!("{}{}{}", parts[0], parts[1], parts[2]); // "-" 제거 후 문자열을 합칩니다.
            concatenated.parse::<f32>().unwrap_or(0.0) // f32 숫자로 변환합니다. 변환이 실패하면 0.0으로 반환합니다.
        })
        .collect();
    let mut dtrain: DMatrix = DMatrix::from_dense(&transformed_data, num_rows).unwrap();

    dtrain.set_labels(&sales_data).unwrap();
    let num_rows = 28512;
    let mut dtest = DMatrix::from_dense(&test_transformed_data, num_rows).unwrap();
    dtest.set_labels(&test_sales_data).unwrap();

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
    /*===================제출용 파일========================= */
    let survived_series = Series::new("sales", &y);
    let passenger_id_series = test_df.column("id").unwrap().clone();
    let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
    let mut output_file: File =
        File::create("./datasets/store-sales-time-series-forecasting/out.csv").unwrap();
    CsvWriter::new(&mut output_file).finish(&mut df).unwrap();
}
