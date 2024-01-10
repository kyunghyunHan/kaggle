use polars::prelude::*;
use std::fs::File;
use xgboost::{parameters, Booster, DMatrix};
pub fn main() {
    /*===============data============= */
    let train_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s3e20/train.csv")
        .unwrap()
        .has_header(true)
        .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
        .finish()
        .unwrap()
        .drop_many(&[
            "UvAerosolLayerHeight_aerosol_height",
            "UvAerosolLayerHeight_aerosol_pressure",
            "UvAerosolLayerHeight_aerosol_optical_depth",
            "UvAerosolLayerHeight_sensor_zenith_angle",
            "UvAerosolLayerHeight_sensor_azimuth_angle",
            "UvAerosolLayerHeight_solar_azimuth_angle",
            "UvAerosolLayerHeight_solar_zenith_angle",
            "ID_LAT_LON_YEAR_WEEK",
        ]);
    println!("{:?}", train_df.shape());
    println!("{:?}", train_df.null_count());
    println!("{:?}", train_df.schema());
    println!("{:?}", train_df.dtypes());
    println!("{:?}", train_df.width());

    let test_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s3e20/test.csv")
        .unwrap()
        .has_header(true)
        .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
        .finish()
        .unwrap()
        .drop_many(&[
            "UvAerosolLayerHeight_aerosol_height",
            "UvAerosolLayerHeight_aerosol_pressure",
            "UvAerosolLayerHeight_aerosol_optical_depth",
            "UvAerosolLayerHeight_sensor_zenith_angle",
            "UvAerosolLayerHeight_sensor_azimuth_angle",
            "UvAerosolLayerHeight_solar_azimuth_angle",
            "UvAerosolLayerHeight_solar_zenith_angle",
            "ID_LAT_LON_YEAR_WEEK",
        ]);

    let submission_df: DataFrame =
        CsvReader::from_path("./datasets/playground-series-s3e20/sample_submission.csv")
            .unwrap()
            .finish()
            .unwrap();
    /*==============null 값 처리============= */
    let train_df = train_df.fill_null(FillNullStrategy::Mean).unwrap();
    let test_df = test_df.fill_null(FillNullStrategy::Mean).unwrap();

    println!("{:?}", submission_df.null_count());
    println!("{}", train_df);
    /*==============model============= */

    println!("{:?}", train_df.shape());
    let x_train = train_df
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();
    let x_train = x_train.into_shape(79023 * 68).unwrap();
    let x_train: Vec<f32> = x_train.into_iter().collect();
    let train_rows = 79023;
    let y_train: Vec<f32> = train_df
        .column("emission")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|x| x as f32)
        .collect();

    let mut dtrain = DMatrix::from_dense(&x_train, train_rows).unwrap();
    dtrain.set_labels(&y_train).unwrap();

    println!("{:?}", test_df.shape());

    let x_test = test_df
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();
    let x_test = x_test.into_shape(24353 * 67).unwrap();
    let x_test: Vec<f32> = x_test.into_iter().collect();
    let test_rows = 24353;
    let y_test: Vec<f32> = submission_df
        .column("emission")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|x| x as f32)
        .collect();

    let mut dtest = DMatrix::from_dense(&x_test, test_rows).unwrap();
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

    /*=============file============= */

    let submussuin_series = Series::new("emission", y.into_iter().collect::<Vec<f64>>());
    let id_series = submission_df.column("ID_LAT_LON_YEAR_WEEK").unwrap().clone();

    let  mut df: DataFrame = DataFrame::new(vec![id_series, submussuin_series]).unwrap();



    let mut output_file: File = File::create("./datasets/playground-series-s3e20/out.csv").unwrap();

    CsvWriter::new(&mut output_file)
        .finish(&mut df)
        .unwrap();
    /*==============결과============= */

}
