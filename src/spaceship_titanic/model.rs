use plotters::prelude::*;
use polars::prelude::cov::pearson_corr;
use polars::{
    lazy::dsl::{col, when},
    prelude::*,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::*;
use smartcore::{
    decomposition::pca::{PCAParameters, PCA},
    ensemble::random_forest_classifier::RandomForestClassifier,
};
use std::fs::File;
pub fn main() {
    /*===================data 불러오기========================= */

    let train_df: DataFrame = CsvReader::from_path("./datasets/spaceship-titanic/train.csv")
        .unwrap()
        .finish()
        .unwrap()
        .drop("PassengerId")
        .unwrap();

    let test_df: DataFrame = CsvReader::from_path("./datasets/spaceship-titanic/test.csv")
        .unwrap()
        .finish()
        .unwrap()
        .drop("PassengerId")
        .unwrap();
    let submission_df = CsvReader::from_path("./datasets/spaceship-titanic/sample_submission.csv")
        .unwrap()
        .finish()
        .unwrap();

    println!("head:{}", train_df.head(Some(3)));
    println!("schema:{:?}", train_df.schema());

    //null값 확인
    println!("{}", test_df.null_count());

    //null값 처리
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
            col("Transported").cast(DataType::Int32),
        ])
        .collect()
        .unwrap();
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
        .collect()
        .unwrap();
    //null값을 가장 많이 사용된 값을 사용

    println!("{}", train_df.null_count());

    /*===============HomePlanet=============== */
    let home_planet_dummies = train_df
        .select(["HomePlanet"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let home_planet_test_dummies = test_df
        .select(["HomePlanet"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();
    let train_df = train_df
        .hstack(home_planet_dummies.get_columns())
        .unwrap()
        .drop("HomePlanet")
        .unwrap();
    let test_df = test_df
        .hstack(home_planet_test_dummies.get_columns())
        .unwrap()
        .drop("HomePlanet")
        .unwrap();
    //

    let homeplanet_earth_corr = pearson_corr(
        train_df.column("HomePlanet_Earth").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let homeplanet_europa_corr = pearson_corr(
        train_df.column("HomePlanet_Europa").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();

    let homeplanet_mars_corr = pearson_corr(
        train_df.column("HomePlanet_Mars").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();

    println!("{}", homeplanet_earth_corr); //-0.16884536382739204
    println!("{}", homeplanet_europa_corr); //0.17691648731695503
    println!("{}", homeplanet_mars_corr); //0.019543527194644604
                                          //mars는 019543527194644604 이므로 상관관계가 낮다고 판단 삭제

    let train_df = train_df.drop("HomePlanet_Mars").unwrap();
    let test_df = test_df.drop("HomePlanet_Mars").unwrap();
    println!("schema:{:?}", train_df.schema());
    /*========CryoSleep======= */
    //상관 관계 확인
    let cryo_sleep_corr = pearson_corr(
        train_df.column("CryoSleep").unwrap().f64().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap().cast(&DataType::Float64).unwrap().f64().unwrap(),
        1,
    )
    .unwrap();
    println!("{}", cryo_sleep_corr); //0.46013235785425843
    //상관 관계가 높은것으로 판단댐


    /*========Cabin======= */

    
    /*========Destination======= */
    /*========Age======= */
    /*========VIP======= */
    /*========RoomService======= */
    /*========FoodCourt======= */
    /*========ShoppingMall======= */
    /*========Spa======= */
    /*========VRDeck======= */
    /*========Name======= */
}
