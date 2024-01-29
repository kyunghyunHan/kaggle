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

    /*========CryoSleep======= */
    //상관 관계 확인
    let cryo_sleep_corr = pearson_corr(
        train_df.column("CryoSleep").unwrap().f64().unwrap(),
        train_df
            .column("Transported")
            .unwrap()
            .i32()
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap(),
        1,
    )
    .unwrap();
    println!("{}", cryo_sleep_corr); //0.46013235785425843
                                     //상관 관계가 높은것으로 판단댐

    /*========Cabin======= */
    //Cabin / 따라 나누기
    let train_df = train_df
        .clone()
        .lazy()
        .with_columns([
            col("Cabin")
                .str()
                .split(lit("/"))
                .list()
                .get(lit(0))
                .alias("Cabin_1"),
            col("Cabin")
                .str()
                .split(lit("/"))
                .list()
                .get(lit(1))
                .alias("Cabin_2")
                .cast(DataType::Float64),
            col("Cabin")
                .str()
                .split(lit("/"))
                .list()
                .get(lit(2))
                .alias("Cabin_3"),
            col("VIP").cast(DataType::Int64),
            col("CryoSleep").cast(DataType::Int64),
        ])
        .collect()
        .unwrap()
        .drop("Cabin")
        .unwrap();

    let test_df: DataFrame = test_df
        .clone()
        .lazy()
        .with_columns([
            col("Cabin")
                .str()
                .split(lit("/"))
                .list()
                .get(lit(0))
                .alias("Cabin_1"),
            col("Cabin")
                .str()
                .split(lit("/"))
                .list()
                .get(lit(1))
                .alias("Cabin_2")
                .cast(DataType::Float64),
            col("Cabin")
                .str()
                .split(lit("/"))
                .list()
                .get(lit(2))
                .alias("Cabin_3"),
            col("VIP").cast(DataType::Int64),
            col("CryoSleep").cast(DataType::Int64),
        ])
        .collect()
        .unwrap()
        .drop("Cabin")
        .unwrap();

    let train_cabin_1 = train_df
        .select(["Cabin_1"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let test_cabin_1 = test_df
        .select(["Cabin_1"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();
    let train_df = train_df
        .hstack(train_cabin_1.get_columns())
        .unwrap()
        .drop("Cabin_1")
        .unwrap();
    let test_df = test_df
        .hstack(test_cabin_1.get_columns())
        .unwrap()
        .drop("Cabin_1")
        .unwrap();
    //상관 관계확인

    println!("head:{:?}", train_df.schema());
    /*========Cabin_1======= */

    let cabin_1_a_corr = pearson_corr(
        train_df.column("Cabin_1_A").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let cabin_1_b_corr = pearson_corr(
        train_df.column("Cabin_1_B").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let cabin_1_c_corr = pearson_corr(
        train_df.column("Cabin_1_C").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let cabin_1_d_corr = pearson_corr(
        train_df.column("Cabin_1_D").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let cabin_1_e_corr = pearson_corr(
        train_df.column("Cabin_1_E").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let cabin_1_f_corr = pearson_corr(
        train_df.column("Cabin_1_F").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let cabin_1_g_corr = pearson_corr(
        train_df.column("Cabin_1_G").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let cabin_1_t_corr = pearson_corr(
        train_df.column("Cabin_1_T").unwrap().i32().unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();

    println!("cabin_1_a_corr:{}", cabin_1_a_corr);
    println!("cabin_1_b_corr:{}", cabin_1_b_corr);
    println!("cabin_1_c_corr:{}", cabin_1_c_corr);
    println!("cabin_1_d_corr:{}", cabin_1_d_corr);
    println!("cabin_1_e_corr:{}", cabin_1_e_corr);
    println!("cabin_1_f_corr:{}", cabin_1_f_corr);
    println!("cabin_1_g_corr:{}", cabin_1_g_corr);
    println!("cabin_1_t_corr:{}", cabin_1_t_corr);

    //상관관계 낮은 값 삭제

    let train_df = train_df.drop_many(&[
        "Cabin_1_A",
        "Cabin_1_D",
        "Cabin_1_E",
        "Cabin_1_F",
        "Cabin_1_T",
    ]);
    let test_df: DataFrame = test_df.drop_many(&[
        "Cabin_1_A",
        "Cabin_1_D",
        "Cabin_1_E",
        "Cabin_1_F",
        "Cabin_1_T",
    ]);
    println!("schema:{:?}", train_df.schema());
    /*========Destination======= */

    let train_destination = train_df
        .select(["Destination"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let test_destination = test_df
        .select(["Destination"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();
    let train_df = train_df
        .hstack(train_destination.get_columns())
        .unwrap()
        .drop("Destination")
        .unwrap();
    let test_df = test_df
        .hstack(test_destination.get_columns())
        .unwrap()
        .drop("Destination")
        .unwrap();
    println!("schema:{:?}", train_df.schema());

    let destination_55_corr = pearson_corr(
        train_df
            .column("Destination_55 Cancri e")
            .unwrap()
            .i32()
            .unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let destination_pro_corr = pearson_corr(
        train_df
            .column("Destination_PSO J318.5-22")
            .unwrap()
            .i32()
            .unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();
    let destination_trrappist_corr = pearson_corr(
        train_df
            .column("Destination_TRAPPIST-1e")
            .unwrap()
            .i32()
            .unwrap(),
        train_df.column("Transported").unwrap().i32().unwrap(),
        1,
    )
    .unwrap();

    println!("destination_55_corr:{}", destination_55_corr);
    println!("destination_pro_corr:{}", destination_pro_corr);
    println!("destination_trrappist_corr:{}", destination_trrappist_corr);
     

    let train_df= train_df.drop_many(&["Destination_PSO J318.5-22","Destination_TRAPPIST-1e"]);
    let test_df= test_df.drop_many(&["Destination_PSO J318.5-22","Destination_TRAPPIST-1e"]);

    /*========Age======= */

    println!("AGE:{}",train_df.column("Age").unwrap());

    /*========VIP======= */
    /*========RoomService======= */
    /*========FoodCourt======= */
    /*========ShoppingMall======= */
    /*========Spa======= */
    /*========VRDeck======= */
    /*========Name======= */
}
