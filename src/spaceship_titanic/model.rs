use plotters::prelude::*;
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
        .unwrap();

    let test_df: DataFrame = CsvReader::from_path("./datasets/spaceship-titanic/test.csv")
        .unwrap()
        .finish()
        .unwrap();
    let submission_df = CsvReader::from_path("./datasets/spaceship-titanic/sample_submission.csv")
        .unwrap()
        .finish()
        .unwrap();

    println!("데이터 미리보기:{}", train_df.head(None));
    println!("데이터 정보 확인:{:?}", train_df.schema());
    println!("결측치 확인:{:?}", train_df.null_count());
    println!("결측치 확인:{:?}", test_df.null_count());

    /*================결측치 확인 후 결측치 채우기================ */
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

    /*===================히스토그램 그리기========================= */

    let age_data: Vec<f64> = train_df
        .column("Age")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let room_service_data: Vec<f64> = train_df
        .column("RoomService")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let food_court_data: Vec<f64> = train_df
        .column("FoodCourt")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let shopping_mall_data: Vec<f64> = train_df
        .column("ShoppingMall")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let spa_data: Vec<f64> = train_df
        .column("Spa")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let vr_deck_data: Vec<f64> = train_df
        .column("VRDeck")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    let root =
        BitMapBackend::new("./src/spaceship_titanic/histogram.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let data = [
        ("Age", &age_data),
        ("RoomService", &room_service_data),
        ("FoodCourt", &food_court_data),
        ("ShoppingMall", &shopping_mall_data),
        ("Spa", &spa_data),
        ("VRDeck", &vr_deck_data),
    ];
    let drawing_areas = root.split_evenly((3, 2));
    for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
        let mut chart_builder = ChartBuilder::on(&drawing_area);
        chart_builder
            .margin(5)
            .set_left_and_bottom_label_area_size(20)
            .caption(format!("{}", data[idx as usize - 1].0), ("sans-serif", 40));
        let mut chart_context = chart_builder
            .build_cartesian_2d(
                (0u32..80u32).step(1).into_segmented(),
                (0f64..2100f64).step(100f64),
            )
            .unwrap();
        chart_context.configure_mesh().draw().unwrap();
        chart_context
            .draw_series(
                Histogram::vertical(&chart_context)
                    .style(BLUE.filled())
                    .margin(10)
                    .data(data[idx - 1].1.iter().map(|v| (*v as u32, 1f64))),
            )
            .unwrap();
    }
    /*==================Cabin========================= */

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
        .unwrap();

    let cabin_1 = train_df
        .group_by(&["Cabin_1"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap()
        .sort(&["Transported_mean"], vec![false, true], false)
        .unwrap();

    let cabin_2 = train_df
        .group_by(&["Cabin_2"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap()
        .sort(&["Transported_mean"], vec![false, true], false)
        .unwrap();

    let cabin_3 = train_df
        .group_by(&["Cabin_3"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap()
        .sort(&["Transported_mean"], vec![false, true], false)
        .unwrap();

    println!("Cabin_1::::{}", cabin_1);
    println!("Cabin_2::::{}", cabin_2);
    println!("Cabin_3::::{}", cabin_3);

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
    let train_df = train_df.hstack(train_cabin_1.get_columns()).unwrap();
    let test_df = test_df.hstack(test_cabin_1.get_columns()).unwrap();

    /*==================필요없는 값 삭제========================= */

    let train_df = train_df.drop_many(&[
        "PassengerId",
        "Name",
        "Cabin",
        "Cabin_1",
        "Cabin_3",
        "Cabin_1_T",
        "Cabin_1_E",
        "Cabin_1_D",
        "Cabin_1_F",
        "Cabin_1_A",
        "Cabin_1_G",
    ]);
    let test_df = test_df.drop_many(&[
        "PassengerId",
        "Name",
        "Cabin",
        "Cabin_1",
        "Cabin_3",
        "Cabin_1_T",
        "Cabin_1_E",
        "Cabin_1_D",
        "Cabin_1_F",
        "Cabin_1_A",
        "Cabin_1_G",
    ]);

    /*==================Vip========================= */
    let vip_result = train_df
        .group_by(&["VIP"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap()
        .sort(&["Transported_mean"], vec![false, true], false)
        .unwrap();
    println!("{}", vip_result);
    let vip_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(true))
        .collect()
        .unwrap()
        .column("VIP")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    let vip_not_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(false))
        .collect()
        .unwrap()
        .column("VIP")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    //VIP 삭제
    let train_df: DataFrame = train_df.drop("VIP").unwrap();
    let test_df: DataFrame = test_df.drop("VIP").unwrap();

    /*==================CryoSleep========================= */

    let cryosleep_result = train_df
        .group_by(&["CryoSleep"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap();
    println!("CryoSleep::{}", cryosleep_result);

    let cryosleep_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(true))
        .collect()
        .unwrap()
        .column("CryoSleep")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    let cryosleep_not_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(false))
        .collect()
        .unwrap()
        .column("CryoSleep")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    println!("Cryosleep Arrived::{}", cryosleep_arrived);
    println!("Cryosleep Not Arrived::{}", cryosleep_not_arrived);

    /*==================HomePlanet========================= */
    let homeplanet_result = train_df
        .group_by(&["HomePlanet"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap()
        .sort(&["Transported_mean"], vec![false, true], false)
        .unwrap();
    println!("HomePlanet::{}", homeplanet_result);

    let homeplanet_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(true))
        .collect()
        .unwrap()
        .column("HomePlanet")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    let homeplanet_not_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(false))
        .collect()
        .unwrap()
        .column("HomePlanet")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    println!("HomePlanet Arrived::{}", homeplanet_arrived);
    println!("HomePlanet Not Arrived::{}", homeplanet_not_arrived);

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

    let train_df = train_df.hstack(home_planet_dummies.get_columns()).unwrap();
    let test_df = test_df
        .hstack(home_planet_test_dummies.get_columns())
        .unwrap();

    let train_df: DataFrame =
        train_df.drop_many(&["HomePlanet", "HomePlanet_Earth", "HomePlanet_Mars"]);
    let test_df: DataFrame =
        test_df.drop_many(&["HomePlanet", "HomePlanet_Earth", "HomePlanet_Mars"]);

    // /*==================Destination========================= */
    let destination_result = train_df
        .group_by(&["Destination"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap()
        .sort(&["Transported_mean"], vec![false, true], false)
        .unwrap();

    let destination_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(true))
        .collect()
        .unwrap()
        .column("Destination")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    let destination_not_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Transported").eq(false))
        .collect()
        .unwrap()
        .column("Destination")
        .unwrap()
        .value_counts(true, false)
        .unwrap();

    println!("Destination::{}", destination_result);
    println!("Destination Arrived::{}", destination_arrived);
    println!("Destination Not Arrived::{}", destination_not_arrived);
    //중요하지 않으니 삭제
    let train_df: DataFrame = train_df.drop("Destination").unwrap();
    let test_df: DataFrame = test_df.drop("Destination").unwrap();

    /*==================Age========================= */

    let train_df = train_df
        .clone()
        .lazy()
        .with_columns([when(col("Age").lt(lit(10.0)))
            .then(lit(0f64))
            .when(col("Age").lt(lit(20.0)))
            .then(lit(10f64))
            .when(col("Age").lt(lit(30.0)))
            .then(lit(20f64))
            .when(col("Age").lt(lit(40.0)))
            .then(lit(30f64))
            .when(col("Age").lt(lit(50.0)))
            .then(lit(40f64))
            .when(col("Age").lt(lit(60.0)))
            .then(lit(50f64))
            .when(col("Age").lt(lit(70.0)))
            .then(lit(60f64))
            .when(col("Age").lt(lit(80.0)))
            .then(lit(70f64))
            .otherwise(19.9)
            .alias("Age")])
        .collect()
        .unwrap();
    println!("{}", train_df);
    println!("{}", train_df.column("Age").unwrap());

    let age_result = train_df
        .group_by(&["Age"])
        .unwrap()
        .select(["Transported"])
        .mean()
        .unwrap()
        .sort(&["Transported_mean"], vec![false, true], false)
        .unwrap();
    let train_df = train_df.drop("Age").unwrap();
    let test_df = test_df.drop("Age").unwrap();

    /*===================processing========================= */

    let train_df = train_df
        .clone()
        .lazy()
        .with_columns([col("Transported").cast(DataType::Float64)])
        .collect()
        .unwrap();

    let submission_df: DataFrame = submission_df
        .clone()
        .lazy()
        .with_columns([col("Transported").cast(DataType::Float64)])
        .collect()
        .unwrap();

    let y_train: Vec<f64> = train_df
        .column("Transported")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let train_df = train_df.drop("Transported").unwrap();
    let y_train: Vec<i32> = y_train.into_iter().map(|x| x as i32).collect();
    let y_test: Vec<f64> = submission_df
        .column("Transported")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let y_test: Vec<i32> = y_test.into_iter().map(|x| x as i32).collect();

    let x_train = train_df
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap();
    let mut x_train_vec: Vec<Vec<_>> = Vec::new();
    for row in x_train.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        x_train_vec.push(row_vec);
    }

    let x_test: ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f64>,
        ndarray::prelude::Dim<[usize; 2]>,
    > = test_df
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap();
    let mut x_test_vec: Vec<Vec<_>> = Vec::new();
    for row in x_test.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        x_test_vec.push(row_vec);
    }
    /*===================pca========================= */

    let x_train = DenseMatrix::from_2d_vec(&x_train_vec);
    let x_test: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_test_vec);
    println!("{:?}", test_df.schema());

    let pca_x_train: PCA<f64, DenseMatrix<f64>> = PCA::fit(
        &x_train,
        PCAParameters::default().with_n_components(train_df.shape().1),
    )
    .unwrap(); // Reduce number of features to 2

    let pca_x_train = pca_x_train.transform(&x_train).unwrap();

    let pca_x_test: PCA<f64, DenseMatrix<f64>> = PCA::fit(
        &x_test,
        PCAParameters::default().with_n_components(test_df.shape().1),
    )
    .unwrap(); // Reduce number of features to 2
    let pca_x_test = pca_x_test.transform(&x_test).unwrap();

    /*===================model========================= */

    let pca_model =
        RandomForestClassifier::fit(&pca_x_train, &y_train, Default::default()).unwrap();
    let y_hat: Vec<i32> = pca_model.predict(&pca_x_test).unwrap(); // use the same data for prediction
    let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&y_test, &y_hat);
    println!("{}", acc);
    // /*===================제출용 파일========================= */

    //random forest 가 가장 빠르기 때문에

    let transported_series = Series::new("Transported", y_hat.into_iter().collect::<Vec<i32>>())
        .cast(&DataType::Boolean)
        .unwrap();
    let passenger_id_series = submission_df.column("PassengerId").unwrap().clone();
    
    let  df: DataFrame = DataFrame::new(vec![passenger_id_series, transported_series]).unwrap();

    let  df = df
        .clone()
        .lazy()
        .with_columns([col("Transported").cast(DataType::String)])
        .collect()
        .unwrap();

    let mut df = df
        .clone()
        .lazy()
        .with_columns([when(col("Transported").eq(lit("true")))
            .then(lit("True"))
            .otherwise(lit("False"))
            .alias("Transported")])
        .collect()
        .unwrap();
    let mut output_file: File = File::create("./datasets/spaceship-titanic/out.csv").unwrap();

    CsvWriter::new(&mut output_file)
        // .has_header(true)
        .finish(&mut df)
        .unwrap();
}
