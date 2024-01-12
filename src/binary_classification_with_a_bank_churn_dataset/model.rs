use plotters::prelude::*;
use polars::prelude::*;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::*;
pub fn main() {
    /*===========data============ */
    let train_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s4e1/train.csv")
        .unwrap()
        .finish()
        .unwrap()
        .drop_many(&["id", "CustomerId", "Surname"]);
    let test_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s4e1/test.csv")
        .unwrap()
        .finish()
        .unwrap()
        .drop_many(&["id", "CustomerId", "Surname"]);
    let submission_df: DataFrame =
        CsvReader::from_path("./datasets/playground-series-s4e1/sample_submission.csv")
            .unwrap()
            .finish()
            .unwrap();
    println!("{}", train_df);
    println!("{:?}", train_df.schema());

    /*============결측치 확인=========== */
    println!("{:?}", train_df.null_count());
    println!("{:?}", test_df.null_count());
    /*============CreditScore=========== */

    /*============Geography=========== */
    let geography_data = train_df
        .group_by(&["Geography"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();
    println!("Geography:::::{}", geography_data);
    let geography_train_dummies: DataFrame = train_df
        .select(["Geography"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();
    let geography_test_dummies: DataFrame = test_df
        .select(["Geography"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let train_df: DataFrame = train_df
        .hstack(geography_train_dummies.get_columns())
        .unwrap()
        .drop_many(&["Geography_France", "Geography_Spain", "Geography"]);

    let test_df: DataFrame = test_df
        .hstack(geography_test_dummies.get_columns())
        .unwrap()
        .drop_many(&["Geography_France", "Geography_Spain", "Geography"]);
    println!("Geography:::::{}", train_df);

    /*============Gender=========== */
    let gender_data = train_df
        .group_by(&["Gender"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();
    println!("Gender::::::{}", gender_data);
    let gender_train_dummies: DataFrame = train_df
        .select(["Gender"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let gender_test_dummies: DataFrame = test_df
        .select(["Gender"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();
    let train_df: DataFrame = train_df
        .hstack(gender_train_dummies.get_columns())
        .unwrap()
        .drop("Gender")
        .unwrap();
    let test_df: DataFrame = test_df
        .hstack(gender_test_dummies.get_columns())
        .unwrap()
        .drop("Gender")
        .unwrap();
    /*============IsActiveMember=========== */
    let is_active_member_data = train_df
        .group_by(&["IsActiveMember"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();

    println!("IsActiveMember::::::{}", is_active_member_data);
    let is_active_member_train_dummies: DataFrame = train_df
        .select(["IsActiveMember"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let is_active_member_test_dummies: DataFrame = test_df
        .select(["IsActiveMember"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let train_df: DataFrame = train_df
        .hstack(is_active_member_train_dummies.get_columns())
        .unwrap()
        .drop("IsActiveMember")
        .unwrap();
    let test_df: DataFrame = test_df
        .hstack(is_active_member_test_dummies.get_columns())
        .unwrap()
        .drop("IsActiveMember")
        .unwrap();
    /*============HasCrCard=========== */
    let has_cr_card_data = train_df
        .group_by(&["HasCrCard"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();
    println!("HasCrCard::::::{}", has_cr_card_data);
    let has_cr_card_train_dummies: DataFrame = train_df
        .select(["HasCrCard"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let has_cr_card_test_dummies: DataFrame = test_df
        .select(["HasCrCard"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();

    let train_df: DataFrame = train_df
        .hstack(has_cr_card_train_dummies.get_columns())
        .unwrap()
        .drop("HasCrCard")
        .unwrap();
    let test_df: DataFrame = test_df
        .hstack(has_cr_card_test_dummies.get_columns())
        .unwrap()
        .drop("HasCrCard")
        .unwrap();
    /*============NumOfProducts=========== */
    let num_of_products_data = train_df
        .group_by(&["NumOfProducts"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();

    println!("NumOfProducts::::::{}", num_of_products_data);
    let num_of_products_train_dummies: DataFrame = train_df
        .select(["NumOfProducts"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();
    let num_of_products_test_dummies: DataFrame = test_df
        .select(["NumOfProducts"])
        .unwrap()
        .to_dummies(None, false)
        .unwrap();
    println!("{}", num_of_products_train_dummies);

    let train_df: DataFrame = train_df
        .hstack(num_of_products_train_dummies.get_columns())
        .unwrap()
        .drop("NumOfProducts_2")
        .unwrap();
    let test_df: DataFrame = test_df
        .hstack(num_of_products_test_dummies.get_columns())
        .unwrap()
        .drop("NumOfProducts_2")
        .unwrap();
    /*============Balance=========== */
    let balance_data = train_df
        .group_by(&["Balance"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();

    println!("Balance::::::{}", balance_data);
    /*============processing=========== */
    println!("Balance::::::{}", train_df);

    /*=====model====== */
    let y_train = train_df.column("Exited").unwrap();
    let y_train: Vec<i32> = y_train
        .i64()
        .unwrap()
        .into_no_null_iter()
        .map(|x| x as i32)
        .collect();
    let x_train = train_df.drop("Exited").unwrap();
    let x_train = x_train
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap();
    let mut x_train_vec: Vec<Vec<f64>> = Vec::new();
    for row in x_train.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        x_train_vec.push(row_vec);
    }
    let x_train: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_train_vec);

    let x_test = test_df
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap();
    let mut x_test_vec: Vec<Vec<f64>> = Vec::new();
    for row in x_test.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        x_test_vec.push(row_vec);
    }
    let x_test: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_test_vec);
    let y_test: Vec<i32> = submission_df
        .column("Exited")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|x| x as i32)
        .collect();
    /*랜덤포레스트 */
    println!("{:?}",train_df.schema());
    let random_forest = RandomForestClassifier::fit(
        &x_train,
        &y_train,
        RandomForestClassifierParameters::default().with_n_trees(10),
    )
    .unwrap();
    let y_pred: Vec<i32> = random_forest.predict(&x_test).unwrap();
    let acc: f64 = ClassificationMetricsOrd::accuracy().get_score(&y_test, &y_pred);
    println!("{:?}", acc);
}
