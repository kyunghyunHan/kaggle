use plotters::prelude::*;
use polars::prelude::*;

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
}
