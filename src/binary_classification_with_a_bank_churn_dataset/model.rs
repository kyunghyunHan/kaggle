use plotters::prelude::*;
use polars::prelude::*;

pub fn main() {
    /*===========data============ */
    /*data
    결측치 없음

     */
    let train_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s4e1/train.csv")
        .unwrap()
        .finish()
        .unwrap()
        .drop_many(&["id"]);
    let test_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s4e1/test.csv")
        .unwrap()
        .finish()
        .unwrap()
        .drop_many(&["id"]);
    let submission_df: DataFrame =
        CsvReader::from_path("./datasets/playground-series-s4e1/sample_submission.csv")
            .unwrap()
            .finish()
            .unwrap();
    println!("결측치 확인:{}", train_df);
    println!("결측치 확인:{:?}", train_df.schema());

    /*============Surname=========== */
    let surname_data = train_df
        .group_by(&["Surname"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();

    println!("Surname::::{}", surname_data);

    /*============CreditScore=========== */

    let credit_score_data = train_df
        .group_by(&["CreditScore"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap()
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();

    println!("CreditScore::::{}", credit_score_data);
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
   let has_cr_card = train_df
   .group_by(&["HasCrCard"])
   .unwrap()
   .select(["Exited"])
   .mean()
   .unwrap()
   .sort(&["Exited_mean"], vec![false, true], false)
   .unwrap();

   println!("HasCrCard::::::{}", has_cr_card);


   /*============NumOfProducts=========== */
   let has_cr_card = train_df
   .group_by(&["NumOfProducts"])
   .unwrap()
   .select(["Exited"])
   .mean()
   .unwrap()
   .sort(&["Exited_mean"], vec![false, true], false)
   .unwrap();

   println!("NumOfProducts::::::{}", has_cr_card);

    /*============Balance=========== */
    let has_cr_card = train_df
    .group_by(&["Balance"])
    .unwrap()
    .select(["Exited"])
    .mean()
    .unwrap()
    .sort(&["Exited_mean"], vec![false, true], false)
    .unwrap();
 
    println!("Balance::::::{}", has_cr_card);


    /*============Tenure=========== */
    let has_cr_card = train_df
    .group_by(&["Tenure"])
    .unwrap()
    .select(["Exited"])
    .mean()
    .unwrap()
    .sort(&["Exited_mean"], vec![false, true], false)
    .unwrap();
 
    println!("Tenure::::::{}", has_cr_card);


    /*============Age=========== */
    let has_cr_card = train_df
    .group_by(&["Age"])
    .unwrap()
    .select(["Exited"])
    .mean()
    .unwrap()
    .sort(&["Exited_mean"], vec![false, true], false)
    .unwrap();
 
    println!("Age::::::{}", has_cr_card);
}
