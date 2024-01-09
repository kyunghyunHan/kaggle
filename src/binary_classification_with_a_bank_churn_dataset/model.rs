use plotters::prelude::*;
use polars::prelude::*;

pub fn main() {
    /*======================= */
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
    /*======================= */
    /*확인 */

    let credit_score_data = train_df
        .group_by(&["CreditScore"])
        .unwrap()
        .select(["Exited"])
        .mean()
        .unwrap();
    let credit_score_data = credit_score_data
        .sort(&["Exited_mean"], vec![false, true], false)
        .unwrap();

    println!("{}", credit_score_data);
    let cryosleep_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Exited").eq(true))
        .collect()
        .unwrap()
        .column("CreditScore")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    println!("{}", cryosleep_arrived);

    let cryosleep_not_arrived = train_df
        .clone()
        .lazy()
        .filter(col("Exited").eq(false))
        .collect()
        .unwrap()
        .column("CreditScore")
        .unwrap()
        .value_counts(true, false)
        .unwrap();
    let cryosleep_arrived_vec: Vec<u32> = cryosleep_arrived
        .column("count")
        .unwrap()
        .u32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let cryosleep_not_arrived_vec: Vec<u32> = cryosleep_not_arrived
        .column("count")
        .unwrap()
        .u32()
        .unwrap()
        .into_no_null_iter()
        .collect();

    let data1: [(u32, u32); 2] = [(0, cryosleep_arrived_vec[1]), (1, cryosleep_arrived_vec[0])];
    let data2: [(u32, u32); 2] = [
        (3, cryosleep_not_arrived_vec[0]),
        (4, cryosleep_not_arrived_vec[1]),
    ];
    let credit_score = train_df.select(["CreditScore", "Exited"]).unwrap();
    let credit_score = credit_score
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap();

    let mut credit_score_vec: Vec<Vec<_>> = Vec::new();
    for row in credit_score.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        credit_score_vec.push(row_vec);
    }
    //350 850
    println!(
        "{:?}",
        train_df.column("CreditScore").unwrap().min_as_series()
    );

    let credit_score_root = BitMapBackend::new(
        "./src/binary_classification_with_a_bank_churn_dataset/credit_score.png",
        (800, 600),
    )
    .into_drawing_area();
    credit_score_root.fill(&WHITE).unwrap();
    let mut chart_builder = ChartBuilder::on(&credit_score_root);
    chart_builder
        .margin(20)
        .set_left_and_bottom_label_area_size(20);
    let mut chart_context = chart_builder
        .build_cartesian_2d(
            (0u32..6u32).step(1).into_segmented(),
            (0f64..3000f64).step(1f64),
        )
        .unwrap();
    chart_context.configure_mesh().draw().unwrap();

    chart_context
        .draw_series(
            Histogram::vertical(&chart_context)
                .style(BLUE.filled())
                .data(data1.iter().map(|v| (v.0, v.1 as f64))),
        )
        .unwrap();

    chart_context
        .draw_series(
            Histogram::vertical(&chart_context)
                .style(RED.filled())
                .data(data2.iter().map(|v| (v.0, v.1 as f64))),
        )
        .unwrap();
}
/*==================*/
