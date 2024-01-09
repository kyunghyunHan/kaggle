use polars::prelude::*;
use plotters::prelude::*;

pub fn main(){
    /*======================= */
    /*data
    결측치 없음
    
     */
    let train_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s4e1/train.csv").unwrap()
    .finish().unwrap();
    let test_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s4e1/test.csv").unwrap()
    .finish().unwrap();
    let submission_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s4e1/sample_submission.csv").unwrap()
     .finish().unwrap();
     println!("결측치 확인:{}",train_df);
    /*======================= */
    /*확인 */
    let credit_score=  train_df.select(["CreditScore","Exited"]).unwrap();
    let credit_score = credit_score.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    let mut credit_score_vec: Vec<Vec<_>> = Vec::new();
    for row in credit_score.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        credit_score_vec.push(row_vec);
    }
    //350 850
    println!("{:?}",train_df.column("CreditScore").unwrap().min_as_series());
       
    let credit_score_root = BitMapBackend::new("./src/binary_classification_with_a_bank_churn_dataset/credit_score.png", (800, 600)).into_drawing_area();
    credit_score_root.fill(&WHITE).unwrap();
 
 
    let mut chart: ChartContext<'_, BitMapBackend<'_>, Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>> = ChartBuilder::on(&credit_score_root)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(350f64..900f64, 0f64..1f64)
        .unwrap();

    chart
        .configure_mesh()
        .draw()
        .unwrap();

    chart
        .draw_series(
            credit_score_vec
                .iter()
                .map(|x| {
                    Circle::new((x[0]as f64, x[1]as f64), 5, Into::<ShapeStyle>::into(RED))
                }),
        )
        .unwrap();


    /*======================= */

}