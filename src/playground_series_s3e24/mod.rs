pub mod model {

    use candle_core::{Result as CRS, Tensor};
    use polars::prelude::cov::{cov, pearson_corr as pc};
    use polars::prelude::*;
    use polars_lazy::dsl::{col, pearson_corr};
    struct Dataset {
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        y_test: Tensor,
    }
    impl Dataset {
        fn new() -> CRS<()> {
            let train_df = CsvReader::from_path("./datasets/playground-series-s3e24/train.csv")
                .unwrap()
                .finish()
                .unwrap();
            let test_df = CsvReader::from_path("./datasets/playground-series-s3e24/test.csv")
                .unwrap()
                .finish()
                .unwrap();
            let submission_df =
                CsvReader::from_path("./datasets/playground-series-s3e24/sample_submission.csv")
                    .unwrap()
                    .finish()
                    .unwrap();
            println!("데이터 미리보기:{}", train_df.head(None));
            println!("데이터 정보 확인:{:?}", train_df.schema());
            println!("null확인:{:?}", train_df.null_count());
            println!("null확인:{:?}", test_df.null_count());
            println!("null확인:{:?}", submission_df.null_count());

            //test와 train의 id 삭제

            //상관관계를 확인

            let age_corr = pc(
                train_df.column("age").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let height_cm_corr = pc(
                train_df.column("height(cm)").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let weight_kg_corr = pc(
                train_df.column("weight(kg)").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let waist_cm_corr = pc(
                train_df.column("waist(cm)").unwrap().f64().unwrap(),
                train_df
                    .column("smoking")
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap(),
                1,
            )
            .unwrap();
            let eyesight_left_corr = pc(
                train_df.column("eyesight(left)").unwrap().f64().unwrap(),
                train_df
                    .column("smoking")
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap(),
                1,
            )
            .unwrap();
            let eyesight_right_corr = pc(
                train_df.column("eyesight(right)").unwrap().f64().unwrap(),
                train_df
                    .column("smoking")
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap(),
                1,
            )
            .unwrap();
            let hearing_left_corr = pc(
                train_df.column("hearing(left)").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let hearing_right_corr = pc(
                train_df.column("hearing(right)").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let systolic_corr = pc(
                train_df.column("systolic").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let relaxation_corr = pc(
                train_df.column("relaxation").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let fasting_blood_sugar_corr = pc(
                train_df
                    .column("fasting blood sugar")
                    .unwrap()
                    .i64()
                    .unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let cholesterol_corr = pc(
                train_df.column("Cholesterol").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let triglyceride_corr = pc(
                train_df.column("triglyceride").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let hdl_corr = pc(
                train_df.column("HDL").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let ldl_corr = pc(
                train_df.column("LDL").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let hemoglobin_corr = pc(
                train_df.column("hemoglobin").unwrap().f64().unwrap(),
                train_df
                    .column("smoking")
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap(),
                1,
            )
            .unwrap();
            let urine_protein_corr = pc(
                train_df.column("Urine protein").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let serum_creatinine_corr = pc(
                train_df.column("serum creatinine").unwrap().f64().unwrap(),
                train_df
                    .column("smoking")
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap(),
                1,
            )
            .unwrap();
            let ast_corr = pc(
                train_df.column("AST").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let alt_corr = pc(
                train_df.column("ALT").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let gtp_corr = pc(
                train_df.column("Gtp").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let dental_caries_corr = pc(
                train_df.column("dental caries").unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();

            println!("{}", age_corr); //-0.20603295889077042
            println!("{}", height_cm_corr); //0.44711105381189986
            println!("{}", weight_kg_corr); //0.35174789218573643
            println!("{}", waist_cm_corr); //0.26271479402119186
            println!("{}", eyesight_left_corr); //0.1004195389620777
            println!("{}", eyesight_right_corr); //0.10978119610956215
            println!("{}", hearing_left_corr); //-0.03821888300087877
            println!("{}", hearing_right_corr); //-0.036857667071997204
            println!("{}", systolic_corr); //0.05864169399586579
            println!("{}", relaxation_corr); //0.10950100949547634
            println!("{}", fasting_blood_sugar_corr); //0.09653442873116959
            println!("{}", cholesterol_corr); //-0.05189605622500711
            println!("{}", triglyceride_corr); //0.33197510463307206
            println!("{}", hdl_corr); //-0.2711856476663979
            println!("{}", ldl_corr); //-0.07228538659871742
            println!("{}", hemoglobin_corr); //0.45067865642973715
            println!("{}", urine_protein_corr); //-0.028548290643707737
            println!("{}", serum_creatinine_corr); //0.2729786244236494
            println!("{}", ast_corr); //0.05939388433682304
            println!("{}", alt_corr); //0.1630160307946045
            println!("{}", gtp_corr); //0.30556103179051425
            println!("{}", dental_caries_corr); //0.10663624290129949

            Ok(())
        }
    }

    fn corr_fn(train_df: &DataFrame, list: Vec<&str>, target: &str, num: f64)->Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for i in 0..list.len() {
            if pc(
                train_df.column(list[i]).unwrap().i64().unwrap(),
                train_df.column("smoking").unwrap().i64().unwrap(),
                1,
            )
            .unwrap()
                > 0.2
            {
                v.push(list[i].to_string())
            }
        }
        v
    }
    pub fn main() -> anyhow::Result<()> {
        Dataset::new().unwrap();
        Ok(())
    }
}
