pub mod model {

    use std::any::TypeId;

    use candle_core::{DType, Device, Result as CRS, Tensor};
    use polars::prelude::cov::{cov, pearson_corr as pc};
    use polars::prelude::*;
    use polars_lazy::dsl::{col, pearson_corr};
    #[derive(Debug)]
    struct Dataset {
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        y_test: Tensor,
    }
    impl Dataset {
        fn new() -> CRS<Dataset> {
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

            let train_samples = train_df.shape().0;

            println!("{}", train_samples);
            println!("데이터 미리보기:{}", train_df.head(None));
            println!("데이터 정보 확인:{:?}", train_df.schema());
            println!("null확인:{:?}", train_df.null_count());
            println!("null확인:{:?}", test_df.null_count());
            println!("null확인:{:?}", submission_df.null_count());

            //test와 train의 id 삭제
            let mut corr_vec =
                corr_fn(&train_df, train_df.schema(), "smoking", TypeId::of::<i64>());
            corr_vec.retain(|x| x != "smoking");
            println!(
                "{:?}",
                corr_fn(&train_df, train_df.schema(), "smoking", TypeId::of::<i64>())
            );
            let labels = train_df
                .column("smoking")
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<f64>>();
            let test_labels = submission_df
                .column("smoking")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<f64>>();

            let train_df = train_df.select(&corr_vec).unwrap();
            let test_df = test_df.select(&corr_vec).unwrap();

            let x_test = test_df
                .to_ndarray::<Float64Type>(IndexOrder::Fortran)
                .unwrap();
            let mut test_buffer_images: Vec<u32> =
                Vec::with_capacity(test_df.shape().0 * test_df.shape().1);
            for i in x_test {
                test_buffer_images.push(i as u32)
            }
            let test_datas = Tensor::from_vec(
                test_buffer_images,
                (test_df.shape().0, test_df.shape().1),
                &Device::Cpu,
            )?
            .to_dtype(DType::F32)?;

            let train_df = train_df.select(&corr_vec).unwrap();

            let x_train = train_df
                .to_ndarray::<Float64Type>(IndexOrder::Fortran)
                .unwrap();
            let mut train_buffer_images: Vec<u32> =
                Vec::with_capacity(train_df.shape().0 * train_df.shape().1);
            for i in x_train {
                train_buffer_images.push(i as u32)
            }
            let train_datas = Tensor::from_vec(
                train_buffer_images,
                (train_df.shape().0, train_df.shape().1),
                &Device::Cpu,
            )?
            .to_dtype(DType::F32)?;
            let train_labels = Tensor::from_vec(labels, train_df.shape().0, &Device::Cpu)?;

            let test_labels = Tensor::from_vec(test_labels, (test_df.shape().0,), &Device::Cpu)?;
            Ok(Self {
                x_train: train_datas,
                y_train: train_labels,
                x_test: test_datas,
                y_test: test_labels,
            })
        }
    }
    /*0.2이상만 출력하는 함수 */

    fn corr_fn(train_df: &DataFrame, list: Schema, target: &str, tp: TypeId) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();

        for i in list {
            if tp == TypeId::of::<i64>() {
                if pc(
                    train_df.column(&i.0).unwrap().cast(&DataType::Int64).unwrap().i64().unwrap(),
                    train_df.column(target).unwrap().i64().unwrap(),
                    1,
                )
                .unwrap()
                    .abs() // 절댓값을 취합니다.
                    > 0.2
                {
                    v.push(i.0.to_string())
                }
            } else if tp == TypeId::of::<f64>() {
                if pc(
                    train_df.column(&i.0).unwrap().cast(&DataType::Float64).unwrap().f64().unwrap(),
                    train_df.column(target).unwrap().f64().unwrap(),
                    1,
                )
                .unwrap()
                    .abs() // 절댓값을 취합니다.
                    > 0.2
                {
                    v.push(i.0.to_string())
                }
            } else if tp == TypeId::of::<u64>() {
                if pc(
                    train_df.column(&i.0).unwrap().cast(&DataType::UInt64).unwrap().u64().unwrap(),
                    train_df.column(target).unwrap().u64().unwrap(),
                    1,
                )
                .unwrap()
                    .abs() // 절댓값을 취합니다.
                    > 0.2
                {
                    v.push(i.0.to_string())
                }
            }
        }
        v
    }

    pub fn main() -> anyhow::Result<()> {
       let m=  Dataset::new().unwrap();
       println!("{:?}",m);
        Ok(())
    }
}
