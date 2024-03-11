pub mod model {

    use std::any::TypeId;

    use candle_core::{DType, Device, Result as CRS, Tensor,D};
    use candle_nn::{loss, ops, Dropout, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};

    use polars::prelude::cov::{cov, pearson_corr as pc};
    use polars::prelude::*;
    use polars_lazy::dsl::{col, pearson_corr};
    use std::fs::File;
    const DATADIM: usize = 9; //2차원 벡터
    const RESULTS: usize = 0; //모델이; 예측하는 개수
    const EPOCHS: usize = 500; //에폭
    const LAYER1_OUT_SIZE: usize = 512; //첫번쨰 출력충의 출력뉴런 수
    const LAYER2_OUT_SIZE: usize = 256; //2번쨰 츨략층의  출력 뉴런 수
    const LEARNING_RATE: f64 = 0.001;
    #[derive(Debug,Clone)]
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
            let train_labels = Tensor::from_vec(labels, train_df.shape().0, &Device::Cpu)?.to_dtype(DType::F32)?;

            let test_labels = Tensor::from_vec(test_labels, (test_df.shape().0,), &Device::Cpu)?.to_dtype(DType::F32)?;
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
    struct MultiLevelPerceptron {
        ln1: Linear,
        ln2: Linear,
        ln3: Linear,
    }

    //3개 => 2개의 은닉충 1개의 출력충
    impl MultiLevelPerceptron {
        fn new(vs: VarBuilder) -> candle_core::Result<Self> {
            let ln1 = candle_nn::linear(DATADIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
            let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
            let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, RESULTS + 1, vs.pp("ln3"))?;

            Ok(Self { ln1, ln2, ln3 })
        }
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
            let xs = self.ln1.forward(xs)?;
            let xs = xs.relu()?;
            let xs = self.ln2.forward(&xs)?;
            let xs = xs.relu()?;
            self.ln3.forward(&xs)
        }
    }

    fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
        let train_results = m.y_train.to_device(dev)?.unsqueeze(1)?;
        println!("{:?}",train_results.shape());
        let train_votes = m.x_train.to_device(dev)?;
        let varmap = VarMap::new(); //VarMap은 변수들을 관리하는 데 사용되는 자료 구조
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let model = MultiLevelPerceptron::new(vs.clone())?;
        let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
        let test_votes = m.x_test.to_device(dev)?;
        let test_results = m.y_test.to_device(dev)?;
        let mut final_accuracy: f32 = 0.;
        for epoch in 1..EPOCHS + 1 {
            let logits = model.forward(&train_votes)?;
            let log_sm = ops::sigmoid(&logits)?; //Minus1:가장 마지막 축
            println!("{:?}",log_sm.shape());
            let loss = loss::binary_cross_entropy_with_logit(&log_sm, &train_results)?; //손실함수
            sgd.backward_step(&loss)?; //역전파
            let test_logits = model.forward(&test_votes)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_results)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?; //정확도 계산
            let test_accuracy = sum_ok / test_results.dims1()? as f32;
            final_accuracy = 100. * test_accuracy;
            println!(
                "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                loss.to_scalar::<f32>()?,
                final_accuracy
            );
            if final_accuracy > 90.0 {
                break;
            }
        }
        if final_accuracy < 89.0 {
            Err(anyhow::Error::msg("The model is not trained well enough."))
        } else {
            Ok(model)
        }
    }
    #[tokio::main]
    pub  async  fn main() -> anyhow::Result<()> {
        let dev = Device::cuda_if_available(0)?;
        let m = Dataset::new()?;
        println!("{:?}", m);
        println!("{:?}", m.x_train.shape());
        println!("{:?}", m.y_train.shape());
        println!("{:?}", m.x_test.shape());
        println!("{:?}", m.y_test.shape());
        let trained_model: MultiLevelPerceptron;

        loop {
            println!("Trying to train neural network.");
            match train(m.clone(), &dev) {
                Ok(model) => {
                    trained_model = model;
                    break;
                }
                Err(e) => {
                    println!("Error: {}", e);
                    continue;
                }
            }
        }
        //추정
        let test = m.clone().x_test.to_vec2::<f32>().unwrap();
        let mut result_vec = Vec::new();
        for i in 0..test.len() {
            let tensor_test_votes =
                Tensor::from_vec(test[i].clone(), (1, DATADIM), &dev)?.to_dtype(DType::F32)?;

            let final_result = trained_model.forward(&tensor_test_votes)?;

            let result = final_result
                .argmax(D::Minus1)?
                .to_dtype(DType::F32)?
                .get(0)
                .map(|x| x.to_scalar::<f32>())??;
            println!("real_life_votes: {:?}", test[i].clone());
            println!("neural_network_prediction_result: {:?}", result);
            result_vec.push(result as i64);
        }
        //random forest 가 가장 빠르기 때문에
        let survived_series = Series::new(
            "Survived",
            result_vec,
        );
        let passenger_id_series = Series::new("PassengerId", (159256..265426).collect::<Vec<i64>>());

        let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
        let mut output_file: File = File::create("./datasets/titanic/out.csv").unwrap();
        CsvWriter::new(&mut output_file).finish(&mut df).unwrap();
        Ok(())
    }
}
