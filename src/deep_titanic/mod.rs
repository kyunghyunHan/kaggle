pub mod model {
    use candle_core::{DType, Device, Result, Tensor, D};
    use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
    use polars::prelude::cov::pearson_corr;
    use polars::prelude::*;
    const IMAGE_DIM: usize = 7; //2차원 벡터
    const RESULTS: usize = 2; //모델이; 예측하는 개수
    const EPOCHS: usize = 10; //에폭
    const LAYER1_OUT_SIZE: usize = 4; //첫번쨰 출력충의 출력뉴런 수
    const LAYER2_OUT_SIZE: usize = 2; //2번쨰 츨략층의  출력 뉴런 수
    const LEARNING_RATE: f64 = 0.05;

    struct model {}
    impl model {}
    #[derive(Clone, Debug)]
    struct Dataset {
        pub train_datas: Tensor, //train data
        pub train_labels: Tensor,
        pub test_datas: Tensor, //test data
        pub test_labels: Tensor,
    }
    struct MultiLevelPerceptron {
        ln1: Linear,
        ln2: Linear, //은닉충
        ln3: Linear, //출력충
    }
    //3개 => 2개의 은닉충 1개의 출력충
    impl MultiLevelPerceptron {
        fn new(vs: VarBuilder) -> candle_core::Result<Self> {
            let ln1 = candle_nn::linear(IMAGE_DIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
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
    impl Dataset {
        fn new() -> Result<Dataset> {
            let train_samples = 891;
            let test_samples = 418;
            //데이터 불러오기
            let train_df = CsvReader::from_path("./datasets/titanic/train.csv")
                .unwrap()
                .finish()
                .unwrap();
            let test_df = CsvReader::from_path("./datasets/titanic/test.csv")
                .unwrap()
                .finish()
                .unwrap();
            let submission_df = CsvReader::from_path("./datasets/titanic/gender_submission.csv")
                .unwrap()
                .finish()
                .unwrap();
            let train_lebels = train_df
                .column("Survived")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>();
            println!("{}", train_df.null_count());
            println!("{}", test_df.null_count());
            println!("{}", submission_df.null_count());

            //필요없는 PassegerId,Name,Ticket제거
            let train_df = train_df.drop_many(&["PassengerId", "Name", "Ticket"]);
            let test_df = test_df.drop_many(&["PassengerId", "Name", "Ticket"]);

            let train_df = train_df
                .clone()
                .lazy()
                .with_columns([
                    col("Age").fill_null((col("Age").mean())),
                    col("Embarked").fill_null(lit("S")),
                ])
                .collect()
                .unwrap()
                .drop_many(&["Cabin"]);
            let test_df: DataFrame = test_df
                .clone()
                .lazy()
                .with_columns([
                    col("Age").fill_null((col("Age").mean())),
                    col("Embarked").fill_null(lit("S")),
                    col("Fare").fill_null(lit(0)),
                ])
                .collect()
                .unwrap()
                .drop_many(&["Cabin"]);

            println!("null확인:{}", train_df.null_count());
            println!("null확인:{}", test_df.null_count());
            //범주형 먼저 처리
            /*Pclass */
            /*One hot encoding */
            //dummies만들기
            let pclass_train_dummies: DataFrame = train_df
                .select(["Pclass"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
            let pclass_test_dummies: DataFrame = test_df
                .select(["Pclass"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
            //join
            let train_df = train_df
                .hstack(pclass_train_dummies.get_columns())
                .unwrap()
                .drop(&"Pclass")
                .unwrap();
            let test_df = test_df
                .hstack(pclass_test_dummies.get_columns())
                .unwrap()
                .drop(&"Pclass")
                .unwrap();

            /*sex */
            let sex_train_dummies: DataFrame = train_df
                .select(["Sex"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
            let sex_test_dummies: DataFrame = test_df
                .select(["Sex"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
            let train_df: DataFrame = train_df
                .hstack(sex_train_dummies.get_columns())
                .unwrap()
                .drop(&"Sex")
                .unwrap();
            let test_df: DataFrame = test_df
                .hstack(sex_test_dummies.get_columns())
                .unwrap()
                .drop(&"Sex")
                .unwrap();
            /*Embarked */

            let embarked_train_dummies: DataFrame = train_df
                .select(["Embarked"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
            let embarked_test_dummies: DataFrame = test_df
                .select(["Embarked"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
            let train_df: DataFrame = train_df
                .hstack(embarked_train_dummies.get_columns())
                .unwrap()
                .drop(&"Embarked")
                .unwrap();
            let test_df: DataFrame = test_df
                .hstack(embarked_test_dummies.get_columns())
                .unwrap()
                .drop(&"Embarked")
                .unwrap();
            println!("{}", train_df.tail(Some(3)));
            println!("{:?}", train_df.schema());
            /*=== 상관 관계 확인 ==== */
            let age_corr = pearson_corr(
                train_df.column("Age").unwrap().f64().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap(),
                1,
            )
            .unwrap();

            let sibsp_corr = pearson_corr(
                train_df.column("SibSp").unwrap().i64().unwrap(),
                train_df.column("Survived").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();

            let parch_corr = pearson_corr(
                train_df.column("Parch").unwrap().i64().unwrap(),
                train_df.column("Survived").unwrap().i64().unwrap(),
                1,
            )
            .unwrap();
            let fare_corr = pearson_corr(
                train_df.column("Fare").unwrap().f64().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap(),
                1,
            )
            .unwrap();
            let pclass_1_corr = pearson_corr(
                train_df.column("Pclass_1").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();
            let pclass_2_corr = pearson_corr(
                train_df.column("Pclass_2").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();
            let pclass_3_corr = pearson_corr(
                train_df.column("Pclass_3").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();

            let sex_female_corr = pearson_corr(
                train_df.column("Sex_female").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();
            let sex_male_corr = pearson_corr(
                train_df.column("Sex_male").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();
            let embarked_c_corr = pearson_corr(
                train_df.column("Embarked_C").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();
            let embarked_q_corr = pearson_corr(
                train_df.column("Embarked_Q").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();
            let embarked_s_corr = pearson_corr(
                train_df.column("Embarked_S").unwrap().i32().unwrap(),
                train_df
                    .column("Survived")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .cast(&DataType::Int32)
                    .unwrap()
                    .i32()
                    .unwrap(),
                1,
            )
            .unwrap();

            println!("age_corr:{}", age_corr);
            println!("sibsp_corr:{}", sibsp_corr);
            println!("parch_corr:{}", parch_corr);
            println!("fare_corr:{}", fare_corr);
            println!("pclass_1_corr:{}", pclass_1_corr);
            println!("pclass_2_corr:{}", pclass_2_corr);
            println!("pclass_3_corr:{}", pclass_3_corr);
            println!("sex_female_corr:{}", sex_female_corr);
            println!("sex_male_corr:{}", sex_male_corr);
            println!("embarked_c_corr:{}", embarked_c_corr);
            println!("embarked_q_corr:{}", embarked_q_corr);
            println!("embarked_s_corr:{}", embarked_s_corr);

            //Tensor로 변경
            //train label
            //train tensor
            //test label
            //testtensor
            let train_df = train_df
                .select([
                    "Pclass_1",
                    "Pclass_3",
                    "Fare",
                    "Sex_female",
                    "Sex_male",
                    "Embarked_C",
                    "Embarked_S",
                    "Survived",
                ])
                .unwrap();
            let test_df: DataFrame = test_df
                .select([
                    "Pclass_1",
                    "Pclass_3",
                    "Fare",
                    "Sex_female",
                    "Sex_male",
                    "Embarked_C",
                    "Embarked_S",
                ])
                .unwrap();

            let labels = train_df
                .column("Survived")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>();
            let test_labels = submission_df
                .column("Survived")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>();

            let train_labels = Tensor::from_vec(labels, (train_samples,), &Device::Cpu)?;
            let test_labels = Tensor::from_vec(test_labels, (test_samples,), &Device::Cpu)?;
            let x_test = test_df
                .to_ndarray::<Int64Type>(IndexOrder::Fortran)
                .unwrap();
            let mut test_buffer_images: Vec<u32> = Vec::with_capacity(test_samples * 7);
            for i in x_test {
                test_buffer_images.push(i as u32)
            }
            let test_datas =
                (Tensor::from_vec(test_buffer_images, (test_samples, 7), &Device::Cpu)?
                    .to_dtype(DType::F32)?
                    / 255.)?;

            let x_train = train_df
                .drop("Survived")
                .unwrap()
                .to_ndarray::<Int64Type>(IndexOrder::Fortran)
                .unwrap();
            let mut train_buffer_images: Vec<u32> = Vec::with_capacity(train_samples * 7);
            for i in x_train {
                train_buffer_images.push(i as u32)
            }
            let train_datas =
                (Tensor::from_vec(train_buffer_images, (train_samples, 7), &Device::Cpu)?
                    .to_dtype(DType::F32)?
                    / 255.)?;

            Ok(Self {
                train_datas: train_datas,
                train_labels: train_labels,
                test_datas: test_datas,
                test_labels: test_labels,
            })
        }
    }
    fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
        let train_results = m.train_labels.to_device(dev)?; //디바이스
        let train_votes = m.train_datas.to_device(dev)?;
        let varmap = VarMap::new(); //VarMap은 변수들을 관리하는 데 사용되는 자료 구조
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let model = MultiLevelPerceptron::new(vs.clone())?;
        let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
        let test_votes = m.test_datas.to_device(dev)?;
        let test_results = m.test_labels.to_device(dev)?;
        let mut final_accuracy: f32 = 0.;
        for epoch in 1..EPOCHS + 1 {
            let logits = model.forward(&train_votes)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?; //Minus1:가장 마지막 축
            let loss = loss::nll(&log_sm, &train_results)?; //손실함수
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
            if final_accuracy == 100.0 {
                break;
            }
        }
        if final_accuracy < 100.0 {
            Err(anyhow::Error::msg("The model is not trained well enough."))
        } else {
            Ok(model)
        }
    }
    #[tokio::main]
    pub async fn main() -> anyhow::Result<()> {
        let dev = Device::cuda_if_available(0)?;
        let m = Dataset::new()?;
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
        let real_world_votes: Vec<u32> = vec![13, 22];

        let tensor_test_votes = Tensor::from_vec(real_world_votes.clone(), (1, IMAGE_DIM), &dev)?
            .to_dtype(DType::F32)?;

        let final_result = trained_model.forward(&tensor_test_votes)?;

        let result = final_result
            .argmax(D::Minus1)?
            .to_dtype(DType::F32)?
            .get(0)
            .map(|x| x.to_scalar::<f32>())??;
        println!("real_life_votes: {:?}", real_world_votes);
        println!("neural_network_prediction_result: {:?}", result);
        Ok(())
    }
}
