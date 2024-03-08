# 타이타닉

## Dataset
 - [Dataset](https://www.kaggle.com/competitions/titanic/data)

## 데이터 불러오기
- 구조체 정의
```rs
  struct Dataset {
        pub train_datas: Tensor, //train data
        pub train_labels: Tensor,
        pub test_datas: Tensor, //test data
        pub test_labels: Tensor,
    }
```
- dataset불러오기
- finish: 작업을 완료하고 최종 결과를 반환
```rs
   let train_df = CsvReader::from_path("./datasets/titanic/train.csv")
                .unwrap()
                .finish()
                .unwrap();

```
## 전처리
- null count 확인
```rs
println!("{}", train_df.null_count());
```
```
shape: (1, 12)
┌─────────────┬──────────┬────────┬──────┬───┬────────┬──────┬───────┬──────────┐
│ PassengerId ┆ Survived ┆ Pclass ┆ Name ┆ … ┆ Ticket ┆ Fare ┆ Cabin ┆ Embarked │
│ ---         ┆ ---      ┆ ---    ┆ ---  ┆   ┆ ---    ┆ ---  ┆ ---   ┆ ---      │
│ u32         ┆ u32      ┆ u32    ┆ u32  ┆   ┆ u32    ┆ u32  ┆ u32   ┆ u32      │
╞═════════════╪══════════╪════════╪══════╪═══╪════════╪══════╪═══════╪══════════╡
│ 0           ┆ 0        ┆ 0      ┆ 0    ┆ … ┆ 0      ┆ 0    ┆ 687   ┆ 2        │
└────────────┴──────────┴────────┴──────┴───┴────────┴──────┴───────┴──────────┘
```
- cabin과 Embarked 가 null값이 있는것으로 판단
- 필요 없는 데이터들 제거
```rs
  //필요없는 PassegerId,Name,Ticket제거
let train_df = train_df.drop_many(&["PassengerId", "Name", "Ticket"]);
let test_df = test_df.drop_many(&["PassengerId", "Name", "Ticket"]);
```
- null 값을 채우기 위해 AGE는 평균값으로 채워준다
- Embarked는 "S"로 채워준다
```rs
 let train_df = train_df
                .clone()
                .lazy()
                .with_columns([
                    col("Age").fill_null((col("Age").mean())),
                    col("Embarked").fill_null(lit("S")),
                ])
                .collect()
                .unwrap()
                .drop(&"Cabin").unwrap();
```
- 1
```rs
null확인:shape: (1, 7)
┌────────┬─────┬─────┬───────┬───────┬──────┬──────────┐
│ Pclass ┆ Sex ┆ Age ┆ SibSp ┆ Parch ┆ Fare ┆ Embarked │
│ ---    ┆ --- ┆ --- ┆ ---   ┆ ---   ┆ ---  ┆ ---      │
│ u32    ┆ u32 ┆ u32 ┆ u32   ┆ u32   ┆ u32  ┆ u32      │
╞════════╪═════╪═════╪═══════╪═══════╪══════╪══════════╡
│ 0      ┆ 0   ┆ 0   ┆ 0     ┆ 0     ┆ 0    ┆ 0        │
└────────┴─────┴─────┴───────┴───────┴──────┴──────────┘
```
- One hot encoding
- Pclass
```rs
 let pclass_train_dummies: DataFrame = train_df
                .select(["Pclass"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
```
```rs
       let train_df = train_df
                .hstack(pclass_train_dummies.get_columns())
                .unwrap()
                .drop(&"Pclass")
                .unwrap();
```
- sex
- Embarked
- 상관관계확인
```rs
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
```
```
age_corr:-0.06980851528714281
sibsp_corr:-0.03532249888573559
parch_corr:0.08162940708348347
fare_corr:0.2573065223849623
pclass_1_corr:0.2859037677837427
pclass_2_corr:0.09334857241192886
pclass_3_corr:-0.3223083573729699
sex_female_corr:0.5433513806577552
sex_male_corr:-0.5433513806577551
embarked_c_corr:0.16824043121823315
embarked_q_corr:0.0036503826839721647
embarked_s_corr:-0.14968272327068557
```
- Tensor로 변환
- Tensor 구조체
```rs
Ok(Self {
                train_datas: train_datas,
                train_labels: train_labels,
                test_datas: test_datas,
                test_labels: test_labels,
            })
```
## Layer
- Layer구성
- 2개의 은닉충과 출력충
```rs
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
```
## 학습
- 데이터셋을 받아 학습
```rs
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
```
```rs
  let dev = Device::cuda_if_available(0)?;
        let m = Dataset::new()?;
        println!("{:?}", m.train_datas.shape());
        println!("{:?}", m.test_datas.shape());
        println!("{:?}", m.train_labels.shape());
        println!("{:?}", m.test_labels.shape());
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
```
## Infer

```rs
  let test = m.clone().test_datas.to_vec2::<f32>().unwrap();
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
```


## CSV 저장하기
 
```rs
 //random forest 가 가장 빠르기 때문에
        let survived_series = Series::new(
            "Survived",
            result_vec,
        );
        let passenger_id_series = Series::new("PassengerId", (892..1310).collect::<Vec<i64>>());

        let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
        let mut output_file: File = File::create("./datasets/titanic/out.csv").unwrap();
        CsvWriter::new(&mut output_file).finish(&mut df).unwrap();
        Ok(())
```