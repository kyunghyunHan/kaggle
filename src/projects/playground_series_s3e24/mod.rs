use candle_core::{DType, Device, IndexOp, Result as CRS, Tensor, D};
use candle_nn::{loss, ops, Dropout, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};
use polars::prelude::cov::{cov, pearson_corr as pc};
use polars::prelude::*;
use std::any::TypeId;
use std::fs::File;

const DATADIM: usize = 9; //2차원 벡터
const RESULTS: usize = 1; //모델이; 예측하는 개수
const EPOCHS: usize = 10000; //에폭
const LEARNING_RATE: f64 = 1e-5;
#[derive(Debug, Clone)]
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
        let train_labels =
            Tensor::from_vec(labels, train_df.shape().0, &Device::Cpu)?.to_dtype(DType::F32)?;

        let test_labels = Tensor::from_vec(test_labels, (test_df.shape().0,), &Device::Cpu)?
            .to_dtype(DType::F32)?;
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
}

//3개 => 2개의 은닉충 1개의 출력충
impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let ln1 = candle_nn::linear(DATADIM, RESULTS, vs.pp("ln1"))?;
        Ok(Self { ln1 })
    }
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.ln1.forward(xs)
    }
}

fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
    let train_results = m.y_train.to_device(dev)?.unsqueeze(1)?;

    let train_votes = m.x_train.to_device(dev)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    let mut final_loss: f32 = 0.;
    for epoch in 0..EPOCHS + 1 {
        
        let logits = model.forward(&train_votes)?;

        let loss = loss::mse(&logits, &train_results)?; // 평균 제곱 오차 손실 함수
        sgd.backward_step(&loss)?;

        final_loss = loss.to_scalar::<f32>()?;
        println!("Epoch: {epoch:3} Train loss: {:8.5}", final_loss);
        if final_loss < 0.001 {
            // 손실이 일정 수준 이하로 감소하면 학습 종료
            break;
        }
    }

    Ok(model)
}
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
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
        let final_result = trained_model.forward(&tensor_test_votes)?.i((0,0))?.to_scalar::<f32>()?;
        result_vec.push(final_result);
    }
    //random forest 가 가장 빠르기 때문에
    let survived_series = Series::new("smoking", result_vec);
    let passenger_id_series = Series::new("id", (159256..=265426).collect::<Vec<i64>>());

    let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
    let mut output_file: File =
        File::create("./datasets/playground-series-s3e24/out.csv").unwrap();
    CsvWriter::new(&mut output_file).finish(&mut df).unwrap();
    Ok(())
}