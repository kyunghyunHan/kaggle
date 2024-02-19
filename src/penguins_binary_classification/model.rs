use burn::data::dataset::{Dataset};
use burn::tensor::ElementConversion;
use burn::{
    backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
    config::Config,
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::Module,
    nn::{
        loss::CrossEntropyLoss,
        Dropout,
        DropoutConfig,
        Linear,
        LinearConfig,
        ReLU,
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Data, Int, Tensor,
    },
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
const WIDTH: usize = 8;
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PenguinsData {
    pub train_datas: [f64; WIDTH],
    pub species: i32,
}
#[derive(Debug)]
pub struct PenguinsDataset {
    dataset: Vec<PenguinsData>,
}

impl PenguinsDataset {
    pub fn new() -> Self {
        let train_df = CsvReader::from_path(
            "./datasets/penguins_binary_classification/penguins_binary_classification.csv",
        )
        .unwrap()
        .finish()
        .unwrap();
        let train_df = train_df
            .clone()
            .lazy()
            .with_columns([when(col("species").eq(lit("Adelie")))
                .then(lit(1))
                .otherwise(lit(0))
                .alias("species")])
            .collect()
            .unwrap();
        let labels: Vec<i32> = train_df
            .column("species")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<i32>>();
        let island_dummies = train_df
            .select(["island"])
            .unwrap()
            .to_dummies(None, false)
            .unwrap();
        let train_df = train_df
            .hstack(island_dummies.get_columns())
            .unwrap()
            .drop("island")
            .unwrap();
        let x_train = train_df
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let mut x_train_vec: [[f64; WIDTH]; 274] = [[0.0; WIDTH]; 274];
        for (index, row) in x_train.outer_iter().enumerate() {
            let row_vec: [f64; WIDTH] = row
                .into_iter()
                .take(WIDTH)
                .cloned()
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();
            x_train_vec[index] = row_vec; // 해당 인덱스에 row_vec를 할당합니다.
        }

        let mut penguins_data_set: Vec<PenguinsData> = Vec::new();
        for i in 0..labels.len() {
            penguins_data_set.push(PenguinsData {
                train_datas: x_train_vec[i],
                species: labels[i],
            })
        }

        PenguinsDataset {
            dataset: penguins_data_set,
        }
    }
}
#[derive(Module, Debug)] //딥러닝 모듈생성

struct Model<B: Backend> {
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, train_datas: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, width] = train_datas.dims();
        let x = train_datas.reshape([batch_size, width]);

        let x = self.dropout.forward(x);

        let x = self.activation.forward(x);

        let x = x.reshape([batch_size, 8]);

        let x = self.linear1.forward(x);

        let x = self.dropout.forward(x);

        let x = self.activation.forward(x);

        let output = self.linear2.forward(x); // [batch_size, num_classes]

        output
    }

    pub fn forward_classification(
        &self,
        train_datas: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(train_datas);
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}
//network의 기본 구성
//구성을 직렬화하여 모델 hyperprameter를 쉽게 저장
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize, //모델의 출력크기수
    hidden_size: usize, //==>은닉충의 크기
    #[config(default = "0.5")]
    dropout: f64, //dropout
}
impl ModelConfig {
    /// Returns the initialized model.
    fn init<B: Backend>(&self) -> Model<B> {
        Model {
            activation: ReLU::new(),
            linear1: LinearConfig::new(8, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

pub struct Tester<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Tester<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}
#[derive(Clone, Debug)]
pub struct Test<B: Backend> {
    pub train_datas: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}
impl Dataset<PenguinsData> for PenguinsDataset {
    fn get(&self, index: usize) -> Option<PenguinsData> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
impl<B: AutodiffBackend> TrainStep<Test<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Test<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.train_datas, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}
impl<B: Backend> ValidStep<Test<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Test<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.train_datas, batch.targets)
    }
}
impl<B: Backend> Batcher<PenguinsData, Test<B>> for Tester<B> {
    fn batch(&self, items: Vec<PenguinsData>) -> Test<B> {
        let train_datas = items
            .iter()
            .map(|item| Data::<f64, 1>::from(item.train_datas))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, WIDTH]))
            // .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.species as i64).elem()])))
            .collect();
        let train_datas = Tensor::cat(train_datas, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);
        Test {
            train_datas,
            targets,
        }
    }
}
//274개,
//
#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,//Adam사용
    #[config(default = 10)]
    pub num_epochs: usize,//epoch
    #[config(default = 32)]
    pub batch_size: usize,//배치사이즈
    #[config(default = 10)]
    pub num_workers: usize,//species
    #[config(default = 42)]
    pub seed: u64,//난수 생성
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);
    let batcher_train = Tester::<B>::new(device.clone());
    let batcher_valid = Tester::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PenguinsDataset::new());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(PenguinsDataset::new());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
pub fn main() {
    let config = ModelConfig::new(2, 1024);
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    train::<MyAutodiffBackend>(
        "./train",
        TrainingConfig::new(config, AdamConfig::new()),
        device,
    );
    println!("{:?}", PenguinsDataset::new())
}
