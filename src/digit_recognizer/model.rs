use burn::{
    backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
    config::Config,
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig}, //기본convolution
        loss::CrossEntropyLoss,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
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
use rayon::prelude::*;
use std::fs::File;

extern crate serde;

use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::data::dataset::SqliteDataset;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::record::Recorder;
use burn::tensor::ElementConversion;
use image::GenericImageView;
use polars::prelude::*;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_bytes::ByteBuf;
use std::fmt;
use std::marker::PhantomData;
#[macro_use]

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Debug, PartialEq, Clone)]
struct DiabetesPatient {
    label: i64,
    pub image: [[f32; WIDTH]; HEIGHT],
}

#[derive(Serialize, Deserialize, Clone)]
struct DiabetesPatientRaw {
    pub label: usize,
    pub image: Vec<f32>,
}

/*데이터 셋 구조체 */
#[derive(Debug)]
pub struct DiabetesDataset {
    dataset: Vec<DiabetesPatient>,
}
#[derive(Module, Debug)] //딥러닝 모듈생성
pub struct Model<B: Backend> {
    //BackEnd:새모델이 모든 벡엔드에서 실행할수 있게함
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}

/* Model method */
impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        /*채널, */
        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        /*1024(16 _ 8 _ 8) */
        self.linear2.forward(x) // [batch_size, num_classes]
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

//network의 기본 구성
//구성을 직렬화하여 모델 hyperprameter를 쉽게 저장
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize, //
    hidden_size: usize, //
    #[config(default = "0.5")]
    dropout: f64, //dropout
}
impl DiabetesDataset {
    pub fn test_data() -> Self {
        let test_df = CsvReader::from_path("./datasets/digit-recognizer/test.csv")
            .unwrap()
            .finish()
            .unwrap();

        let pixel = test_df
            .to_ndarray::<Float32Type>(IndexOrder::Fortran)
            .unwrap();

        let mut pixex_vec: Vec<Vec<_>> = Vec::new();
        for row in pixel.outer_iter() {
            let row_vec: Vec<_> = row.iter().cloned().collect();
            pixex_vec.push(row_vec);
        }
        let mut bb: Vec<DiabetesPatient> = Vec::new();

        for k in 0..pixex_vec.len() {
            let two_dimensional_array = vec_to_2d_array(pixex_vec[k].clone());

            bb.push(DiabetesPatient {
                label: 0,
                image: two_dimensional_array,
            });
        }
        DiabetesDataset { dataset: bb }
    }
    pub fn new() -> Self {
        let train_df = CsvReader::from_path("./datasets/digit-recognizer/train.csv")
            .unwrap()
            .finish()
            .unwrap();
        let labels: Vec<i64> = train_df
            .column("label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let pixel = train_df
            .drop("label")
            .unwrap()
            .to_ndarray::<Float32Type>(IndexOrder::Fortran)
            .unwrap();

        let mut year_built_vec: Vec<Vec<_>> = Vec::new();
        for row in pixel.outer_iter() {
            let row_vec: Vec<_> = row.iter().cloned().collect();
            year_built_vec.push(row_vec);
        }

        //3차원배열로만들어야함
        let mut bb: Vec<DiabetesPatient> = Vec::new();

        for k in 0..labels.len() {
            let two_dimensional_array = vec_to_2d_array(year_built_vec[k].clone());

            bb.push(DiabetesPatient {
                label: labels[k],
                image: two_dimensional_array,
            });
        }
        DiabetesDataset { dataset: bb }
    }
}
impl<B: AutodiffBackend> TrainStep<Test<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Test<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Test<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Test<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            //커널 크기 3사용
            //채널 1에서 8로 확장
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
            // //8에서 16으로 확장
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
            //적응형 평균 폴링 모듈을 사용 이미지의 차원을 8x8으로 축소
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes)
                .init_with(record.linear2),
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
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}
impl Dataset<DiabetesPatient> for DiabetesDataset {
    fn get(&self, index: usize) -> Option<DiabetesPatient> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
impl<B: Backend> Batcher<DiabetesPatient, Test<B>> for Tester<B> {
    fn batch(&self, items: Vec<DiabetesPatient>) -> Test<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, WIDTH, WIDTH]))
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()])))
            .collect();
        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);
        Test { images, targets }
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}
fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: DiabetesPatient) -> i32 {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model should exist");
    let model = config.model.init_with::<B>(record).to_device(&device);
    let batcher = Tester::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    let a: i32 = predicted.elem();
    a
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
        .build(DiabetesDataset::new());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DiabetesDataset::new());

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
    let config = ModelConfig::new(10, 1024);
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    //학습
    //  train::<MyAutodiffBackend>(
    //     "./train",
    //     TrainingConfig::new(config, AdamConfig::new()),
    //     device,
    // );

    // let a= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,113,183,174,253,193,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,190,183,252,252,252,252,253,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,237,252,253,252,252,252,252,253,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,158,252,252,253,231,189,119,128,253,224,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,199,252,252,164,104,54,0,0,0,165,217,110,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,253,253,253,165,0,0,0,0,0,166,253,253,253,147,0,0,0,0,0,0,0,0,0,0,0,0,0,43,252,252,252,252,129,128,78,0,116,253,252,252,252,112,0,0,0,0,0,0,0,0,0,0,0,0,0,43,252,252,252,252,253,252,251,232,249,253,252,252,141,4,0,0,0,0,0,0,0,0,0,0,0,0,0,14,163,247,252,252,253,252,252,252,252,253,252,244,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,47,103,235,253,252,252,252,252,253,252,252,198,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,107,27,216,253,253,255,253,253,253,253,175,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,252,252,186,205,252,252,252,253,170,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,252,252,106,16,118,196,249,253,252,170,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,252,252,62,0,0,0,115,253,252,245,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,252,252,150,0,0,0,0,165,252,252,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,204,253,255,174,12,0,0,61,253,253,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,90,252,253,252,195,190,146,227,252,252,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,161,253,252,252,252,252,253,252,252,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,156,252,252,252,252,253,252,141,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,77,200,252,252,253,173,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    // ];
    // let mut result: [[f32; 28]; 28] = [[0.0; 28]; 28];

    // // Convert the 1D array to a 2D array
    // for i in 0..28 {
    //     for j in 0..28 {
    //         // Calculate the index in the 1D array
    //         let index = i * 28 + j;
    //         // Copy the value from the 1D array to the 2D array
    //         result[i][j] = a[index] as f32;
    //     }
    // }

    let test_data = DiabetesDataset::test_data();
    let batch_size = 1000; // 필요에 따라 일괄 크기 조절
    let result: Vec<i32> = test_data
        .dataset
        .par_chunks(batch_size)
        .enumerate()
        .flat_map(|(count, chunk)| {
            println!("count:{}", count + 1);
            chunk.par_iter().map(|item| {
                infer::<MyAutodiffBackend>(
                    "./train",
                    device.clone(),
                    DiabetesPatient {
                        image: item.image,
                        label: 8,
                    },
                )
            })
        })
        .collect();

    let survived_series = Series::new("Label", result.into_iter().collect::<Vec<i32>>());
    let passenger_id_series = Series::new("ImageId", (1..=28000).collect::<Vec<i32>>());

    let mut df: DataFrame = DataFrame::new(vec![passenger_id_series, survived_series]).unwrap();
    let mut output_file: File = File::create("./datasets/digit-recognizer/out.csv").unwrap();
    CsvWriter::new(&mut output_file).finish(&mut df).unwrap();
}

fn vec_to_2d_array(input: Vec<f32>) -> [[f32; WIDTH]; WIDTH] {
    let mut result = [[0.0; WIDTH]; WIDTH];
    for i in 0..WIDTH {
        let start = i * WIDTH;
        let end = start + WIDTH;
        result[i].copy_from_slice(&input[start..end]);
    }
    result
}
