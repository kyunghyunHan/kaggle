
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},//기본convolution
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
        loss::CrossEntropyLoss
    },
    tensor::{
        backend::{
            Backend,AutodiffBackend
        }, Data, Int, Tensor
    },
    train::{
        ClassificationOutput,TrainStep,TrainOutput,ValidStep,LearnerBuilder,
        metric::{
            AccuracyMetric,
            LossMetric
        }
    },
    optim::AdamConfig,
    data::dataloader::{DataLoaderBuilder,batcher::Batcher},
    record::CompactRecorder,
    backend::{
        Autodiff,
        Wgpu,
        wgpu::AutoGraphicsApi
    }
};
extern crate serde;
#[macro_use]
use serde_derive;
use serde_big_array::BigArray;

use serde::{Serialize, Deserialize, Serializer, Deserializer};

use burn::data::dataset::{Dataset, InMemDataset};
use std::path::Path;
use burn::tensor::ElementConversion;
use burn::record::Recorder;
use serde_bytes::ByteBuf;
use burn::data::dataset::transform::{Mapper, MapperDataset};
use image::GenericImageView;
use std::fmt; use std::marker::PhantomData;
use serde::de::Error;
use burn::data::dataset::SqliteDataset;
use polars::prelude::*;
#[macro_use]

const WIDTH: usize = 28; 
const HEIGHT: usize = 28; 


#[derive(Debug, PartialEq,Clone)]
struct DiabetesPatient {
    label: i64,
    pub image:[[f32;WIDTH];WIDTH],
}

#[derive(Serialize, Deserialize,Clone)]
struct DiabetesPatientRaw {
    pub label: usize,
    pub image:Vec<f32>,
}

/*데이터 셋 구조체 */
#[derive(Debug)]
pub struct DiabetesDataset {
    dataset: Vec<DiabetesPatient>,
}
#[derive(Module, Debug)]//딥러닝 모듈생성
pub struct Model<B: Backend> {//BackEnd:새모델이 모든 벡엔드에서 실행할수 있게함
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
        images: Tensor<B,3>,
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
    num_classes: usize,//
    hidden_size: usize,//
    #[config(default = "0.5")]
    dropout: f64,//dropout
}
impl DiabetesDataset {
    pub fn new() -> Self {
        let train_df= CsvReader::from_path("./datasets/digit-recognizer/train.csv").unwrap().finish().unwrap();
        let labels:Vec<i64>= train_df.column("label").unwrap().i64().unwrap().into_no_null_iter().collect();


        let pixel= train_df.drop("label").unwrap().to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();

 let mut year_built_vec: Vec<Vec<_>> = Vec::new();
    for row in pixel.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        year_built_vec.push(row_vec);
    }
  
      //3차원배열로만들어야함
        let mut bb: Vec<DiabetesPatient>= Vec::new();
  
        for k in 0..labels.len(){
            let two_dimensional_array = vec_to_2d_array(year_built_vec[k].clone());

            bb.push(DiabetesPatient{label:labels[k],image:two_dimensional_array});
        }        
        DiabetesDataset{dataset:bb}
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
 fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: DiabetesPatient) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let label = item.label;
    let batcher = Tester::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);

    println!("{}",output.to_data());
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    //예측값과 실제 레이블값
    //학습이 이상하게 댐
    println!("Predicted {} Expected {}", predicted, label);
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
pub fn main(){

  

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

    let a= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,48,143,186,244,143,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,209,253,252,252,252,252,192,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,166,241,252,253,252,170,162,252,252,113,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,61,234,252,252,243,121,44,2,21,245,252,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,252,252,243,163,50,0,0,0,5,101,88,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,105,234,252,210,88,0,0,0,0,74,199,240,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,185,252,210,21,0,4,12,41,231,249,252,252,55,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,242,252,218,154,154,184,252,253,252,252,248,184,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,209,252,252,252,252,252,252,253,252,252,196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,57,142,95,142,61,81,253,252,209,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,177,255,230,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,124,252,245,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,252,252,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,248,252,233,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,231,252,202,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,175,248,252,136,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,109,252,252,159,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,218,252,252,192,141,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,132,252,252,252,205,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,132,252,252,146,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ];
    let mut result: [[f32; 28]; 28] = [[0.0; 28]; 28];

    // Convert the 1D array to a 2D array
    for i in 0..28 {
        for j in 0..28 {
            // Calculate the index in the 1D array
            let index = i * 28 + j;
            // Copy the value from the 1D array to the 2D array
            result[i][j] = a[index] as f32;
        }
    }
    infer::<MyAutodiffBackend >("./train",device,DiabetesPatient{image:result,label:2})
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