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
use serde::{Deserialize, Serialize};
use burn::data::dataset::{Dataset, InMemDataset};
use std::path::Path;
use burn::tensor::ElementConversion;
use burn::record::Recorder;
use polars::prelude::*;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PenguinsData {
    #[serde(rename = "Width")]
    pub width:i64,

}
pub struct PenguinsDataset {
    dataset: Vec<PenguinsData>,
}
impl PenguinsDataset {
    pub fn new() -> Self {
        let train_df = CsvReader::from_path("./datasets/digit-recognizer/train.csv")
            .unwrap()
            .finish()
            .unwrap();
        let penguins_data= PenguinsData{
            width:1
        };
        let a= vec![penguins_data];
        PenguinsDataset { dataset: a}

    }
}
pub fn main(){

    println!("{}","펭귄");
}