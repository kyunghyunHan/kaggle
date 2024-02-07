use burn::data::dataset::{Dataset, InMemDataset};
use burn::record::Recorder;
use burn::tensor::ElementConversion;
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
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PenguinsData {
    pub species: i64,
    pub island: i64,
    pub bill_length_mm: i64,
    pub bill_depth_mm: i64,
    pub flipper_length_mm: i64,
    pub body_mass_g: i64,
    pub year: i64,
}
pub struct PenguinsDataset {
    dataset: Vec<PenguinsData>,
}
impl PenguinsDataset {
    pub fn new() -> Self {
        let train_df = CsvReader::from_path("./datasets/digit-penguins_binary_classification/penguins_binary_classification.csv")
            .unwrap()
            .finish()
            .unwrap();
        let penguins_data = PenguinsData {
            species: 1,
            island: 1,
            bill_length_mm: 1,
            bill_depth_mm: 1,
            flipper_length_mm: 1,
            body_mass_g: 1,
            year: 1,
        };
        let a = vec![penguins_data];
        PenguinsDataset { dataset: a }
    }
}
pub fn main() {
    println!("{}", "펭귄");
}
