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
const WIDTH: usize = 8;
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PenguinsData {
    // pub species: i64,
    // pub island: i64,
    // pub bill_length_mm: i64,
    // pub bill_depth_mm: i64,
    // pub flipper_length_mm: i64,
    // pub body_mass_g: i64,
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
pub fn main() {
    println!("{:?}", PenguinsDataset::new())
}
