use burn::data::dataset::Dataset;
use burn::tensor::ElementConversion;
use burn::{
    backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
    config::Config,
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::Module,
    nn::{loss::CrossEntropyLoss, Dropout, DropoutConfig, Linear, LinearConfig, ReLU},
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
const WIDTH: usize = 256;
const HEIGHT: usize = 256;

#[derive(Debug, Clone)]
pub struct ImageData {
    pub id: i64,
    pub image: [[f32; WIDTH]; HEIGHT],
    pub label: i64,
}
#[derive(Debug)]
pub struct ImageDataset {
    dataset: Vec<ImageData>,
}

impl ImageDataset {
    pub fn new() -> Self {
        let train_df = CsvReader::from_path("./datasets/image-classifications/train.csv")
        .unwrap()
        .finish()
        .unwrap();
    let ids: Vec<i64> = train_df
        .column("ID")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<i64>>();
    let labels: Vec<i64> = train_df
        .column("Class")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<i64>>();
    let heards: Vec<String> = (1..=65536).map(|x| x.to_string()).collect();
    let images = train_df
        .select(&heards)
        .unwrap()
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();
    let mut images_vec: Vec<Vec<_>> = Vec::new();
    for row in images.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        images_vec.push(row_vec);
    }

    let mut image_dataset: Vec<ImageData> = Vec::new();

    for k in 0..images_vec.len() {
        let images_arr = vec_to_2d_array(images_vec[k].clone());

        image_dataset.push(ImageData {
            id:ids[k],
            image: images_arr,
            label:labels[k] ,
        });
    }
    ImageDataset { dataset: image_dataset }

    }
}

pub fn main() {
    let df= ImageDataset::new();
    println!("{:?}",df);
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