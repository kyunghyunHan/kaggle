// use burn::{
//     data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
//     tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
// };

// pub struct MNISTBatcher<B: Backend> {
//     device: B::Device,
// }

// impl<B: Backend> MNISTBatcher<B> {
//     pub fn new(device: B::Device) -> Self {
//         Self { device }
//     }
// }

// #[derive(Clone, Debug)]
// pub struct MNISTBatch<B: Backend> {
//     pub images: Tensor<B, 3>,
//     pub targets: Tensor<B, 1, Int>,
// }

// impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
//     fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
//         let images: Vec<_> = items
//             .iter()
//             .map(|item| Data::<f32, 2>::from(item.image))
//             .map(|data| Tensor::<B, 2>::from_data(data.convert()))
//             .map(|tensor| tensor.reshape([1, 28, 28]))
//             .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
//             .collect();
//         let targets: Vec<_> = items
//             .iter()
//             .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as f64).elem()])))
//             .collect();
//         let images = Tensor::cat(images, 0).to_device(&self.device);
//         let targets = Tensor::cat(targets, 0).to_device(&self.device);

//         MNISTBatch { images, targets }
//     }
// }
// use std::fmt::format;
// use crate::data::{MNISTBatch, MNISTBatcher};
// use burn::data::dataloader::DataLoaderBuilder;
// use burn::data::dataset::source::huggingface::{MNISTDataset, MNISTItem};
// use burn::module::AutodiffModule;
// use burn::nn::loss::CrossEntropyLoss;
// use burn::optim::{AdamConfig, Optimizer};
// use burn::record::{CompactRecorder, Recorder};
// use burn::train::metric::{AccuracyMetric, LossMetric};
// use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
// use burn::{
//     config::Config,
//     module::Module,
//     nn::{
//         conv::{Conv2d, Conv2dConfig},
//         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
//         Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
//     },
//     tensor::{
//         backend::{AutodiffBackend, Backend},
//         Int, Tensor,
//     },
// };
// use burn::data::dataloader::batcher::Batcher;
// use burn::tensor::Device;

// #[derive(Module, Debug)]
// pub struct Model<B: Backend> {
//     conv1: Conv2d<B>,
//     conv2: Conv2d<B>,
//     pool: AdaptiveAvgPool2d,
//     dropout: Dropout,
//     linear1: Linear<B>,
//     linear2: Linear<B>,
//     activation: ReLU,
// }

// #[derive(Config, Debug)]
// pub struct ModelConfig {
//     num_classes: usize,
//     hidden_size: usize,
//     #[config(default = "0.5")]
//     dropout: f64,
// }

// impl ModelConfig {
//     fn init<B: Backend>(&self) -> Model<B> {
//         Model {
//             conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
//             conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
//             pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
//             dropout: DropoutConfig::new(self.dropout).init(),
//         }
//     }

//     fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
//         Model {
//             conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
//             conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
//             pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(16*8*8, self.hidden_size).init_with(record.linear1),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init_with(record.linear2),
//             dropout: DropoutConfig::new(self.dropout).init()
//         }
//     }
// }

// impl<B: Backend> Model<B> {
//     pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
//         let [batch_size, height, width] = images.dims();
//         // create a channel at the second dimension
//         let x = images.reshape([batch_size, 1, height, width]);

//         let x = self.conv1.forward(x);
//         let x = self.dropout.forward(x);
//         let x = self.conv2.forward(x);
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);

//         let x = self.pool.forward(x);
//         let x = x.reshape([batch_size, 16 * 8 * 8]);
//         let x = self.linear1.forward(x);
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);

//         self.linear2.forward(x)
//     }

//     pub fn forward_classification(
//         &self,
//         images: Tensor<B, 3>,
//         targets: Tensor<B, 1, Int>,
//     ) -> ClassificationOutput<B> {
//         let output = self.forward(images);
//         let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());
//         ClassificationOutput::new(loss, output, targets)
//     }
// }

// impl<B: AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let item = self.forward_classification(batch.images, batch.targets);

//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

// impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
//         self.forward_classification(batch.images, batch.targets)
//     }
// }

// #[derive(Config)]
// pub struct TrainingConfig {
//     pub model: ModelConfig,
//     pub optimizer: AdamConfig,
//     #[config(default = 10)]
//     pub num_epochs: usize,
//     #[config(default = 64)]
//     pub batch_size: usize,
//     #[config(default = 4)]
//     pub num_workers: usize,
//     #[config(default = 42)]
//     pub seed: u64,
//     #[config(default = 1.0e-4)]
//     pub learning_rate: f64,
// }

// pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
//     std::fs::create_dir_all(artifact_dir).ok();
//     config
//         .save(format!("{artifact_dir}/config.json"))
//         .expect("Config should be saved successfully");

//     B::seed(config.seed);

//     let batcher_train = MNISTBatcher::<B>::new(device.clone());
//     let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(MNISTDataset::train());
//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(MNISTDataset::test());

//     let learner = LearnerBuilder::new(artifact_dir)
//         .metric_train_numeric(AccuracyMetric::new())
//         .metric_valid_numeric(AccuracyMetric::new())
//         .metric_train_numeric(LossMetric::new())
//         .metric_valid_numeric(LossMetric::new())
//         .with_file_checkpointer(CompactRecorder::new())
//         .devices(vec![device])
//         .num_epochs(config.num_epochs)
//         .build(
//             config.model.init::<B>(),
//             config.optimizer.init(),
//             config.learning_rate,
//         );
//     let model_trained = learner.fit(dataloader_train, dataloader_test);

//     model_trained
//         .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
//         .expect("Trained model should be saved successfully")
// }

// pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
//     let config = TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("Config should exist for the model");
//     let record = CompactRecorder::new().load(format!("{artifact_dir}/model").into()).expect("Trained model should exist");

//     let model = config.model.init_with::<B>(record).to_device(&device);

//     let label = item.label;
//     let batcher = MNISTBatcher::new(device);
//     let batch = batcher.batch(vec![item]);
//     let output = model.forward(batch.images);
//     let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

//     println!("Predicted {predicted} Expected {label}");
// }
pub fn main(){

    // type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    // type MyAutodiffBackend = Autodiff<MyBackend>;

    // let device = burn::backend::wgpu::WgpuDevice::default();
    // train::<MyAutodiffBackend>(
    //     "/kaggle/working/guide",
    //     TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()).with_num_workers(16),
    //     device,
    // );
}