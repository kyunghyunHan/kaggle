pub mod model {
    use candle_core::{DType, Device, Result, Tensor, D};
    use candle_nn::{
        activation, loss, ops, Conv2d, Dropout, Linear, Module, ModuleT, Optimizer, VarBuilder,
        VarMap,
    };
    use polars::prelude::*;
    use rand::prelude::*;
    use clap::{Parser, ValueEnum};

    const RESULTS: usize = 10; //모델이; 예측하는 개수
  
    #[derive(Clone, Debug)]
    pub struct Dataset {
        pub train_images: Tensor, //train data
        pub train_labels: Tensor,
        pub test_images: Tensor, //test data
        pub test_labels: Tensor,
    }
    impl Dataset {
        fn new() -> candle_core::Result<Self> {
            let train_samples = 42_000;
            let test_samples = 28_000;
            //데이터불러오기
            let train_df = CsvReader::from_path("./datasets/digit-recognizer/train.csv")
                .unwrap()
                .finish()
                .unwrap();
            let test_df = CsvReader::from_path("./datasets/digit-recognizer/test.csv")
                .unwrap()
                .finish()
                .unwrap();
            let submission_df =
                CsvReader::from_path("./datasets/digit-recognizer/sample_submission.csv")
                    .unwrap()
                    .finish()
                    .unwrap();

            let labels = train_df
                .column("label")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>();
            let test_labels = submission_df
                .column("Label")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|x| x as u32)
                .collect::<Vec<u32>>();

            let train_labels = Tensor::from_vec(labels, (train_samples,), &Device::Cpu)?;
            let test_labels = Tensor::from_vec(test_labels, (test_samples,), &Device::Cpu)?;

            let x_test = test_df
                .to_ndarray::<Int64Type>(IndexOrder::Fortran)
                .unwrap();
            let mut test_buffer_images: Vec<u32> = Vec::with_capacity(test_samples * 784);
            for i in x_test {
                test_buffer_images.push(i as u32)
            }
            let test_images =
                (Tensor::from_vec(test_buffer_images, (test_samples, 784), &Device::Cpu)?
                    .to_dtype(DType::F32)?
                    / 255.)?;

            let x_train = train_df
                .drop("label")
                .unwrap()
                .to_ndarray::<Int64Type>(IndexOrder::Fortran)
                .unwrap();
            let mut train_buffer_images: Vec<u32> = Vec::with_capacity(train_samples * 784);
            for i in x_train {
                train_buffer_images.push(i as u32)
            }
            let train_images =
                (Tensor::from_vec(train_buffer_images, (train_samples, 784), &Device::Cpu)?
                    .to_dtype(DType::F32)?
                    / 255.)?;
            Ok(Self {
                train_images: train_images,
                train_labels: train_labels,
                test_images: test_images,
                test_labels: test_labels,
            })
        }
    }

    struct ConvNet {
        conv1: Conv2d,
        conv2: Conv2d,
        fc1: Linear,
        fc2: Linear,
        dropout: Dropout,
    }
    impl ConvNet {
        fn new(vs: VarBuilder) -> Result<Self> {
            let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
            let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
            let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
            let fc2 = candle_nn::linear(1024, RESULTS, vs.pp("fc2"))?;
            let dropout = candle_nn::Dropout::new(0.5);

            Ok(Self {
                conv1,
                conv2,
                fc1,
                fc2,
                dropout,
            })
        }
        fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
            let (b_sz, _img_dim) = xs.dims2()?;
            let xs = xs
                .reshape((b_sz, 1, 28, 28))?
                .apply(&self.conv1)?
                .max_pool2d(2)?
                .apply(&self.conv2)?
                .max_pool2d(2)?
                .flatten_from(1)?
                .apply(&self.fc1)?
                .relu()?;
            self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
        }
    }
  
   
    #[derive(ValueEnum, Clone)]
enum WhichModel {
    Linear,
    Mlp,
    Cnn,
}

#[derive(Parser)]
struct Args {
    #[clap(value_enum, default_value_t = WhichModel::Cnn)]
    model: WhichModel,

    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 200)]
    epochs: usize,

    /// The file where to save the trained weights, in safetensors format.
    #[arg(long)]
    save: Option<String>,

    /// The file where to load the trained weights from, in safetensors format.
    #[arg(long)]
    load: Option<String>,

    /// The directory where to load the dataset from, in ubyte format.
    #[arg(long)]
    local_mnist: Option<String>,
}


    struct TrainingArgs {
        learning_rate: f64,
        load: Option<String>,
        save: Option<String>,
        epochs: usize,
    }
    


    fn training_loop_cnn(
        m: Dataset,
        args: &TrainingArgs,
    ) -> anyhow::Result<()> {
        const BSIZE: usize = 64;

        let dev = candle_core::Device::cuda_if_available(0)?;

        let train_labels = m.train_labels;
        let train_images = m.train_images.to_device(&dev)?;
        let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = ConvNet::new(vs.clone())?;

        if let Some(load) = &args.load {
            println!("loading weights from {load}");
            varmap.load(load)?
        }

        let adamw_params = candle_nn::ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;
        let test_images = m.test_images.to_device(&dev)?;
        let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let n_batches = train_images.dim(0)? / BSIZE;
        let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        for epoch in 1..args.epochs {
            let mut sum_loss = 0f32;
            batch_idxs.shuffle(&mut thread_rng());
            for batch_idx in batch_idxs.iter() {
                let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
                let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;
                let logits = model.forward(&train_images, true)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_labels)?;
                opt.backward_step(&loss)?;
                sum_loss += loss.to_vec0::<f32>()?;
            }
            let avg_loss = sum_loss / n_batches as f32;

            let test_logits = model.forward(&test_images, false)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!(
                "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
                avg_loss,
                100. * test_accuracy
            );
        }
        if let Some(save) = &args.save {
            println!("saving trained weights in {save}");
            varmap.save(save)?
        }
        Ok(())
    }

    #[tokio::main]
    pub async fn main() -> anyhow::Result<()> {
        let args = Args::parse();
        let training_args = TrainingArgs{
            learning_rate:0.001,
            epochs: args.epochs,
            load: args.load,
            save: args.save,
            
        };
        let m = Dataset::new()?;
       
        training_loop_cnn(m, &training_args).unwrap();

        //추정
        let real_world_votes: Vec<u8> = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 188, 255, 94, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 191, 250, 253, 93, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 248, 253, 167, 10, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 247, 253, 208, 13, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 207, 253, 235, 77, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 209, 253, 253, 88, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 254, 253, 238, 170, 17,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 210, 254, 253, 159,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 209, 253, 254,
            240, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 253,
            253, 254, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20,
            206, 254, 254, 198, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 168, 253, 253, 196, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 20, 203, 253, 248, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 22, 188, 253, 245, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 103, 253, 253, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 89, 240, 253, 195, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 15, 220, 253, 253, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 253, 253, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 251, 253, 250, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 214, 218, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ];

        // let tensor_test_votes = Tensor::from_vec(real_world_votes.clone(), (1, IMAGE_DIM), &dev)?
        //     .to_dtype(DType::F32)?;

        // let final_result = trained_model.forward(&tensor_test_votes)?;

        // let result = final_result
        //     .argmax(D::Minus1)?
        //     .to_dtype(DType::F32)?
        //     .get(0)
        //     .map(|x| x.to_scalar::<f32>())??;
        // println!("real_life_votes: {:?}", real_world_votes);
        // println!("neural_network_prediction_result: {:?}", result);
        Ok(())
    }
}
