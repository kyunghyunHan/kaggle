use ndarray::prelude::*;
use polars::prelude::*;
use plotters::prelude::*;
const OUT_FILE_NAME: &str = "./src/kaggle/house_price/house_price.png";

pub fn main(){

    let train_df: DataFrame = CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/train.csv")
    .unwrap()
    .has_header(true)
    .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
    .finish()
    .unwrap();   
    println!("{}",train_df);


    let mut c= train_df.select(["YrSold","SalePrice","MoSold"]).unwrap();
    let c: DataFrame= c.sort(&["YrSold","MoSold"], vec![false, false], true).unwrap();
    // let c= c.sort(&["MoSold"], vec![false, true], true).unwrap();

    let  x_train: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 2]>> = c.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();
    
    println!("{:?}",x_train);
    let mut data: Vec<Vec<_>> = Vec::new();
    for row in x_train.outer_iter() {
        let row_vec: Vec<_> = row.iter().cloned().collect();
        data.push(row_vec);
    }
    println!("{:?}",data);
    
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (824, 768)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();//배경
    let root_area: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = root_area.titled("House Price", ("sans-serif", 60)).unwrap();
    let mut linear_function = ChartBuilder::on(&root_area)
    .margin(5)
    .set_all_label_area_size(10)
    .build_cartesian_2d((2006f32..2011f32).step(1f32), (10000.0..800000f32).step(1500f32)).unwrap();

    linear_function
    .configure_mesh()
    .x_labels(11)
    .y_labels(31)
    .x_label_formatter(&|v| format!("{:.1}", v))
    .y_label_formatter(&|v| format!("{:.1}", v))
    .draw()
    .unwrap();

   

    linear_function.draw_series(PointSeries::of_element(
    &mut data.clone().into_iter().map(|x| ((x[0] + x[2] / 10.0), x[1])),
    3,
    ShapeStyle::from(&RED).filled(),
    &|coord, size, style| {
        EmptyElement::at(coord)
            + Circle::new((0, 0), size, style)
            // + Text::new(format!("{:?}", coord), (0, 15), ("sans-serif", 15))
    },
)).unwrap();

    linear_function.draw_series(LineSeries::new(
    &mut data.clone().into_iter().map(|x| ((x[0] + x[2] / 10.0), x[1])),
    &BLUE,
))
.unwrap();
}
