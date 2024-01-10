use polars::prelude::*;
pub fn main(){
    /*===============data============= */
    let train_df: DataFrame = CsvReader::from_path("./datasets/playground-series-s3e20/train.csv")
    .unwrap()
    .finish()
    .unwrap().drop_many(&["UvAerosolLayerHeight_aerosol_height", "UvAerosolLayerHeight_aerosol_pressure", "UvAerosolLayerHeight_aerosol_optical_depth",
    "UvAerosolLayerHeight_sensor_zenith_angle", "UvAerosolLayerHeight_sensor_azimuth_angle", "UvAerosolLayerHeight_solar_azimuth_angle",
    "UvAerosolLayerHeight_solar_zenith_angle", "ID_LAT_LON_YEAR_WEEK"]);
    println!("{:?}",train_df.shape());
    println!("{:?}",train_df.null_count());
    println!("{:?}",train_df.schema());
    println!("{:?}",train_df.dtypes());
    println!("{:?}",train_df.width());


    /*===============data============= */
    
    // let train_df: LazyFrame= train_df.clone().lazy().with_column(cols().all(true))
  
  
}