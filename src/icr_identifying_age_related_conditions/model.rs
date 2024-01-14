use plotters::prelude::*;
use polars::prelude::*;

pub fn main() {
    /*============data============= */
    let train_df: DataFrame =
        CsvReader::from_path("./datasets/icr-identify-age-related-conditions/train.csv")
            .unwrap()
            .finish()
            .unwrap().drop("Id").unwrap();

    let test_df = CsvReader::from_path("./datasets/icr-identify-age-related-conditions/test.csv")
        .unwrap()
        .finish()
        .unwrap().drop("Id").unwrap();


    println!("{}", train_df);
    println!("{:?}",train_df.schema());

    /*======processing======= */

    /*========result========= */



}
