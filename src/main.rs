mod house_price;
mod spaceship_titanic;
mod titanic;
mod store_sales;
mod binary_classification_with_a_bank_churn_dataset;
mod price_prediction_for_used_cars;
mod predict_co2_emissions_in_rwanda;
mod forecasting_mini_cource_sales;
mod data_science_london;
mod icr_identifying_age_related_conditions;
mod digit_recognizer;
mod facial_keypoints_detection;
mod natural_language_processing_with_disaster_tweets;
mod binary_prediction_of_smoker_status_using_bio_signals;
mod penguins_binary_classification;
use rayon::prelude::*;
fn main() {
    // titanic::titanic::main();
    // price_prediction_for_used_cars::model::main();
    // spaceship_titanic::model::main();
    // house_price::model::main();
    // store_sales::model::main();
    // binary_classification_with_a_bank_churn_dataset::model::main();
    // predict_co2_emissions_in_rwanda::model::main();
    // forecasting_mini_cource_sales::model::main();
    // data_science_london::model::main();
    // icr_identifying_age_related_conditions::model::main();
    // digit_recognizer::model::main();
    // facial_keypoints_detection::model::main();
    // natural_language_processing_with_disaster_tweets::model::main();
    // binary_prediction_of_smoker_status_using_bio_signals::model::main();
    penguins_binary_classification::model::main();//이진 분류를 위한 펭귄
    
}
