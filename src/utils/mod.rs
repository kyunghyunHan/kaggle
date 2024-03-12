use polars::prelude::cov::{cov, pearson_corr as pc};
use polars::prelude::*;
use std::any::TypeId;
pub fn corr_fn(train_df: &DataFrame,target: &str, tp: TypeId) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();

    for i in train_df.schema() {
        if tp == TypeId::of::<i64>() {
            if pc(
                train_df.column(&i.0).unwrap().cast(&DataType::Int64).unwrap().i64().unwrap(),
                train_df.column(target).unwrap().i64().unwrap(),
                1,
            )
            .unwrap()
                .abs() // 절댓값을 취합니다.
                > 0.2
            {
                v.push(i.0.to_string())
            }
        } else if tp == TypeId::of::<f64>() {
            if pc(
                train_df.column(&i.0).unwrap().cast(&DataType::Float64).unwrap().f64().unwrap(),
                train_df.column(target).unwrap().f64().unwrap(),
                1,
            )
            .unwrap()
                .abs() // 절댓값을 취합니다.
                > 0.2
            {
                v.push(i.0.to_string())
            }
        } else if tp == TypeId::of::<u64>() {
            if pc(
                train_df.column(&i.0).unwrap().cast(&DataType::UInt64).unwrap().u64().unwrap(),
                train_df.column(target).unwrap().u64().unwrap(),
                1,
            )
            .unwrap()
                .abs() // 절댓값을 취합니다.
                > 0.2
            {
                v.push(i.0.to_string())
            }
        }
    }
    v
}
