# Mnist

## Dataset
 - [Dataset](https://www.kaggle.com/competitions/titanic/data)

## 데이터 불러오기
- 데이터 불러오기
```rs
      let  train_df=  CsvReader::from_path("./datasets/playground-series-s3e24/train.csv").unwrap().finish().unwrap();
```
## 데이터 확인
```rs
println!("데이터 정보 확인:{:?}", train_df.schema());
```
```
데이터 정보 확인:Schema:
name: id, data type: Int64
name: age, data type: Int64
name: height(cm), data type: Int64
name: weight(kg), data type: Int64
name: waist(cm), data type: Float64
name: eyesight(left), data type: Float64
name: eyesight(right), data type: Float64
name: hearing(left), data type: Int64
name: hearing(right), data type: Int64
name: systolic, data type: Int64
name: relaxation, data type: Int64
name: fasting blood sugar, data type: Int64
name: Cholesterol, data type: Int64
name: triglyceride, data type: Int64
name: HDL, data type: Int64
name: LDL, data type: Int64
name: hemoglobin, data type: Float64
name: Urine protein, data type: Int64
name: serum creatinine, data type: Float64
name: AST, data type: Int64
name: ALT, data type: Int64
name: Gtp, data type: Int64
name: dental caries, data type: Int64
name: smoking, data type: Int64

```
## 전처리
 - null 값 확인
```rs

```
- 데이터 확인 결과: =>null 값이 없는 것으로 확인
```rs
null확인:shape: (1, 24)
┌─────┬─────┬────────────┬────────────┬───┬─────┬─────┬───────────────┬─────────┐
│ id  ┆ age ┆ height(cm) ┆ weight(kg) ┆ … ┆ ALT ┆ Gtp ┆ dental caries ┆ smoking │
│ --- ┆ --- ┆ ---        ┆ ---        ┆   ┆ --- ┆ --- ┆ ---           ┆ ---     │
│ u32 ┆ u32 ┆ u32        ┆ u32        ┆   ┆ u32 ┆ u32 ┆ u32           ┆ u32     │
╞═════╪═════╪════════════╪════════════╪═══╪═════╪═════╪═══════════════╪═════════╡
│ 0   ┆ 0   ┆ 0          ┆ 0          ┆ … ┆ 0   ┆ 0   ┆ 0             ┆ 0       │
└─────┴─────┴────────────┴────────────┴───┴─────┴─────┴───────────────┴─────────┘
```
## 
## 학습

## Test


## CSV 저장하기