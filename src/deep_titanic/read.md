# 타이타닉

## Dataset
 - [Dataset](https://www.kaggle.com/competitions/titanic/data)

## 데이터 불러오기
- 구조체 정의
```rs
  struct Dataset {
        pub train_datas: Tensor, //train data
        pub train_labels: Tensor,
        pub test_datas: Tensor, //test data
        pub test_labels: Tensor,
    }
```
- dataset불러오기
- finish: 작업을 완료하고 최종 결과를 반환
```rs
   let train_df = CsvReader::from_path("./datasets/titanic/train.csv")
                .unwrap()
                .finish()
                .unwrap();

```
## 전처리
- null count 확인
```rs
println!("{}", train_df.null_count());
```
```
shape: (1, 12)
┌─────────────┬──────────┬────────┬──────┬───┬────────┬──────┬───────┬──────────┐
│ PassengerId ┆ Survived ┆ Pclass ┆ Name ┆ … ┆ Ticket ┆ Fare ┆ Cabin ┆ Embarked │
│ ---         ┆ ---      ┆ ---    ┆ ---  ┆   ┆ ---    ┆ ---  ┆ ---   ┆ ---      │
│ u32         ┆ u32      ┆ u32    ┆ u32  ┆   ┆ u32    ┆ u32  ┆ u32   ┆ u32      │
╞═════════════╪══════════╪════════╪══════╪═══╪════════╪══════╪═══════╪══════════╡
│ 0           ┆ 0        ┆ 0      ┆ 0    ┆ … ┆ 0      ┆ 0    ┆ 687   ┆ 2        │
└────────────┴──────────┴────────┴──────┴───┴────────┴──────┴───────┴──────────┘
```
- cabin과 Embarked 가 null값이 있는것으로 판단
- 필요 없는 데이터들 제거
```rs
  //필요없는 PassegerId,Name,Ticket제거
let train_df = train_df.drop_many(&["PassengerId", "Name", "Ticket"]);
let test_df = test_df.drop_many(&["PassengerId", "Name", "Ticket"]);
```
- null 값을 채우기 위해 AGE는 평균값으로 채워준다
- Embarked는 "S"로 채워준다
```rs
 let train_df = train_df
                .clone()
                .lazy()
                .with_columns([
                    col("Age").fill_null((col("Age").mean())),
                    col("Embarked").fill_null(lit("S")),
                ])
                .collect()
                .unwrap()
                .drop(&"Cabin").unwrap();
```
- 1
```rs
null확인:shape: (1, 7)
┌────────┬─────┬─────┬───────┬───────┬──────┬──────────┐
│ Pclass ┆ Sex ┆ Age ┆ SibSp ┆ Parch ┆ Fare ┆ Embarked │
│ ---    ┆ --- ┆ --- ┆ ---   ┆ ---   ┆ ---  ┆ ---      │
│ u32    ┆ u32 ┆ u32 ┆ u32   ┆ u32   ┆ u32  ┆ u32      │
╞════════╪═════╪═════╪═══════╪═══════╪══════╪══════════╡
│ 0      ┆ 0   ┆ 0   ┆ 0     ┆ 0     ┆ 0    ┆ 0        │
└────────┴─────┴─────┴───────┴───────┴──────┴──────────┘
```
- One hot encoding
- Pclass
```rs
 let pclass_train_dummies: DataFrame = train_df
                .select(["Pclass"])
                .unwrap()
                .to_dummies(None, false)
                .unwrap();
```
```rs
       let train_df = train_df
                .hstack(pclass_train_dummies.get_columns())
                .unwrap()
                .drop(&"Pclass")
                .unwrap();
```
- sex
- Embarked
- 상관관계확인
```rs
            println!("age_corr:{}", age_corr);
            println!("sibsp_corr:{}", sibsp_corr);
            println!("parch_corr:{}", parch_corr);
            println!("fare_corr:{}", fare_corr);
            println!("pclass_1_corr:{}", pclass_1_corr);
            println!("pclass_2_corr:{}", pclass_2_corr);
            println!("pclass_3_corr:{}", pclass_3_corr);
            println!("sex_female_corr:{}", sex_female_corr);
            println!("sex_male_corr:{}", sex_male_corr);
            println!("embarked_c_corr:{}", embarked_c_corr);
            println!("embarked_q_corr:{}", embarked_q_corr);
            println!("embarked_s_corr:{}", embarked_s_corr);
```
```
age_corr:-0.06980851528714281
sibsp_corr:-0.03532249888573559
parch_corr:0.08162940708348347
fare_corr:0.2573065223849623
pclass_1_corr:0.2859037677837427
pclass_2_corr:0.09334857241192886
pclass_3_corr:-0.3223083573729699
sex_female_corr:0.5433513806577552
sex_male_corr:-0.5433513806577551
embarked_c_corr:0.16824043121823315
embarked_q_corr:0.0036503826839721647
embarked_s_corr:-0.14968272327068557
```
- Tensor로 변환
- Tensor 구조체
```rs
Ok(Self {
                train_datas: train_datas,
                train_labels: train_labels,
                test_datas: test_datas,
                test_labels: test_labels,
            })
```
## 학습

## Test


## CSV 저장하기