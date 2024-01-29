# Titanic - Machine Learning from Disaster

## 범주형과 수치형

- 데이터에는 크게 두가지 타입으로 나눌수 있다.

### 1.범주형 데이터

- 명목형:성별,혈핵형
- 순서형:개개의 값들은 이산적이며 순서 관계가 존재하는 자료

### 2.수치형 데이터 이산형과 연속형으로 이루어진 데이터,숫자로표시

- 이산형:이산적인 값을 갖는 데이터
- 연속형:연속적인 값을 갖는 데이터

### 데이터 확인

```
┌──────────┬────────┬────────┬──────┬───┬───────┬─────────┬───────┬──────────┐
│ Survived ┆ Pclass ┆ Sex    ┆ Age  ┆ … ┆ Parch ┆ Fare    ┆ Cabin ┆ Embarked │
│ ---      ┆ ---    ┆ ---    ┆ ---  ┆   ┆ ---   ┆ ---     ┆ ---   ┆ ---      │
│ i64      ┆ i64    ┆ str    ┆ f64  ┆   ┆ i64   ┆ f64     ┆ str   ┆ str      │
╞══════════╪════════╪════════╪══════╪═══╪═══════╪═════════╪═══════╪══════════╡
│ 0        ┆ 3      ┆ male   ┆ 22.0 ┆ … ┆ 0     ┆ 7.25    ┆ null  ┆ S        │
│ 1        ┆ 1      ┆ female ┆ 38.0 ┆ … ┆ 0     ┆ 71.2833 ┆ C85   ┆ C        │
│ 1        ┆ 3      ┆ female ┆ 26.0 ┆ … ┆ 0     ┆ 7.925   ┆ null  ┆ S        │
│ 1        ┆ 1      ┆ female ┆ 35.0 ┆ … ┆ 0     ┆ 53.1    ┆ C123  ┆ S        │
│ …        ┆ …      ┆ …      ┆ …    ┆ … ┆ …     ┆ …       ┆ …     ┆ …        │
│ 0        ┆ 1      ┆ male   ┆ 54.0 ┆ … ┆ 0     ┆ 51.8625 ┆ E46   ┆ S        │
│ 0        ┆ 3      ┆ male   ┆ 2.0  ┆ … ┆ 1     ┆ 21.075  ┆ null  ┆ S        │
│ 1        ┆ 3      ┆ female ┆ 27.0 ┆ … ┆ 2     ┆ 11.1333 ┆ null  ┆ S        │
│ 1        ┆ 2      ┆ female ┆ 14.0 ┆ … ┆ 0     ┆ 30.0708 ┆ null  ┆ C        │
└──────────┴────────┴────────┴──────┴───┴───────┴─────────┴───────┴──────────┘
```
```
데이터 정보 확인:Schema:
name: PassengerId, data type: Int64
name: Survived, data type: Int64
name: Pclass, data type: Int64
name: Name, data type: String
name: Sex, data type: String
name: Age, data type: Float64
name: SibSp, data type: Int64
name: Parch, data type: Int64
name: Ticket, data type: String
name: Fare, data type: Float64
name: Cabin, data type: String
name: Embarked, data type: String
```
- 범주형 : Survived, Sex, Embarkd / PClass
- 수치형 : SibSp, Parch / Age, Fare
- 혼합형:Ticket, Cabin
## null 처리

- null값을 확인 후

```
┌──────────┬────────┬─────┬─────┬───┬───────┬──────┬───────┬──────────┐
│ Survived ┆ Pclass ┆ Sex ┆ Age ┆ … ┆ Parch ┆ Fare ┆ Cabin ┆ Embarked │
│ ---      ┆ ---    ┆ --- ┆ --- ┆   ┆ ---   ┆ ---  ┆ ---   ┆ ---      │
│ u32      ┆ u32    ┆ u32 ┆ u32 ┆   ┆ u32   ┆ u32  ┆ u32   ┆ u32      │
╞══════════╪════════╪═════╪═════╪═══╪═══════╪══════╪═══════╪══════════╡
│ 0        ┆ 0      ┆ 0   ┆ 177 ┆ … ┆ 0     ┆ 0    ┆ 687   ┆ 2        │
└──────────┴────────┴─────┴─────┴───┴───────┴──────┴───────┴──────────┘
```

- null 값을 처리합니다.

```rs
  .with_columns([
            col("Age").fill_null((col("Age").mean())),
            col("Embarked").fill_null(lit("S")),
        ])
```

- null값이 있는지 확인
- Cabin의 경우 Nan값이 너무 많기 떄문에 drop해주고 시작
```
null확인:shape: (1, 8)
┌──────────┬────────┬─────┬─────┬───────┬───────┬──────┬──────────┐
│ Survived ┆ Pclass ┆ Sex ┆ Age ┆ SibSp ┆ Parch ┆ Fare ┆ Embarked │
│ ---      ┆ ---    ┆ --- ┆ --- ┆ ---   ┆ ---   ┆ ---  ┆ ---      │
│ u32      ┆ u32    ┆ u32 ┆ u32 ┆ u32   ┆ u32   ┆ u32  ┆ u32      │
╞══════════╪════════╪═════╪═════╪═══════╪═══════╪══════╪══════════╡
│ 0        ┆ 0      ┆ 0   ┆ 0   ┆ 0     ┆ 0     ┆ 0    ┆ 0        │
└──────────┴────────┴─────┴─────┴───────┴───────┴──────┴──────────┘
```

## 범주형 데이터 처리

- 잘 나누어 진 것을 확인

```
┌──────────┬──────────┬───────┬───────┬───┬──────────┬──────────┬──────────┬──────────┐
│ Survived ┆ Age      ┆ SibSp ┆ Parch ┆ … ┆ Sex_male ┆ Embarked ┆ Embarked ┆ Embarked │
│ ---      ┆ ---      ┆ ---   ┆ ---   ┆   ┆ ---      ┆ _C       ┆ _Q       ┆ _S       │
│ i64      ┆ f64      ┆ i64   ┆ i64   ┆   ┆ i32      ┆ ---      ┆ ---      ┆ ---      │
│          ┆          ┆       ┆       ┆   ┆          ┆ i32      ┆ i32      ┆ i32      │
╞══════════╪══════════╪═══════╪═══════╪═══╪══════════╪══════════╪══════════╪══════════╡
│ 0        ┆ 29.69911 ┆ 1     ┆ 2     ┆ … ┆ 0        ┆ 0        ┆ 0        ┆ 1        │
│          ┆ 8        ┆       ┆       ┆   ┆          ┆          ┆          ┆          │
│ 1        ┆ 26.0     ┆ 0     ┆ 0     ┆ … ┆ 1        ┆ 1        ┆ 0        ┆ 0        │
│ 0        ┆ 32.0     ┆ 0     ┆ 0     ┆ … ┆ 1        ┆ 0        ┆ 1        ┆ 0        │
└──────────┴──────────┴───────┴───────┴───┴──────────┴──────────┴──────────┴──────────┘
```

## 상관관계를 확인

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

- 상관관계를 통해 어느것을 사용할지 선택

## model 적용

- Random_Forest:0.8851674641148325

## result

- 10024/16008 기록
