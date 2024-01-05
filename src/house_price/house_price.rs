use linfa::correlation::{self, PearsonCorrelation};
use ndarray::prelude::*;
use polars::prelude::*;
use plotters::prelude::*;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use xgboost::{parameters, DMatrix, Booster};

/*
SalePrice - 부동산 판매 가격(달러)입니다. 이것이 예측하려는 목표 변수입니다.
MSSubClass : 건물 클래스
MSZoning : 일반적인 구역 분류
LotFrontage : 부동산과 연결된 거리의 선형 피트
LotArea : 부지 크기(평방피트)
Street : 도로 접근 유형
Alley : 골목 접근 방식
LotShape : 부동산의 일반적인 형태
LandContour : 대지의 평탄도
유틸리티 : 사용 가능한 유틸리티 종류
LotConfig : 로트 구성
LandSlope : 토지의 경사
인근 지역 : Ames 시 경계 내의 물리적 위치
조건 1 : 주요 도로 또는 철도에 근접함
조건 2 : 주요 도로 또는 철도에 근접함(두 번째가 있는 경우)
BldgType : 주거 유형
HouseStyle : 주거 스타일
전반적인 품질 (OverallQual) : 전체적인 재질 및 마감 품질
GeneralCond : 전반적인 상태 등급
YearBuilt : 원래 건설 날짜
YearRemodAdd : 리모델링 날짜
RoofStyle : 지붕 유형
RoofMatl : 지붕재
Exterior1st : 주택 외부 피복재
Exterior2nd : 집의 외부 덮개(재료가 두 개 이상인 경우)
MasVnrType : 조적 베니어 유형
MasVnrArea : 석조 베니어 면적(평방 피트)
ExterQual : 외장재 품질
ExterCond : 외부 재질의 현재 상태
기초 : 기초의 종류
BsmtQual : 지하실 높이
BsmtCond : 지하실의 일반상태
BsmtExposure : 산책 또는 정원 수준 지하 벽
BsmtFinType1 : 지하 마감면적의 품질
BsmtFinSF1 : 유형 1 마감 평방피트
BsmtFinType2 : 두 번째 완성된 영역의 품질(있는 경우)
BsmtFinSF2 : 유형 2 마감 평방피트
BsmtUnfSF : 지하실의 미완성 평방피트
TotalBsmtSF : 지하 면적의 총 평방피트
난방 : 난방방식
HeatingQC : 가열 품질 및 상태
CentralAir : 중앙 에어컨
전기 : 전기 시스템
1stFlrSF : 1층 평방 피트
2ndFlrSF : 2층 평방 피트
LowQualFinSF : 낮은 품질로 마감된 평방 피트(모든 층)
GrLivArea : 지상(지상) 생활 면적 평방 피트
BsmtFullBath : 지하 욕실
BsmtHalfBath : 지하 반욕실
FullBath : 1층 이상 욕실 완비
HalfBath : 지상층 이상의 반욕실
침실 : 지하층 이상 침실 수
주방 : 주방 개수
KitchenQual : 주방 품질
TotRmsAbvGrd : 1층 위의 총 객실 수(욕실은 포함되지 않음)
기능성 : 홈 기능성 평가
벽난로 : 벽난로 수
FireplaceQu : 벽난로 품질
GarageType : 차고 위치
GarageYrBlt : 차고가 건설된 연도
GarageFinish : 차고 내부 마감
GarageCars : 차량 수용 차고의 크기
GarageArea : 차고의 크기(평방피트)
GarageQual : 차고 품질
GarageCond : 차고 상태
PavedDrive : 포장된 진입로
WoodDeckSF : 목재 데크 면적(평방피트)
OpenPorchSF : 개방형 현관 면적(제곱피트)
EnclosedPorch : 닫힌 현관 면적(평방피트)
3SsnPorch : 3계절 현관 면적(제곱피트)
ScreenPorch : 스크린 현관 면적(평방피트)
PoolArea : 평방 피트 단위의 수영장 면적
PoolQC : 수영장 품질
울타리 : 울타리 품질
MiscFeature : 다른 카테고리에서 다루지 않는 기타 기능
MiscVal : 기타 기능의 $Value
MoSold : 판매월
YrSold : 판매된 연도
SaleType : 판매 유형
SaleCondition : 판매 조건
*/
pub fn main(){
    /*===================data 불러오기========================= */

    let train_df: DataFrame = CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/train.csv")
    .unwrap()
    .has_header(true)
    .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
    .finish()
    .unwrap();   

    let test_df: DataFrame = CsvReader::from_path("./datasets/house-prices-advanced-regression-techniques/test.csv")
    .unwrap()
    .has_header(true)
    .with_null_values(Some(NullValues::AllColumns(vec!["NA".to_owned()])))
    .finish()
    .unwrap();  
    

    println!("{:?}",train_df.shape());
    println!("{:?}",train_df.head(None));
    println!("데이터 정보 확인:{:?}",train_df.schema());
    println!("수치형 데이터 확인:{:?}",train_df.describe(None).unwrap());
    println!("범주형 데이터 확인:{:?}",train_df.describe(Some(&[0f64])).unwrap());

    /*===================sale_price log 정규화========================= */
    let sale_price_data:Vec<i64>= train_df.column("SalePrice").unwrap().i64().unwrap().into_no_null_iter().collect();
    let sale_price_data:Vec<f64>=sale_price_data.into_iter().map(|x|x as f64).collect();
    let sale_price_data: Vec<f64> = sale_price_data.iter().map(|&x| x.ln()).collect();
    println!("{:?}",sale_price_data);
    /*===================sale_price log 정규화========================= */

    /*===================data 불러오기========================= */
    /*====================correlation=================== */
    // let x_data= train_df.drop("SalePrice").unwrap().to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
    // let y_data:Vec<f64>= train_df.column("SalePrice").unwrap().i64().unwrap().into_no_null_iter().into_iter().map(|x|x.as_f64()).collect();
    // let y_data= arr1(&y_data);
    // let a= Dataset::new(x_data, y_data);
    //  println!("{:?}",a.pearson_correlation_with_p_value(100).get_coeffs());
    /*====================correlation=================== */
    /*===================model========================= */

    // let target_data= train_df.column("SalePrice").unwrap();
    // let target_data:Vec<f64>= target_data.i64().unwrap().into_no_null_iter().into_iter().map(|x|x.as_f64()).collect();
    // let target_data= arr1(&target_data);
    // println!("{:?}",target_data);
    // let train_data= train_df.select(["MSSubClass"]).unwrap();
    // let x_data= train_data.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
    // let a= Dataset::new(x_data, target_data);
    // println!("{:?}",a);
    // let tt= a.pearson_correlation_with_p_value(2);
    // println!("{}",tt);
    // let model = LinearRegression::default().fit(&a).unwrap();
    // let pred = model.predict(&a);
    // let r2 = pred.r2(&a).unwrap();
    // println!("r2 from prediction: {}", r2);
    /*===================model========================= */
/*===================xgboost========================= */
// println!("{:?}",x_data.shape());
// let b = x_data.into_shape(8693* 22).unwrap();
// let x_train:Vec<f32>= b.into_iter().collect();

// let num_rows = 8693;
// let y_train:Vec<f32>= y_target.iter().map(|x|*x as f32).collect();

// // convert training data into XGBoost's matrix format
// let mut dtrain = DMatrix::from_dense(&x_train,num_rows).unwrap();

// // set ground truth labels for the training matrix
// dtrain.set_labels(&y_train).unwrap();

// // test matrix with 1 row
// println!("{:?}",test_data.shape());

// let x_test = test_data.into_shape(4277* 22).unwrap();
// let x_test:Vec<f32>= x_test.into_iter().collect();

// let num_rows = 4277;
// let result_train:Vec<f32>= result_train.iter().map(|x|*x as f32).collect();


// let mut dtest = DMatrix::from_dense(&x_test, num_rows).unwrap();
// dtest.set_labels(&result_train).unwrap();

// // specify datasets to evaluate against during training
// let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

// // specify overall training setup
// let training_params = parameters::TrainingParametersBuilder::default()
//     .dtrain(&dtrain)
//     .evaluation_sets(Some(evaluation_sets))
//     .build()
//     .unwrap();

// // train model, and print evaluation data
// let bst = Booster::train(&training_params).unwrap();
// let y= bst.predict(&dtest).unwrap();
// println!("{:?}", bst.predict(&dtest).unwrap());
/*===================xgboost========================= */

}