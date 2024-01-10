# Bank Customer Churn Prediction

## DATA

- Customer ID: 각 고객의 고유 식별자
- Surname: 고객의 성 또는 성
- Credit Score: 고객의 신용점수를 나타내는 수치
- Geography: 고객이 거주하는 국가(프랑스, 스페인, 독일)
- Gender: 고객의 성별(남성 또는 여성)
- Age: 고객의 나이입니다.
- Tenure: 고객이 은행에 근무한 연수
- Balance: 잔액 : 고객의 계정 잔액
- NumOfProducts: 고객이 이용하는 은행 상품 수(예: 적금, 신용카드)
- HasCrCard: 고객의 신용카드 보유 여부(1 = 예, 0 = 아니요)
- IsActiveMember: 고객이 활성 회원인지 여부(1 = 예, 0 = 아니요)
- EstimatedSalary: 고객의 예상 급여
- Exited: Whether 고객이 이탈했는지 여부(1 = 예, 0 = 아니요)


## data 확인
- 범주형:Surname(명목형),Geography(명목형),Gender(명목형),HasCrCard(명목형),IsActiveMember(명목형)
- 수치형:CreditScore(이산형),Age(연속형),Tenure(이산형),NumOfProducts(이산형),EstimatedSalary(연속형)

## 결측치 확인
- 없음

## Surname
- 필요 없으므로 drop

## CreditScore


## Geography
```
Geography:::::shape: (3, 2)
┌───────────┬─────────────┐
│ Geography ┆ Exited_mean │
│ ---       ┆ ---         │
│ str       ┆ f64         │
╞═══════════╪═════════════╡
│ France    ┆ 0.165282    │
│ Spain     ┆ 0.172176    │
│ Germany   ┆ 0.378952    │
└───────────┴─────────────┘
```
- Germany에 사는 사람이 이탈을 많이 함

## Gender

```
Gender::::::shape: (2, 2)
┌────────┬─────────────┐
│ Gender ┆ Exited_mean │
│ ---    ┆ ---         │
│ str    ┆ f64         │
╞════════╪═════════════╡
│ Male   ┆ 0.159055    │
│ Female ┆ 0.279687    │
└────────┴─────────────┘
```
- 남성보다 여성이 더 이탈을 마니함

## IsActiveMember

## HasCrCard

## NumOfProducts