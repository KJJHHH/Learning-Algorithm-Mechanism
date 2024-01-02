# 新型學習演算法
新型學習演算法(2021)：蔡瑞煌教授於政大資管系所開授的課程

## Introduction
The learning algorithm is revised learning mechanism based on two layer net with dynamic adjusting learning rate and three stopping criteria, which is epochs, small residual, and small learning rate, mentioned in cs231n. To overcome the overfitting and learning dilemma issues, two module, reorganizing module and cramming, are proposed to cope with each issues respectively.

## Learning Goals
Predicting copper price.

## Data
- Cleaned data provided by teacher with 18 covariate with dependent variable as price
- TODO: download more data form kaggle to experiment

## Environment
- CPU
- Python 3.10
- Pytorch framework


## Mechanism

### LTS (L11)
- Select the {ns} samples that fit the learning goal 
- Select {ks} of k element that is not in {ns}
- Take {ns} + {ks} as training data
- Keep doing full learning algorithm to get all training data

### Weight Tuning Module (L6)
- Simple learning
- If acceptable: go to Reorganise module
- If not acceptable: go to Cramming

### Reorganising Module (L7)
- Complex learning
- Removing irrelavent nodes
- Coping with Overfitting problem

### Cramming Module (L9)
- Ruled based adding nodes
- For each case that did not fit well to the model, assign three nodes for the case in the model, where the weights for each nodes is predefined

### Full learning algorithm (L11)
```
Notation
# n: picked data to train in traing data
# N: all training data
```
1. Start and INitilasie with hidden node size 1: Initilialise Module
2. Let n = obtaining_LTS, n += 1 
    - The obtaining_LTS
    - if n > N break 
3. Selecting _LTS(n). I(n) = the picked data indexes
4. If the learning goal for picked train data satisfied (max(eps) <= learning goal), Go step 7; Otherwise, there is oen and onlly one k in n that cause contradicton and k = [n]    
5. Save weight
6. Weight tune the current SLFN
    - IF acceptable: go step 7
    - Otherwise, restore weight cram to get acceptable SLFN
7. Reorganise SLFN
8. GO to step 2


## Result

1. Process | Weight tune, Reorganising

    | Train loss/Test loss| Weight Tune | Reorganise | Nodes after Reorganise |
    | ---------------| -------------- | ----------------| --------------|
    | Trial 1 | 24.27/946.46 | 13.19/716.24 | 24 |


2. Process | Weight tune, Cramming , and Reorganise | Train loss/Test loss 

    |         | Weight Tune  | Cram          | Reorganise   | Nodes after Cram | Nodes after Reorganise | 
    | ------- | ------------ | ------------- | ------------ | ---------------- | ---------------------- |
    | Trial 1 | 28.61/767    | 26.28/679     | 11.03/612    | 164              |  not recorded          |
    | Trial 2 | 16.23/805.49 | 23.17/845.56  | 12.52/629.38 | 164              |    28                  |

3. Process | Full Path
    - Utilising full learning algorithm to predict different data.

        |Train loss / Test loss| Full learning algorithm | Train time            |
        | -------------------- | ----------------------- | --------------------- |
        | Copper               |        1.505/19.585     |   250 min (not sure)  |
    
    - Benchmark: simple two layer net (2 ~ 5)

        |Train loss / Test loss| Two layer net      | Three layer net | Four layer net |
        | -------------------- | ------------------ | --------------- | -------------- |
        | Copper               |    6.648/836.299   |                 |                |

