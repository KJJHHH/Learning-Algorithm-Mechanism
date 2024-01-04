# 新型學習演算法 (New Learning Algorithm)
New Learning Algorithm (2021), by Prof. TSAIH RUA-HUAN at Management Information Systems in NCCU

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

1. Full learning algorithm 
    Train loss / Test loss
    - epoch 50               # for each module
    - learning goal: e ** 2  # for all residual square < learning goals in train data
    - learning rate: 0.01    # for each module

    |        Dataset       | Full learning algorithm | Train time            |
    | -------------------- | ----------------------- | --------------------- |
    | Copper               |        1.366/19.586     |   250 min (not sure)  |

2. Benchmark: simple fully connected net (2 ~ 3)\
    Trainloss / Testloss/ Epochs / Traintime(Min)

    |  Dataset   | Two layer net           | Three layer net        | Four layer net |
    | ---------- | ----------------------- | ---------------------- | -------------- |
    |   Copper   | 1.075/19.261/2000/4     |   Not good (NOTE 3)    |        -       |

    - learning rate: 0.001, 0.01 will explode
    - epoch: 2000 (check NOTE 3)
    - hidden nodes: 50

3. The residual of Full learning algorithm and twolayer net (blue: Full learning algorithm; red: two layer net)
    - Full learning algorithm maximum test square loss: 136.271
    - Two layer net maximum test square loss: 223.862
    - Full learning algorithm test square loss variance: 532.09
    - Two layer net test square loss variance: 1760.782
        ![Alt text](image-8.png)


## Conclusion 
1. The full learning algorithm might not get a better model, but a more stable model
2. Need more data to validate
3. The mechanisms proposed in the class are not only this one, could try others

> NOTE
1. The Full learning algorithm do not need large epochs since got initialisation; the benchmark

2. The learning goal cannot be too long; otherwise the training time will be too long

3. Benchmark model with 10000 epochs
    - Two layer net on copper
        ![Alt text](image-3.png)
        ![Alt text](image-4.png)
    - Three layer net on copper
        ![Alt text](image-5.png)

<!--- 1. Process | Weight tune, Reorganising

    | Train loss/Test loss| Weight Tune | Reorganise | Nodes after Reorganise |
    | ---------------| -------------- | ----------------| --------------|
    | Trial 1 | 24.27/946.46 | 13.19/716.24 | 24 |


2. Process | Weight tune, Cramming , and Reorganise | Train loss/Test loss 

    |         | Weight Tune  | Cram          | Reorganise   | Nodes after Cram | Nodes after Reorganise | 
    | ------- | ------------ | ------------- | ------------ | ---------------- | ---------------------- |
    | Trial 1 | 28.61/767    | 26.28/679     | 11.03/612    | 164              |  not recorded          |
    | Trial 2 | 16.23/805.49 | 23.17/845.56  | 12.52/629.38 | 164              |    28                  |

>


