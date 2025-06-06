# Описания датасетов

## 1.  wisc_bc_data

- **Источник:** [UCI Machine Learning Repository | Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Примеры:** 569
- **Признаки:** 30
- **Задача:** классификация

Примеры получены из оцифрованных изображений тонкоигольных аспиратов образований молочной железы. Примеры отражают характеристики клеточных ядер, такие как радиус, текстура, периметр, площадь, гладкость и другие. С каждым примером ассоциирован диагноз, относящий исследованную опухоль к злокачественной (M, 212 примеров) либо к доброкачественной (B, 357 примеров). Простой статистический анализ приведён в таблице 1.1.

*Таблица 1.1. df.describe() для wisc_bc_data*

|      |               id |   radius_mean |   texture_mean |   perimeter_mean |   area_mean |   smoothness_mean |   compactness_mean |   concavity_mean |   points_mean |   symmetry_mean |   dimension_mean |   radius_se |   texture_se |   perimeter_se |   area_se |   smoothness_se |   compactness_se |   concavity_se |    points_se |   symmetry_se |   dimension_se |   radius_worst |   texture_worst |   perimeter_worst |   area_worst |   smoothness_worst |   compactness_worst |   concavity_worst |   points_worst |   symmetry_worst |   dimension_worst |
|:------|-----------------:|--------------:|---------------:|-----------------:|------------:|------------------:|-------------------:|-----------------:|--------------:|----------------:|-----------------:|------------:|-------------:|---------------:|----------:|----------------:|-----------------:|---------------:|-------------:|--------------:|---------------:|---------------:|----------------:|------------------:|-------------:|-------------------:|--------------------:|------------------:|---------------:|-----------------:|------------------:|
| mean  |      3.03718e+07 |      14.1273  |       19.2896  |           91.969 |     654.889 |         0.0963603 |          0.104341  |        0.0887993 |     0.0489191 |       0.181162  |       0.0627976  |    0.405172 |     1.21685  |        2.86606 |   40.3371 |      0.00704098 |        0.0254781 |      0.0318937 |   0.0117961  |    0.0205423  |     0.0037949  |       16.2692  |        25.6772  |          107.261  |      880.583 |          0.132369  |            0.254265 |          0.272188 |      0.114606  |        0.290076  |         0.0839458 |
| std   |      1.25021e+08 |       3.52405 |        4.30104 |           24.299 |     351.914 |         0.0140641 |          0.0528128 |        0.0797198 |     0.0388028 |       0.0274143 |       0.00706036 |    0.277313 |     0.551648 |        2.02185 |   45.491  |      0.00300252 |        0.0179082 |      0.0301861 |   0.00617029 |    0.00826637 |     0.00264607 |        4.83324 |         6.14626 |           33.6025 |      569.357 |          0.0228324 |            0.157336 |          0.208624 |      0.0657323 |        0.0618675 |         0.0180613 |
| min   |   8670           |       6.981   |        9.71    |           43.79  |     143.5   |         0.05263   |          0.01938   |        0         |     0         |       0.106     |       0.04996    |    0.1115   |     0.3602   |        0.757   |    6.802  |      0.001713   |        0.002252  |      0         |   0          |    0.007882   |     0.0008948  |        7.93    |        12.02    |           50.41   |      185.2   |          0.07117   |            0.02729  |          0        |      0         |        0.1565    |         0.05504   |
| 25%   | 869218           |      11.7     |       16.17    |           75.17  |     420.3   |         0.08637   |          0.06492   |        0.02956   |     0.02031   |       0.1619    |       0.0577     |    0.2324   |     0.8339   |        1.606   |   17.85   |      0.005169   |        0.01308   |      0.01509   |   0.007638   |    0.01516    |     0.002248   |       13.01    |        21.08    |           84.11   |      515.3   |          0.1166    |            0.1472   |          0.1145   |      0.06493   |        0.2504    |         0.07146   |
| 50%   | 906024           |      13.37    |       18.84    |           86.24  |     551.1   |         0.09587   |          0.09263   |        0.06154   |     0.0335    |       0.1792    |       0.06154    |    0.3242   |     1.108    |        2.287   |   24.53   |      0.00638    |        0.02045   |      0.02589   |   0.01093    |    0.01873    |     0.003187   |       14.97    |        25.41    |           97.66   |      686.5   |          0.1313    |            0.2119   |          0.2267   |      0.09993   |        0.2822    |         0.08004   |
| 75%   |      8.81313e+06 |      15.78    |       21.8     |          104.1   |     782.7   |         0.1053    |          0.1304    |        0.1307    |     0.074     |       0.1957    |       0.06612    |    0.4789   |     1.474    |        3.357   |   45.19   |      0.008146   |        0.03245   |      0.04205   |   0.01471    |    0.02348    |     0.004558   |       18.79    |        29.72    |          125.4    |     1084     |          0.146     |            0.3391   |          0.3829   |      0.1614    |        0.3179    |         0.09208   |
| max   |      9.11321e+08 |      28.11    |       39.28    |          188.5   |    2501     |         0.1634    |          0.3454    |        0.4268    |     0.2012    |       0.304     |       0.09744    |    2.873    |     4.885    |       21.98    |  542.2    |      0.03113    |        0.1354    |      0.396     |   0.05279    |    0.07895    |     0.02984    |       36.04    |        49.54    |          251.2    |     4254     |          0.2226    |            1.058    |          1.252    |      0.291     |        0.6638    |         0.2075    |

*Таблица 1.2. Выборка случайных примеров*

|  Ряд  |     id | diagnosis   |   radius_mean |   texture_mean |   perimeter_mean |   area_mean |   smoothness_mean |   compactness_mean |   concavity_mean |   points_mean |   symmetry_mean |   dimension_mean |   radius_se |   texture_se |   perimeter_se |   area_se |   smoothness_se |   compactness_se |   concavity_se |   points_se |   symmetry_se |   dimension_se |   radius_worst |   texture_worst |   perimeter_worst |   area_worst |   smoothness_worst |   compactness_worst |   concavity_worst |   points_worst |   symmetry_worst |   dimension_worst |
|----:|-------:|:------------|--------------:|---------------:|-----------------:|------------:|------------------:|-------------------:|-----------------:|--------------:|----------------:|-----------------:|------------:|-------------:|---------------:|----------:|----------------:|-----------------:|---------------:|------------:|--------------:|---------------:|---------------:|----------------:|------------------:|-------------:|-------------------:|--------------------:|------------------:|---------------:|-----------------:|------------------:|
| 222 | 892438 | M           |         19.53 |          18.9  |           129.5  |      1217   |           0.115   |            0.1642  |          0.2197  |       0.1062  |          0.1792 |          0.06552 |      1.111  |       1.161  |          7.237 |    133    |        0.006056 |          0.03203 |        0.05638 |    0.01733  |       0.01884 |       0.004787 |          25.93 |           26.24 |            171.1  |       2053   |             0.1495 |              0.4116 |           0.6121  |        0.198   |           0.2968 |           0.09929 |
| 358 | 844359 | M           |         18.25 |          19.98 |           119.6  |      1040   |           0.09463 |            0.109   |          0.1127  |       0.074   |          0.1794 |          0.05742 |      0.4467 |       0.7732 |          3.18  |     53.91 |        0.004314 |          0.01382 |        0.02254 |    0.01039  |       0.01369 |       0.002179 |          22.88 |           27.66 |            153.2  |       1606   |             0.1442 |              0.2576 |           0.3784  |        0.1932  |           0.3063 |           0.08368 |
| 537 | 893783 | B           |         11.7  |          19.11 |            74.33 |       418.7 |           0.08814 |            0.05253 |          0.01583 |       0.01148 |          0.1936 |          0.06128 |      0.1601 |       1.43   |          1.109 |     11.28 |        0.006064 |          0.00911 |        0.01042 |    0.007638 |       0.02349 |       0.001661 |          12.61 |           26.55 |             80.92 |        483.1 |             0.1223 |              0.1087 |           0.07915 |        0.05741 |           0.3487 |           0.06958 |


В репозитории [norinhossamm/Breast-Cancer-Wisconsin](https://github.com/norinhossamm/Breast-Cancer-Wisconsin/blob/main/README.md) была проведена работа по исследованию датасета:

- рассчитаны [корреляции между признаками](https://github.com/user-attachments/assets/9dd81beb-1b72-4d28-8450-c786a1868fbc)
- построены [точечные диаграммы](https://github.com/user-attachments/assets/b395ca8c-4faf-4fa8-b324-73d459f8ad35), показывающие отношения между каждой парой признаков

Возможный источник шума в этом датасете - колонка id, содержащая возрастающие целые числа, никак не влияющие на целевую переменную. В эксперименте это учтено.


## 2. car_evaluation

- **Источник:** [UCI Machine Learning Repository | Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- **Примеры:** 1728
- **Признаки:** 6
- **Задача:** классификация

Это синтетический датасет, полученный из простой иерархической модели, созданной для демонстрации экспертной системы DECMAK [1]. Датасет моделирует покупателя, выбирающего автомобиль и присваивающего оценку (`good`, `vgood`, `acc`, `unacc`) каждому экземпляру автомобиля в зависимости от стоимости покупки, цены обслуживания, количества дверей и т.д..

Особенность датасета — в нём есть только категориальные значения и нет численных.

*Таблица 2.1. df.describe() для car_evaluation: *top* — наиболее часто встречающееся значение, freq — частота наиболее частого значения*

|        | buying   | maint   |   doors |   persons | lug_boot   | safety   | class   |
|:-------|:---------|:--------|--------:|----------:|:-----------|:---------|:--------|
| unique | 4        | 4       |       4 |         3 | 3          | 3        | 4       |
| top    | vhigh    | vhigh   |       2 |         2 | small      | low      | unacc   |
| freq   | 432      | 432     |     432 |       576 | 576        | 576      | 1210    |

*Таблица 2.2. Выборка случайных примеров*

|    Ряд    | buying   | maint   |   doors |   persons | lug_boot   | safety   | class   |
|-----:|:--------|:----------|:------|:------|:--------|:------|:--------|
| 1683 | low     | low       | 4     | 4     | small   | med   | acc     |
| 1150 | med     | med       | 4     | 4     | big     | high  | vgood   |
|  314 | vhigh   | med       | 5more | more  | small   | low   | unacc   |

import pandas as pd
df = pd.read_csv('data/car_evaluation.csv')
print(df.sample(n=3).to_markdown())


## 3. StudentPerformanceFactors

- **Источник:** [Kaggle | Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data)
- **Примеры:** 6607
- **Признаки:** 19
- **Задача:** регрессия

Этот [синтетический](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/discussion/532279#3001125) датасет моделирует успеваемость учащихся в зависимости от многих факторов: количества часов обучения в неделю, доступ к интернету, тип школы — частная или муниципальная, — пол ученика, доход семьи и других. Целевая переменная — оценка на экзамене.


*Таблица 3.1. df.describe() для StudentPerformanceFactors, численные значения*
|       |   Hours_Studied |   Attendance |   Sleep_Hours |   Previous_Scores |   Tutoring_Sessions |   Physical_Activity |   Exam_Score |
|:------|----------------:|-------------:|--------------:|------------------:|--------------------:|--------------------:|-------------:|
| mean  |        19.9753  |      79.9774 |       7.02906 |           75.0705 |             1.49372 |             2.96761 |     67.2357  |
| std   |         5.99059 |      11.5475 |       1.46812 |           14.3998 |             1.23057 |             1.03123 |      3.89046 |
| min   |         1       |      60      |       4       |           50      |             0       |             0       |     55       |
| 25%   |        16       |      70      |       6       |           63      |             1       |             2       |     65       |
| 50%   |        20       |      80      |       7       |           75      |             1       |             3       |     67       |
| 75%   |        24       |      90      |       8       |           88      |             2       |             4       |     69       |
| max   |        44       |     100      |      10       |          100      |             8       |             6       |    101       |

*Таблица 3.2. df.describe() для StudentPerformanceFactors, категориальные значения*

|        | Parental_Involvement   | Access_to_Resources   | Extracurricular_Activities   | Motivation_Level   | Internet_Access   | Family_Income   | Teacher_Quality   | School_Type   | Peer_Influence   | Learning_Disabilities   | Parental_Education_Level   | Distance_from_Home   | Gender   |
|:-------|:-----------------------|:----------------------|:-----------------------------|:-------------------|:------------------|:----------------|:------------------|:--------------|:-----------------|:------------------------|:---------------------------|:---------------------|:---------|
| unique | 3                      | 3                     | 2                            | 3                  | 2                 | 3               | 3                 | 2             | 3                | 2                       | 3                          | 3                    | 2        |
| top    | Medium                 | Medium                | Yes                          | Medium             | Yes               | Low             | Medium            | Public        | Positive         | No                      | High School                | Near                 | Male     |
| freq   | 3362                   | 3319                  | 3938                         | 3351               | 6108              | 2672            | 3925              | 4598          | 2638             | 5912                    | 3223                       | 3884                 | 3814     |

*Таблица 3.3. Выборка случайных примеров*

|      |   Hours_Studied |   Attendance | Parental_Involvement   | Access_to_Resources   | Extracurricular_Activities   |   Sleep_Hours |   Previous_Scores | Motivation_Level   | Internet_Access   |   Tutoring_Sessions | Family_Income   | Teacher_Quality   | School_Type   | Peer_Influence   |   Physical_Activity | Learning_Disabilities   | Parental_Education_Level   | Distance_from_Home   | Gender   |   Exam_Score |
|-----:|----------------:|-------------:|:-----------------------|:----------------------|:-----------------------------|--------------:|------------------:|:-------------------|:------------------|--------------------:|:----------------|:------------------|:--------------|:-----------------|--------------------:|:------------------------|:---------------------------|:---------------------|:---------|-------------:|
|  471 |               6 |           93 | Medium                 | Medium                | No                           |             7 |                52 | High               | Yes               |                   0 | Medium          | Medium            | Public        | Neutral          |                   2 | No                      | College                    | Near                 | Female   |           64 |
| 5023 |              10 |           66 | Medium                 | Medium                | No                           |             4 |                74 | Low                | Yes               |                   2 | Low             | Medium            | Public        | Negative         |                   2 | No                      | High School                | Moderate             | Male     |           59 |
|  921 |              14 |           74 | High                   | High                  | Yes                          |             6 |                73 | High               | Yes               |                   2 | Medium          | Medium            | Private       | Neutral          |                   3 | No                      | High School                | Near                 | Female   |           67 |


## 4. allhyper

- **Источник:** [PMLB | allhyper](https://epistasislab.github.io/pmlb/profile/allhyper.html)
- **UCI ML URL:** https://archive.ics.uci.edu/dataset/102/thyroid+disease
- **Примеры:** 3771
- **Признаки:** 29
- **Задача:** классификация

Набор данных представляет собой медицинские показатели пациентов, у которых подозревается/диагностирован гипертериоз, заболевание щитовидной железы. Наиболее восприимчивы к этому заболеванию пациентки женского пола. Болезнь проявляется повышенным содержанием гормонов трийодтиронина и тироксина.

Целевая переменная принимает значения 3, 2, 1 и 0, обозначающие:

- 3 - нет имеет гипертиреоза (3670 примеров);
- 2 - *первичный гипертиреоз* (79 примеров);
- 1 - *компенсированный гипертиреоз* (12 примеров);
- 0 - *вторичный гипертериоз* (10 примеров).

<!-- https://github.com/MazyCarneiro/Thyroid-disease-dataset/tree/master -->

## 5. eye_movements

- **Источник:** [HuggingFace | inria-soda:tabular-benchmark/clf_cat/](https://huggingface.co/datasets/inria-soda/tabular-benchmark/tree/dabc0f5cea2459217a54bf275227e68cda218e9d/clf_cat)
- **Год**: 2005
- **OpenML URL**: https://www.openml.org/search?type=data&sort=runs&id=1044&status=active
- **Примеры:** 7608
- **Признаки:** 23
- **Задача:** классификация

Датасет *eye_movements* был собран для исследования возможности выведения релевантности текста на основании считывания движения глаз [2]. Различным траекториям и длительностям движений глаз присвоет один из двух классов (0 - релевантно, 1 - нерелевантно). Каждый пример представляет собой одно видимое испытуемыми слово. Оба класса представлены в наборе в равном количестве.

## 6. wine_quality

- **Источник:** [HuggingFace | inria-soda:tabular-benchmark/num_reg/](https://huggingface.co/datasets/inria-soda/tabular-benchmark/tree/dabc0f5cea2459217a54bf275227e68cda218e9d/reg_num)
- **Год**: 2009
- **Примеры:** 6497
- **Признаки:** 11
- **Задача:** регрессия

Этот набор данных состоит из примеров химического анализа  образцов вин португальского вина "Vinho Verde" [3]. В качестве целоевой переменной представлена вкусовая оценка эксперта по шкале от 3 (худшее) до 9 (лучшее). Особенность: значения признаков даны с небольшой фиксированной точностью, что привносит шум, связанный с точностью измерений.

## 7. Hill_Valley_with_noise

- **Источник:** [PMLB | Hill_Valley_with_noise](https://epistasislab.github.io/pmlb/profile/Hill_Valley_with_noise.html)
- **Год**: 2008
- **UCI ML URL:** https://archive.ics.uci.edu/dataset/166/hill+valley
- **Примеры:** 1212
- **Признаки:** 100
- **Задача:** классификация

Каждый пример является совокупностью из 100 точек на двухмерном графике. Каждое значение точки это её *y*-координата, а *x* координата варьируется от 0 до 100. Таким образом образуется кривая, которая классифицируется либо как "впадина" (0), либо как "холм" (1).

Набор данных представлен в виде двух вариантов: первый вариант (этот) представляет собой более примеры более грубых "рельефов", а второй – более плавных.

## 8. Hill_Valley_without_noise

- **Источник:** [PMLB | Hill_Valley_without_noise](https://epistasislab.github.io/pmlb/profile/Hill_Valley_without_noise.html)
- **Год**: 2008
- **UCI ML URL:** https://archive.ics.uci.edu/dataset/166/hill+valley
- **Примеры:** 1212
- **Признаки:** 100
- **Задача:** классификация

Вариант предыдущего набора данных с плавными "рельефами".

## 9. 294_satellite_image
- **Источник:** [PMLB | 294_satellite_image](https://epistasislab.github.io/pmlb/profile/294_satellite_image.html)
- **Год**: 1993
- **OpenML URL:** https://www.openml.org/search?type=data&sort=runs&id=294&status=active
- **Задача:** регрессия
- **Примеры:** 6435
- **Признаки:** 20

OpenML description:

> The database consists of the multi-spectral values of pixels in 3x3 neighbourhoods in a satellite image, and the classification associated with the central pixel in each neighbourhood. The aim is to predict this classification, given the multi-spectral values. In the sample database, the class of a pixel is coded as a number.


## 10. 1030_ERA

- **Примеры:**
- **Признаки:**

https://www.openml.org/search?type=data&sort=runs&id=573&status=active


## Список литературы

1. M. Bohanec, V. Rajkovič, *Knowledge acquisition and explanation for multi-attribute decision making*. Proceedings of the 8th International Workshop 'Expert Systems and Their Applications AVIGNON 88', 1:59-78, 1988 (https://kt.ijs.si/MarkoBohanec/pub/Avignon88.pdf)
2. Jarkko Salojarvi, Kai Puolamaki, Jaana Simola, Lauri Kovanen, Ilpo Kojo, Samuel Kaski. Inferring Relevance from Eye Movements: Feature Extraction. Helsinki University of Technology, Publications in Computer and Information Science, Report A82. 3 March 2005. Data set at http://www.cis.hut.fi/eyechallenge2005/
3. P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.