import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

BASE_PATH = 'good_dataset.csv'
# Не обращай внимания на некоторые конкретные номера фич в фичах с перемножением. 
# В Этом списке я задаю фичи (далее общие фичи) которые пересоздаются для тренировочного и для тестового датасетов.
# В общие фичи я кладу первый участок для обучения и второй участок для тестирования
# Закоментированные фичи были пропущенны авторами в датасете, к сожалению
def preprocess(path_to_csv,delimiter):
    train = pd.read_csv(path_to_csv,delimiter=delimiter)
    BASE_FEATURES = [
    'Мощность МПСИ',
    'Мощность МШЦ',
    'Ток МПСИ',
    'Ток МШЦ',
    'Исходное питание МПСИ',
    'Возврат руды МПСИ',
    'Общее питание МПСИ',
    'Расход воды МПСИ PV',
    'Расход воды МПСИ SP',
    'Расход воды МПСИ CV',
    'факт соотношение руда/вода МПСИ',
    'Давление на подшипник МПСИ загрузка',
    'Давление на подшипник МПСИ разгрузка',
    'Поток',
    'Поток_×_Мощность',
    'Поток_×_Ток',
    'Поток_×_Питание МПСИ 2',
    'Поток_×_Расход воды МПСИ 2 PV',
    'Поток_×_Расход воды МПСИ 2 SP',
    'Поток_×_Расход воды МПСИ 2 CV',
    'Поток_×_Давление на подшипник МПСИ 2 загрузка',
    'Поток_×_Давление на подшипник МПСИ 2 разгрузка',
    'sin_поток',
    'поток_квадрат',
    'поток_логарифм',
    'поток_лаг_1',
    'поток_лаг_2',
    'поток_лаг_3',
    'поток_лаг_4',
    'поток_лаг_5',
    'поток_лаг_6',
    'поток_лаг_7',
    'поток_лаг_8',
    'поток_лаг_9',
    'поток_лаг_10',
    'поток_лаг_11',
    'поток_лаг_12',
    'поток_лаг_13',
    'поток_лаг_14',
    'поток_лаг_15',
    'поток_лаг_16',
    'поток_rolling_10_mean',
    'поток_rolling_10_std',
    'поток_rolling_10_min',
    'поток_rolling_10_max',
    'поток_rolling_10_median',
    'поток_rolling_10_sum',
    'поток_rolling_20_mean',
    'поток_rolling_20_std',
    'поток_rolling_20_min',
    'поток_rolling_20_max',
    'поток_rolling_20_median',
    'поток_rolling_20_sum',
    'поток_rolling_30_mean',
    'поток_rolling_30_std',
    'поток_rolling_30_min',
    'поток_rolling_30_max',
    'поток_rolling_30_median',
    'поток_rolling_30_sum',
    'cos_поток',
    'tg_поток',
    'Гранулометрия в сливе',
    'Cu',
    'S',
    'Zn',
    'Переработка_руды_ВМТ',
    'Влажность',
    'Переработка_руды_СМТ',

    #'Температура масла основной маслостанции подача МПСИ',
    #'Температура масла основной маслостанции слив МПСИ',
    #'Температура масла маслостанции электродвигатель МПСИ',
    #'Температура масла редуктора МПСИ',
    #'Температура масла основной маслостанции подача МШЦ',
    #'Температура масла основной маслостанции слив МШЦ',
    #'Температура масла маслостанции электродвигатель МШЦ',
    #'Температура масла редуктора МШЦ',
    'Расход извести МШЦ',
    'Уровень в зумпфе',
    'Обороты насоса',
    'Давление в ГЦ насоса',
    'Плотность слива ГЦ',
    ]
    
    TARGET = 'Гранулометрия 2 %'

    # Создаю фичи для тестирования
    train['Поток_×_Мощность'] = train['Поток 1 л/мин'] * train['Мощность МПСИ 1 кВт']
    train['Поток_×_Ток'] = train['Поток 1 л/мин'] * train['Ток МШЦ 1 А']
    train['Поток_×_Питание МПСИ 2'] = train['Поток 1 л/мин'] * train['Исходное питание МПСИ 1 т/ч']
    train['Поток_×_Расход воды МПСИ 2 PV'] = train['Поток 1 л/мин'] * train['Расход воды МПСИ 1 PV м3/ч']
    train['Поток_×_Расход воды МПСИ 2 SP'] = train['Поток 1 л/мин'] * train['Расход воды МПСИ 1 SP м3/ч']
    train['Поток_×_Расход воды МПСИ 2 CV'] = train['Поток 1 л/мин'] * train['Расход воды МПСИ 1 CV %']
    train['Поток_×_Давление на подшипник МПСИ 2 загрузка'] = train['Поток 1 л/мин'] * train['Давление на подшипник МПСИ 1 загрузка Бар']
    train['Поток_×_Давление на подшипник МПСИ 2 разгрузка'] = train['Поток 1 л/мин'] * train['Давление на подшипник МПСИ 1 разгрузка Бар']
    train['sin_поток'] = np.sin(train['Поток 1 л/мин'])

    train['cos_поток'] = np.cos(train['Поток 1 л/мин'])
    train['tg_поток'] = np.sin(train['Поток 1 л/мин'])/np.cos(train['Поток 1 л/мин'])

    train['поток_квадрат'] = train['Поток 1 л/мин'] ** 2
    train['поток_логарифм'] = np.log(train['Поток 1 л/мин'] + 1e-6)
    train['поток_лаг_1'] = train['Поток 1 л/мин'].shift(1)
    train['поток_лаг_2'] = train['Поток 1 л/мин'].shift(2)
    train['поток_лаг_3'] = train['Поток 1 л/мин'].shift(3)
    train['поток_лаг_4'] = train['Поток 1 л/мин'].shift(4)
    train['поток_лаг_5'] = train['Поток 1 л/мин'].shift(5)
    train['поток_лаг_6'] = train['Поток 1 л/мин'].shift(6)
    train['поток_лаг_7'] = train['Поток 1 л/мин'].shift(7)
    train['поток_лаг_8'] = train['Поток 1 л/мин'].shift(8)
    train['поток_лаг_9'] = train['Поток 1 л/мин'].shift(9)
    train['поток_лаг_10'] = train['Поток 1 л/мин'].shift(10)
    train['поток_лаг_11'] = train['Поток 1 л/мин'].shift(11)
    train['поток_лаг_12'] = train['Поток 1 л/мин'].shift(12)
    train['поток_лаг_13'] = train['Поток 1 л/мин'].shift(13)
    train['поток_лаг_14'] = train['Поток 1 л/мин'].shift(14)
    train['поток_лаг_15'] = train['Поток 1 л/мин'].shift(15)
    train['поток_лаг_16'] = train['Поток 1 л/мин'].shift(16)

    train['Мощность МПСИ'] =  train['Мощность МПСИ 1 кВт']
    train['Общее питание МПСИ'] = train['Общее питание МПСИ 1 т/ч']
    train['Мощность МШЦ'] =  train['Мощность МШЦ 1 кВт']
    train['Ток МШЦ'] =  train['Ток МШЦ 1 А']
    train['Ток МПСИ'] =  train['Ток МПСИ 1 А']
    train['Возврат руды МПСИ'] = train['Возврат руды МПСИ 1 т/ч']
    train['Исходное питание МПСИ'] = train['Исходное питание МПСИ 1 т/ч']
    train['Расход воды МПСИ PV'] = train['Расход воды МПСИ 1 PV м3/ч']
    train['Расход воды МПСИ SP'] = train['Расход воды МПСИ 1 SP м3/ч']
    train['Расход воды МПСИ CV'] = train['Расход воды МПСИ 1 CV %']
    train['факт соотношение руда/вода МПСИ'] =train['факт соотношение руда/вода МПСИ 1']
    train['Давление на подшипник МПСИ загрузка'] =  train['Давление на подшипник МПСИ 1 загрузка Бар']
    train['Давление на подшипник МПСИ разгрузка'] =train['Давление на подшипник МПСИ 1 разгрузка Бар']
    train['Поток'] = train['Поток 1 л/мин']

    window = 10

    train['поток_rolling_10_mean'] = train['Поток 1 л/мин'].rolling(window=window).mean()
    train['поток_rolling_10_std'] = train['Поток 1 л/мин'].rolling(window=window).std()
    train['поток_rolling_10_min'] = train['Поток 1 л/мин'].rolling(window=window).min()
    train['поток_rolling_10_max'] = train['Поток 1 л/мин'].rolling(window=window).max()
    train['поток_rolling_10_median'] = train['Поток 1 л/мин'].rolling(window=window).median()
    train['поток_rolling_10_sum'] = train['Поток 1 л/мин'].rolling(window=window).sum()
    
    window = 20

    train['поток_rolling_20_mean'] = train['Поток 1 л/мин'].rolling(window=window).mean()
    train['поток_rolling_20_std'] = train['Поток 1 л/мин'].rolling(window=window).std()
    train['поток_rolling_20_min'] = train['Поток 1 л/мин'].rolling(window=window).min()
    train['поток_rolling_20_max'] = train['Поток 1 л/мин'].rolling(window=window).max()
    train['поток_rolling_20_median'] = train['Поток 1 л/мин'].rolling(window=window).median()
    train['поток_rolling_20_sum'] = train['Поток 1 л/мин'].rolling(window=window).sum()

    window = 30

    train['поток_rolling_30_mean'] = train['Поток 1 л/мин'].rolling(window=window).mean()
    train['поток_rolling_30_std'] = train['Поток 1 л/мин'].rolling(window=window).std()
    train['поток_rolling_30_min'] = train['Поток 1 л/мин'].rolling(window=window).min()
    train['поток_rolling_30_max'] = train['Поток 1 л/мин'].rolling(window=window).max()
    train['поток_rolling_30_median'] = train['Поток 1 л/мин'].rolling(window=window).median()
    train['поток_rolling_30_sum'] = train['Поток 1 л/мин'].rolling(window=window).sum()

    train['Cu'] = train['Cu_1']
    train['Zn'] = train['Zn_1']
    train['S'] = train['S_1']

    train['Переработка_руды_СМТ'] = train['Переработка_руды_СМТ_1']
    train['Переработка_руды_ВМТ'] = train['Переработка_руды_ВМТ_1']
    train['Гранулометрия в сливе'] = train['Гранулометрия_1']
    train['Влажность'] = train['Влажность_1']

    #train['Температура масла основной маслостанции подача МПСИ'] = train['Температура масла основной маслостанции подача МПСИ 1']
    #train['Температура масла основной маслостанции слив МПСИ'] = train['Температура масла основной маслостанции слив МПСИ 1']
    #train['Температура масла маслостанции электродвигатель МПСИ'] = train['Температура масла маслостанции электродвигатель МПСИ 1']
    #train['Температура масла редуктора МПСИ'] = train['Температура масла редуктора МПСИ 1']
    #train['Температура масла основной маслостанции подача МШЦ'] = train['Температура масла основной маслостанции подача МШЦ 1']
    #train['Температура масла основной маслостанции слив МШЦ'] = train['Температура масла основной маслостанции слив МШЦ 1']
    #train['Температура масла маслостанции электродвигатель МШЦ'] = train['Температура масла маслостанции электродвигатель МШЦ 1']
    #train['Температура масла редуктора МШЦ'] = train['Температура масла редуктора МШЦ 1']
    #train['Расход извести МШЦ'] = train['Расход извести МШЦ 1 л/ч']
    #train['Уровень в зумпфе'] = train['Уровень в зумпфе 1 %']
    train['Обороты насоса'] = train['Обороты насоса 1 1 %']
    train['Давление в ГЦ насоса'] = train['Давление в ГЦ насоса 1 1 Бар']
    #train['Плотность слива ГЦ'] = train['Плотность слива ГЦ 1 кг/л']

    train.dropna( inplace=True)

    X_test_f = train[BASE_FEATURES]
    
    #Создаю фичи для обучения (Второй участок)

    train['Поток_×_Мощность'] = train['Поток 2 л/мин'] * train['Мощность МПСИ 2 кВт']
    train['Поток_×_Ток'] = train['Поток 2 л/мин'] * train['Ток МШЦ 2 А']
    train['Поток_×_Питание МПСИ 2'] = train['Поток 2 л/мин'] * train['Исходное питание МПСИ 2 т/ч']
    train['Поток_×_Расход воды МПСИ 2 PV'] = train['Поток 2 л/мин'] * train['Расход воды МПСИ 2 PV м3/ч']
    train['Поток_×_Расход воды МПСИ 2 SP'] = train['Поток 2 л/мин'] * train['Расход воды МПСИ 2 SP м3/ч']
    train['Поток_×_Расход воды МПСИ 2 CV'] = train['Поток 2 л/мин'] * train['Расход воды МПСИ 2 CV %']
    train['Поток_×_Давление на подшипник МПСИ 2 загрузка'] = train['Поток 2 л/мин'] * train['Давление на подшипник МПСИ 2 загрузка Бар']
    train['Поток_×_Давление на подшипник МПСИ 2 разгрузка'] = train['Поток 2 л/мин'] * train['Давление на подшипник МПСИ 2 разгрузка Бар']
    train['sin_поток'] = np.sin(train['Поток 2 л/мин'])

    train['cos_поток'] = np.cos(train['Поток 2 л/мин'])
    train['tg_поток'] = np.sin(train['Поток 2 л/мин'])/np.cos(train['Поток 2 л/мин'])

    train['поток_квадрат'] = train['Поток 2 л/мин'] ** 2
    train['поток_логарифм'] = np.log(train['Поток 2 л/мин'] + 1e-6)
    train['поток_лаг_1'] = train['Поток 2 л/мин'].shift(1)
    train['поток_лаг_2'] = train['Поток 2 л/мин'].shift(2)
    train['поток_лаг_3'] = train['Поток 2 л/мин'].shift(3)
    train['поток_лаг_4'] = train['Поток 2 л/мин'].shift(4)
    train['поток_лаг_5'] = train['Поток 2 л/мин'].shift(5)
    train['поток_лаг_6'] = train['Поток 2 л/мин'].shift(6)
    train['поток_лаг_7'] = train['Поток 2 л/мин'].shift(7)
    train['поток_лаг_8'] = train['Поток 2 л/мин'].shift(8)
    train['поток_лаг_9'] = train['Поток 2 л/мин'].shift(9)
    train['поток_лаг_10'] = train['Поток 2 л/мин'].shift(10)
    train['поток_лаг_11'] = train['Поток 2 л/мин'].shift(11)
    train['поток_лаг_12'] = train['Поток 2 л/мин'].shift(12)
    train['поток_лаг_13'] = train['Поток 2 л/мин'].shift(13)
    train['поток_лаг_14'] = train['Поток 2 л/мин'].shift(14)
    train['поток_лаг_15'] = train['Поток 2 л/мин'].shift(15)
    train['поток_лаг_16'] = train['Поток 2 л/мин'].shift(16)

    train['Мощность МПСИ'] =  train['Мощность МПСИ 2 кВт']
    train['Общее питание МПСИ'] = train['Общее питание МПСИ 2 т/ч']
    train['Мощность МШЦ'] =  train['Мощность МШЦ 2 кВт']
    train['Ток МШЦ'] =  train['Ток МШЦ 2 А']
    train['Ток МПСИ'] =  train['Ток МПСИ 2 А']
    train['Возврат руды МПСИ'] = train['Возврат руды МПСИ 2 т/ч']
    train['Исходное питание МПСИ'] = train['Исходное питание МПСИ 2 т/ч']
    train['Расход воды МПСИ PV'] = train['Расход воды МПСИ 2 PV м3/ч']
    train['Расход воды МПСИ SP'] = train['Расход воды МПСИ 2 SP м3/ч']
    train['Расход воды МПСИ CV'] = train['Расход воды МПСИ 2 CV %']
    train['факт соотношение руда/вода МПСИ'] =train['факт соотношение руда/вода МПСИ 2']
    train['Давление на подшипник МПСИ загрузка'] =  train['Давление на подшипник МПСИ 2 загрузка Бар']
    train['Давление на подшипник МПСИ разгрузка'] =train['Давление на подшипник МПСИ 2 разгрузка Бар']
    train['Поток'] = train['Поток 2 л/мин']

    window = 10

    train['поток_rolling_10_mean'] = train['Поток 2 л/мин'].rolling(window=window).mean()
    train['поток_rolling_10_std'] = train['Поток 2 л/мин'].rolling(window=window).std()
    train['поток_rolling_10_min'] = train['Поток 2 л/мин'].rolling(window=window).min()
    train['поток_rolling_10_max'] = train['Поток 2 л/мин'].rolling(window=window).max()
    train['поток_rolling_10_median'] = train['Поток 2 л/мин'].rolling(window=window).median()
    train['поток_rolling_10_sum'] = train['Поток 2 л/мин'].rolling(window=window).sum()
    
    window = 20

    train['поток_rolling_20_mean'] = train['Поток 2 л/мин'].rolling(window=window).mean()
    train['поток_rolling_20_std'] = train['Поток 2 л/мин'].rolling(window=window).std()
    train['поток_rolling_20_min'] = train['Поток 2 л/мин'].rolling(window=window).min()
    train['поток_rolling_20_max'] = train['Поток 2 л/мин'].rolling(window=window).max()
    train['поток_rolling_20_median'] = train['Поток 2 л/мин'].rolling(window=window).median()
    train['поток_rolling_20_sum'] = train['Поток 2 л/мин'].rolling(window=window).sum()

    window = 30

    train['поток_rolling_30_mean'] = train['Поток 2 л/мин'].rolling(window=window).mean()
    train['поток_rolling_30_std'] = train['Поток 2 л/мин'].rolling(window=window).std()
    train['поток_rolling_30_min'] = train['Поток 2 л/мин'].rolling(window=window).min()
    train['поток_rolling_30_max'] = train['Поток 2 л/мин'].rolling(window=window).max()
    train['поток_rolling_30_median'] = train['Поток 2 л/мин'].rolling(window=window).median()
    train['поток_rolling_30_sum'] = train['Поток 2 л/мин'].rolling(window=window).sum()

    train['Cu'] = train['Cu_2']
    train['Zn'] = train['Zn_2']
    train['S'] = train['S_2']

    train['Переработка_руды_СМТ'] = train['Переработка_руды_СМТ_2']
    train['Переработка_руды_ВМТ'] = train['Переработка_руды_ВМТ_2']
    train['Гранулометрия в сливе'] = train['Гранулометрия_2']
    train['Влажность'] = train['Влажность_2']

    #train['Температура масла основной маслостанции подача МПСИ'] = train['Температура масла основной маслостанции подача МПСИ 2']
    #train['Температура масла основной маслостанции слив МПСИ'] = train['Температура масла основной маслостанции слив МПСИ 2']
    #train['Температура масла маслостанции электродвигатель МПСИ'] = train['Температура масла маслостанции электродвигатель МПСИ 2']
    #train['Температура масла редуктора МПСИ'] = train['Температура масла редуктора МПСИ 1']
    #train['Температура масла основной маслостанции подача МШЦ'] = train['Температура масла основной маслостанции подача МШЦ 2']
    #train['Температура масла основной маслостанции слив МШЦ'] = train['Температура масла основной маслостанции слив МШЦ 2']
    #train['Температура масла маслостанции электродвигатель МШЦ'] = train['Температура масла маслостанции электродвигатель МШЦ 2']
    #train['Температура масла редуктора МШЦ'] = train['Температура масла редуктора МШЦ 2']
    #train['Расход извести МШЦ'] = train['Расход извести МШЦ 2 л/ч']
    #train['Уровень в зумпфе'] = train['Уровень в зумпфе 2 %']
    train['Обороты насоса'] = train['Обороты насоса 2 1 %']
    train['Давление в ГЦ насоса'] = train['Давление в ГЦ насоса 2 1 Бар']
    #train['Плотность слива ГЦ'] = train['Плотность слива ГЦ 2 кг/л']

    train.dropna( inplace=True)
    X_train = train[BASE_FEATURES]
    y_train = train[TARGET]
    return X_train,y_train,X_test_f


def obj_c(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10),
        'verbose': False
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    return mean_squared_error(y_test_f, model.predict(X_test_f))


def obj_x(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10),
        'objective': 'reg:squarederror',
        'verbose':False
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return mean_squared_error(y_test_f, model.predict(X_test_f))

def obj_l(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10),
        'objective': 'regression',
        'verbose':0
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return mean_squared_error(y_test_f, model.predict(X_test_f))



def tune_models(obj_c,obj_x,obj_l):
    study = optuna.create_study(direction='minimize')
    study.optimize(obj_c, n_trials=50)
    params_c = study.best_params

    study = optuna.create_study(direction='minimize')
    study.optimize(obj_x, n_trials=50)
    params_x = study.best_params

    study = optuna.create_study(direction='minimize')
    study.optimize(obj_l, n_trials=50)
    params_l = study.best_params

    return params_c,params_l,params_x

#Сама модель!
def model(X_train,y_train,X_test_f,params_l,params_x,params_c):  

    X_train_cb1,y_train_cb1 = resample(X_train,y_train,replace=True)
    X_train_xgb1,y_train_xgb1 = resample(X_train,y_train,replace=True)
    X_train_cb2,y_train_cb2 = resample(X_train,y_train,replace=True)

    light = LGBMRegressor(**params_l)
    xgb = XGBRegressor(**params_x)
    cat = CatBoostRegressor(**params_c)

    light.fit(X_train_cb1, y_train_cb1)
    cat.fit(X_train_cb2, y_train_cb2)
    xgb.fit(X_train_xgb1, y_train_xgb1)
    pred_light1 = light.predict(X_test_f)
    pred_light2 = cat.predict(X_test_f)
    pred_light3 = xgb.predict(X_test_f)
    y_pred= np.mean([pred_light3,pred_light2,pred_light1], axis=0)
    return y_pred

X_train,y_train,X_test_f = preprocess(BASE_PATH,',')
params_l,params_x,params_c = tune_models(obj_c,obj_l,obj_x)
prediction = model(X_train,y_train,X_test_f,params_l,params_x,params_c )