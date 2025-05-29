# О решении

Мы решаем две задачи: предсказание гранулометрии по показаниям счетчиков и анализам в данный момент, а также прогнозирование гранулометрии на будущее (на следующие 60 минут) по данным прошлого.

Для запуска решения требуется скачать датасет: [тык](https://drive.google.com/file/d/1xkll8qfUENHlHUMlo-29sd5Zw9pIwOHB/view?usp=sharing)

## Регрессия

- Модели: catboost, xgboost, lightgbm
- Фичи: исходные, лаги, скользящие окна, периодическое кодирование
- Оптимизация: optuna
- Итоговые метрики: ~5 RMSE

Код: [тык](https://github.com/hackathonsrus/LI_misis_kopim_elik_118/blob/main/regressor.ipynb)

## Предсказание будущего

- Модели: lightgbm
- Фичи: исходные, лаги, скользящие окна, отношения
- Оптимизация: optuna
- Итоговые метрики: ~3.6 RMSE

Код: [тык](https://github.com/hackathonsrus/LI_misis_kopim_elik_118/blob/main/forecast_future.ipynb)
