import lightgbm as lgb
from sklearn.metrics import mean_squared_error

maes, rmses = [], []
for fold, (tr, va) in enumerate(tscv.split(X)):
    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr], y[va]

    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)

    params = dict(
        objective="regression",
        metric=["l2","l1"],
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=42
    )
    model = lgb.train(params, dtr, num_boost_round=2000,
                      valid_sets=[dtr, dva], valid_names=["train","valid"],
                      early_stopping_rounds=100, verbose_eval=False)
    pred = model.predict(Xva, num_iteration=model.best_iteration)
    mae = mean_absolute_error(yva, pred)
    rmse = mean_squared_error(yva, pred, squared=False)
    maes.append(mae); rmses.append(rmse)
    print(f"Fold {fold+1}: MAE={mae:.6f} RMSE={rmse:.6f}")

print("Avg MAE:", np.mean(maes), "Avg RMSE:", np.mean(rmses))
importance = pd.Series(model.feature_importance(importance_type="gain"), index=feat_cols)
print("\nLightGBM 重要度(按gain):")
print(importance.sort_values(ascending=False).head(15))