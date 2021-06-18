# 모델링
skf = StratifiedKFold(n_splits = 5, random_state = 2021, shuffle = True)

RF_models={}

for n_fold, (train_index, val_index) in enumerate(skf.split(train, target)):
        trn_x, val_x = train.iloc[train_index], train.iloc[val_index]
        trn_y, val_y = target.iloc[train_index], target.iloc[val_index]
        model = RandomForestClassifier(1000, random_state=2021)
        model.fit(trn_x, trn_y)
        pred = model.predict_proba(val_x)
        print(roc_auc_score(val_y, pred[:,1:]))
        RF_models[n_fold] = model


pred_list = []
for fold in range(5):
    pred = RF_models[fold].predict_proba(test)[:, 1]
    pred_list.append(pred)

preds = np.asarray(pred_list).mean(axis=0)
