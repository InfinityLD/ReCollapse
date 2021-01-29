import numpy as np
import pandas as pd
import os
import torch
import pickle
import time

from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchkeras import Model,summary
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.models.fm import FactorizationMachineModel
from src.models.deepfm import DeepFMModel
from src.utils.parameters import arg_parser

args = arg_parser()


def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')
    dfresult['Age'] = pd.cut(dfresult['Age'], bins=10, labels=list(range(10)))

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = pd.cut(dfdata['Fare'], bins=10, labels=list(range(10)))

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    for col in dfresult.columns:
        lbe = LabelEncoder()
        dfresult[col] = lbe.fit_transform(dfresult[col])
        with open(os.path.join(args.data_path, f'{col}_lbe.pickle'), 'wb') as fp:
            pickle.dump(lbe, fp)

    nunique_list = dfresult.nunique().tolist()

    dfresult['type'] = dfdata['type']
    dfresult['Survived'] = dfdata['Survived']

    return dfresult, nunique_list


def train(model, data_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.
    tq = tqdm(data_loader, smoothing=0, mininterval=1.0)

    for i, (features, labels) in enumerate(tq, 1):
        features, labels = features.to(device), labels.to(device)
        y_pred = model(features)
        loss = criterion(y_pred, labels.float())

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % args.log_interval == 0:
            tq.set_postfix(loss=total_loss / args.log_interval)
            total_loss = 0


def test(model, data_loader, device):
    trues, preds = [], []
    model.eval()

    with torch.no_grad():
        for features, labels in tqdm(data_loader, smoothing=0, mininterval=1.0):
            features, labels = features.to(device), labels.to(device)
            y_pred = model(features)
            preds.extend(y_pred.tolist())
            trues.extend(labels.tolist())

    return roc_auc_score(trues, preds)


def main():
    data_path = args.data_path

    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

    train_df['type'] = 1
    test_df['type'] = 0

    df = pd.concat([train_df, test_df], ignore_index=True)

    df, field_dims = preprocessing(df)

    train_df = df[df['type'] == 1]
    test_df = df[df['type'] == 0]

    x_train = train_df[[col for col in train_df.columns if col not in ['Survived', 'type']]].values
    y_train = train_df[['Survived']].values.ravel()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1024)

    x_test = test_df.drop(['Survived', 'type'], axis=1).values

    print("x_train.shape =", x_train.shape)
    print("x_val.shape =", x_val.shape)
    print("x_test.shape =", x_test.shape)
    print("y_train.shape =", y_train.shape)
    print("y_val.shape =", y_val.shape)

    X_train_tensor = torch.tensor(x_train).long()
    y_train_tensor = torch.tensor(y_train).float()
    X_val_tensor = torch.tensor(x_val).long()
    y_val_tensor = torch.tensor(y_val).float()
    X_test_tensor = torch.tensor(x_test).long()

    dl_train = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), shuffle=True, batch_size=8)
    dl_val = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), shuffle=True, batch_size=8)
    dl_test = DataLoader(TensorDataset(X_test_tensor))

    device = torch.device(args.device)
    # fm = FactorizationMachineModel(field_dims=field_dims, embed_dim=16)
    fm = DeepFMModel(field_dims=field_dims, embed_dim=16)
    summary(fm, input_shape=(15,), input_dtype=torch.long)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params=fm.parameters(), lr=0.001, weight_decay=1e-6)

    writer = SummaryWriter('../data/tensorboard/')
    for e in range(args.epochs):
        train(fm, dl_train, device, criterion, optimizer)
        train_auc_score = test(fm, dl_train, device)
        val_auc_score = test(fm, dl_val, device)
        writer.add_scalars('AUC', {'Train': train_auc_score,
                                   'Val': val_auc_score}, e)
        print(f'\nepoch {e}\t train auc: {round(train_auc_score, 4)}\tvalidation auc: {round(val_auc_score, 4)}')

    writer.add_graph(fm, input_to_model=torch.rand(1, 15).long())
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
