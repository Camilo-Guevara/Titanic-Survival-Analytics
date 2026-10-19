from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parent
original=pd.read_csv(p/'data/titanic_original.csv')
processed=pd.read_csv(p/'data/titanic_procesado.csv')
assert original.shape==(891,12)
assert processed.shape==(891,21)
assert original.PassengerId.equals(processed.PassengerId)
assert original.Survived.equals(processed.Survived)
assert original.Age.isna().sum()==177
assert processed.Age.isna().sum()==0
assert original.Cabin.isna().sum()==687
assert processed.Cabin.isna().sum()==687
assert original.Embarked.isna().sum()==2
assert processed.Embarked.isna().sum()==0
print('OK: archivos, integridad, columnas y transformaciones verificadas')
