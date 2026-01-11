# <YOUR_IMPORTS>
import dill
import json
from datetime import datetime
import os
import pandas as pd

path = os.environ.get("PROJECT_PATH", '.')


with open(f'{path}/data/models/cars_pipe_202601110327.pkl', 'rb') as f:
    model = dill.load(f)


list_test = ['7310993818',
             '7313922964',
             '7315173150',
             '7316152972',
             '7316509996' ]

result_df = pd.DataFrame(columns=['car_id','pred'])


def predict():
    import pandas as pd
    for i in list_test:
        with open(f'{path}/data/test/{i}.json') as file:
            form = json.load(file)
        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)
        index = list_test.index(i)
        result_df.loc[index, 'car_id'] = i
        result_df.loc[index, 'pred'] = y[0]
        print(y)
    return result_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime('%Y%m%d%H%M')}.csv', index=False)


if __name__ == '__main__':
    predict()

