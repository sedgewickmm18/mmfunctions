#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class RobustThreshold(SupervisedLearningTransformer):

    def __init__(self, input_item, output_item1,output_item2,output_item2):
        super().__init__(features=[input_item], targets=[output_item1,output_item2,output_item3])

        self.input_item = input_item
        self.output_item1 = output_item1
        self.output_item2 = output_item2
        self.output_item3 = output_item3

        self.whoami = 'AnomalyThreshold'

#         logger.info(self.whoami + ' from ' + self.input_item + ' quantile threshold ' +  str(self.threshold) +
#                     ' exceeding boolean ' + self.output_item)
        
        
    def execute(self, df):
        # set output columns to zero
        logger.debug('Called ' + self.whoami + ' with columns: ' + str(df.columns))
        df[self.output_item] = 0
        return super().execute(df)
    
    
    def _calc(self, df):
    entity = df.index[0][0]

    # obtain db handler
    db = self.get_db()

    model_name, robust_model, version = self.load_model(suffix=entity)

    feature = df[self.input_item].values

    if robust_model is None and self.auto_train:
        # we don't do that now, the model *has* to be there

    if robust_model is not None:
        self.Min[entity] = robust_model.Min
        df[self.min] = robust_model.Min       # set the min threshold column
        self.Max[entity] = robust_model.Max  
        df[self.max] = robust_model.Min       # set the max threshold column
        df[self.output_item] = (feature < robust_model.Max)& (feature < robust_model.Min)   # extend that to cover Min, too
    else:
        df[self.output_item] = 0

    return df.droplevel(0)

    
    


# In[ ]:


class RobustModel:
    def __init__(self, min, max, median)
    self.Min = -8.07
    self.Max = 7.2
    self.Median = [['-0.275',
 '-0.275',
 '-0.23',
 '-0.45999999999999996',
 '-0.034999999999999996',
 '0.58',
 '0.905',
 '1.69',
 '3.325',
 '4.525',
 '5.55',
 '5.445',
 '5.55',
 '5.79',
 '5.7',
 '5.5',
 '5.475',
 '5.465',
 '5.38',
 '5.21',
 '5.035',
 '4.675',
 '3.4000000000000004',
 '2.4450000000000003',
 '1.545',
 '0.785',
 '-0.525',
 '-1.11',
 '-1.13',
 '-1.23',
 '-1.545',
 '1.09',
 '0.92',
 '0.92',
 '0.995',
 '0.855',
 '1.0150000000000001',
 '0.855',
 '0.87',
 '0.9',
 '0.88',
 '0.805',
 '0.87',
 '0.84',
 '0.72',
 '0.77',
 '0.865',
 '0.78',
 '0.81',
 '0.875',
 '0.785',
 '0.69',
 '0.74',
 '0.76',
 '0.655',
 '0.675',
 '0.6699999999999999',
 '0.705',
 '0.695',
 '0.63',
 '0.615',
 '0.645',
 '0.43',
 '-0.07',
 '-0.47',
 '-0.265',
 '-0.575',
 '-0.705',
 '-0.385',
 '0.6799999999999999',
 '1.98',
 '3.425',
 '5.22',
 '5.77',
 '6.225',
 '6.325',
 '6.355',
 '4.355',
 '3.95',
 '1.44',
 '-0.64',
 '-2.13',
 '-2.755',
 '-4.46',
 '-5.8149999999999995',
 '-6.275',
 '-4.82',
 '-4.27',
 '-4.355',
 '-5.145',
 '-5.32',
 '-4.305',
 '-3.26',
 '-1.3650000000000002',
 '-1.3900000000000001',
 '-2.785',
 '-1.38',
 '-1.455',
 '-0.38',
 '-0.72',
 '-0.9199999999999999',
 '-0.46499999999999997',
 '1.725',
 '1.725',
 '1.415',
 '-0.005',
 '0.435',
 '0.88',
 '0.82',
 '-0.29000000000000004',
 '-0.245',
 '-0.265',
 '-0.255',
 '-0.255',
 '-0.24',
 '-0.255',
 '-0.265',
 '-0.255',
 '-0.20500000000000002',
 '0.8099999999999999',
 '1.59',
 '1.5550000000000002',
 '1.58',
 '1.48',
 '1.54',
 '1.35',
 '1.38',
 '1.3050000000000002',
 '1.26',
 '1.18',
 '1.3050000000000002',
 '1.1',
 '1.1549999999999998',
 '1.14',
 '0.96',
 '1.065',
 '0.805',
 '0.7',
 '0.92',
 '0.925',
 '0.43',
 '0.31',
 '0.345',
 '0.445',
 '0.48',
 '-0.555',
 '-1.1749999999999998',
 '-1.3',
 '-1.36',
 '-1.36',
 '-1.38',
 '-1.4',
 '-1.38',
 '-1.3',
 '-1.295',
 '-0.14500000000000002',
 '-0.015',
 '0.0',
 '-0.005',
 '-0.135',
 '-0.23',
 '-0.245',
 '-0.345',
 '-0.345',
 '-0.485',
 '-0.52',
 '-0.425',
 '-0.42',
 '-0.37',
 '-0.315',
 '-0.075',
 '-0.16499999999999998',
 '-0.545',
 '-0.615',
 '-0.59',
 '-0.45999999999999996',
 '-0.265',
 '0.185',
 '0.445',
 '0.21000000000000002',
 '0.21000000000000002',
 '0.21000000000000002',
 '0.21000000000000002',
 '0.185',
 '0.24',
 '0.24',
 '0.07',
 '-0.1',
 '-0.43999999999999995',
 '-0.28',
 '-0.29500000000000004',
 '-0.42000000000000004',
 '-0.315',
 '-0.26',
 '-0.23',
 '-0.405',
 '-0.44',
 '-0.49',
 '-0.475',
 '-0.49',
 '-0.31',
 '-0.43999999999999995',
 '-0.48',
 '-0.6699999999999999',
 '-0.41',
 '-0.14',
 '-0.16999999999999998',
 '-0.47',
 '-0.6799999999999999',
 '-0.655',
 '-0.735',
 '-0.02',
 '0.26',
 '0.665',
 '0.79',
 '0.77',
 '0.805',
 '0.77',
 '0.765',
 '0.77',
 '0.765',
 '0.63',
 '0.45999999999999996',
 '0.33',
 '0.34',
 '0.34',
 '0.365',
 '0.35',
 '0.405',
 '0.41',
 '0.435',
 '0.43',
 '0.45499999999999996',
 '0.44',
 '0.435',
 '0.44',
 '0.45499999999999996',
 '0.45499999999999996',
 '0.47',
 '0.48',
 '0.48',
 '0.48',
 '0.52',
 '0.255',
 '-1.41',
 '-1.685',
 '-1.55',
 '-0.375',
 '0.445',
 '1.17',
 '0.865',
 '0.19',
 '0.12',
 '0.11',
 '0.11499999999999999',
 '0.11499999999999999',
 '0.11499999999999999',
 '0.12',
 '0.14500000000000002',
 '0.215',
 '0.365',
 '0.29',
 '0.32',
 '0.33',
 '0.33',
 '0.385',
 '0.37',
 '0.35',
 '0.36',
 '0.36',
 '0.365',
 '0.35',
 '0.355',
 '0.4',
 '0.4',
 '0.37',
 '0.33999999999999997',
 '0.385',
 '0.3',
 '0.5549999999999999',
 '0.49',
 '0.37',
 '0.745',
 '0.83',
 '0.795',
 '0.16999999999999998',
 '0.29000000000000004',
 '0.845',
 '0.9199999999999999',
 '0.905',
 '0.005',
 '0.175',
 '0.59',
 '0.72',
 '0.765',
 '-0.009999999999999998',
 '0.035',
 '0.11',
 '0.29500000000000004',
 '0.025',
 '-0.03',
 '-0.33499999999999996',
 '-0.47',
 '-0.38',
 '-0.555',
 '-0.78',
 '-0.995',
 '-1.0350000000000001',
 '-1.05',
 '-1.06',
 '-1.06',
 '-1.045',
 '-1.065',
 '-1.15',
 '-1.1749999999999998',
 '-1.2149999999999999',
 '-1.48',
 '-1.78',
 '-1.96',
 '-2.2350000000000003',
 '-2.49',
 '-2.74',
 '-2.91',
 '-3.14',
 '-3.32',
 '-3.45',
 '-3.6399999999999997',
 '-3.81',
 '-3.89',
 '-4.0',
 '-4.02',
 '-3.94',
 '-2.98']
    return
    


# In[ ]:


@classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze"))

        inputs.append(UISingle(name="threshold", datatype=int,
                               description="Threshold to determine outliers by quantile. Typically set to 0.95", ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="min", datatype=float,
                                           description="Boolean outlier condition"))
        outputs.append(UIFunctionOutSingle(name="max", datatype=float,
                                           description="Boolean outlier condition"))
        outputs.append(UIFunctionOutSingle(name="std_cyce", datatype=float,
                                           description="Boolean outlier condition"))
        outputs.append(UIFunctionOutSingle(name="outlier", datatype=bool,
                                           description="Boolean outlier condition"))
        return (inputs, outputs)

