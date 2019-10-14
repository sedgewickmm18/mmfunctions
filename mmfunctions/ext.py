class BaseWindowedFeatureExtractor(BaseTransformer):
    '''
    Base class to chop a multidimensional time series into digestible, overlapping windows and
    then apply a nifty algorithm to extract and score features. Since the length of the overlapping
    windows sequence and hence the sequence of scores is shorter than the original
    time series stretch the result sequence by interpolating.
    '''

    # input items - columns to extract features from
    # feature     - extracted features (output), for example closest centroid for proximity based algos or spectral densities 
    # anomalyscore - anomaly score column name (output)
    # windowsize  - size of windows after chopping the time series into pieces
    # windowoverlap - size of the overlap, if 0 relative to windowsize
    # nochop      - use an integrate algorithm like spectrogram instead of 'manually' chopping the time series data
    def __init__(self, input_items, features, anomalyscore = 'anomalyscore',
                 windowsize = 24, windowoverlap = 0, windowfunction = 'none', nochop = False, **kwargs):
        self.input_items = input_items
        self.feature = features
        self.anomalyscore = anomalyscore
        self.windowsize = windowsize
        self.windowoverlap = windowoverlap
        self.windowfunction = windowfunction
        self.nochop = nochop
        super().__init__()

    # returns output features (multidimensional, time), score (time)
    #  this is to be implemented by the subclass (or passed to this k-means proximity example
    def scorer(self, inputData, slices = 'nan'):

        if math.isnan(slices):
            return 'nan'

        # construct PyOD K-Means proximity measure
        cblofwin = CBLOF(n_clusters=40, n_jobs=-1)
        cblofwin.fit(slices)
        preddec = cblofwin.decision_function(slices)

        # proximity measures do not really extract features
        #   they return a measure how far a datapoint is from the center of a centroid/cluster
        return 'nan', preddec

    def extractPerEntity(self, dfent):
        # ToDo get timestamp handling right - hardcoded evt_timestamp for now
        dfent.set_index('evt_timestamp')        

        # missing data - data imputation ?
        dfe = dfent.interpolate(method='time')
        
        # get multi-dimensional numpy array from the input items
        inputData = dfe[[self.input_items]].fillna(0).to_numpy().reshape(-1,)

        if nochop:
            output, score = scorer(inputData)

        else:
            # provide windows only in time direction
            step = np.linspace(0, inputData.ndim, 0)
            step[0] = 1
            slices = skiutil.view_as_windows(inputData,
                                        window_shape=(self.windowsize,), step = step)
        
            # Apply hanning window function
            if self.windowfunction == 'hanning':
                slices = slices * np.hanning(self.windowsize + 1)[:-1]

            score = scorer(inputData, slices = slices)

            # rescale score to fit to the original timescale
            timesI = np.linspace(0, len(dfe.index)-1, len(dfe.index))
            timesTS = np.linspace(0, slices.size-1, slices.size)

            # timesTS does not exist here, how should I compute it
            dfe[self.anomalyscore] = np.interp(timesI, timesTS, score)

        return (dfe)
    

    def execute(self, df, func = func):
        # sanity check
        if self.input_items.size < 1:
            return 'nan'

        # per entity id
        entities = np.unique(df.deviceid.values)

        for entity in entities:

            # ToDo get deviceid handling right - using deviceid for now
            dfe = extractPerEntity(df.where(df.deviceid == entity).dropna(how='all'))

            #.....
            

        # how to integrate the dfe slices after interpolation in time and projecting to each entity id into the overall df


    @classmethod
    def build_ui(cls):
        '''
        Registration metadata
        '''
        inputs = []
        inputs.append(UIMultiItem(name='input_items',
                                  datatype=float,
                                  description='Input data items'
                                  ))
        inputs.append(UISingleItem(name='windowsize',
                                   datatype=int,
                                   description='Window Size'
                                   ))
        inputs.append(UISingleItem(name='windowoverlap',
                                   datatype=int,
                                   description='Window Overlap'
                                   ))
        inputs.append(UISingleItem(name='nochop',
                                   datatype=bool,
                                   description='Function does not need excplicit chopping of the timeseries data'
                                   ))
        
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='anomalyscore',
                                           datatype=float,
                                           description='Anomaly Score - Column Name'
                                           ))
        #self.feature = features - not reflected

        return (inputs, o

    @classmethod
    def build_ui(cls):
        '''
        Registration metadata
        '''
        inputs = []
        inputs.append(UIMultiItem(name='input_items',
                                  datatype=float,
                                  description='Input data items'
                                  ))
        inputs.append(UISingleItem(name='windowsize',
                                   datatype=int,
                                   description='Window Size'
                                   ))
        inputs.append(UISingleItem(name='windowoverlap',
                                   datatype=int,
                                   description='Window Overlap'
                                   ))
        inputs.append(UISingleItem(name='nochop',
                                   datatype=bool,
                                   description='Function does not need excplicit chopping of the timeseries data'
                                   ))
        
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='anomalyscore',
                                           datatype=float,
                                           description='Anomaly Score - Column Name'
                                           ))
        #self.feature = features - not reflected

        return (inputs, outputs)
