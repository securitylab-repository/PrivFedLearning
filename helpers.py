import pandas as pd

class Helpers():

    def to_dataframe(self, input):
        """
        Converts input into a dataframe with corresponding output from the
        first value of the input array
            Parameters:
                input (ndarray): features matrix
            Returns:
                df (dataframe): dataframe with inputs and outputs
        """
        df = pd.DataFrame()
        df = pd.concat([pd.DataFrame(input)], axis=1)
        df['Output'] = df[0].map(lambda x: 1 if (x == 1) else 0)
        df.columns = ['Input_0', 'Input_1', 'Input_2', 'Output']
        return df

    def ndarray_to_string(self, ndarray):
        """
        Converts ndaray into a string
            Parameters:
                ndarray (ndarray): the ndarray you wish to convert
            Returns:
                ndarray_str (str): the converted ndarray
        """
        ndarray_list = ndarray.tolist()
        ndarray_str = ''.join(str(e) for e in ndarray_list)
        return ndarray_str

    def ndarray_to_int(self, ndarray):
        """
        Converts ndaray into a list of floats
            Parameters:
                ndarray (ndarray): the ndarray you wish to convert
            Returns:
                ndarray_int (list): the converted ndarray
        """
        ndarray_list = ndarray.tolist()
        ndarray_str = ''.join(str(e) for e in ndarray_list)
        ndarray_int = int(''.join(str(ord(c)) for c in ndarray_str[:2]))
        return ndarray_int

    def df_to_list(self, dataframe):
        """
        Converts dataframe into a list of training data
            Parameters:
                dataframe (DataFrame): the dataframe you wish to convert
            Returns:
                data (list): the converted dataframe
        """
        df = pd.read_csv(dataframe) 
        col_one=df['Input_0'].tolist()
        col_two=df['Input_1'].tolist()
        col_three=df['Input_2'].tolist()
        col_four=df['Output'].tolist()
        data = list(zip(col_one, col_two, col_three, col_four))
        return data
