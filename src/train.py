"""
Main file for Model Training
"""
# Import Core Libraries
import os
import sys
sys.path.append("../src")

# Setup Tracking Locations
os.environ["METAFLOW_DATASTORE_SYSROOT_LOCAL"] = "../logs/metaflow"
MLFLOW_TRACKING_URI = "../logs/mlflow/mlruns"
OUTPUTS_LOCATION = "../models/outputs"

# Import Libraries
import pandas as pd
import numpy as np
import preprocess as pp
import model_wide as m
import utils as u
from metaflow import FlowSpec, step, IncludeFile, metaflow_config, Flow, Parameter

# Set up the Experiment
import mlflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set up Logging
from logger import get_logger
loggy = get_logger(__name__)

class TensorflowPipeline(FlowSpec):
    """
    A flow to train a tensorflow machine learning model.
    """

    learning_rate = Parameter('learning_rate',
                              help='Learning rate',
                              required = False,
                              default=0.01)

    epochs = Parameter('epochs',
                       help='Epochs',
                       required = False,
                       default=5)

    exp_name = Parameter('experiment_name',
                         help='Name for the experiment in MLFlow',
                         required = False,
                         default='TensorflowPipeline')


    @step
    def start(self):
        """
        The start step:
        1) Loads the movie metadata into pandas dataframe.
        2) Preprocesses the Data
        3) Launches the next step.

        """
        loggy.info("Metaflow Run Start")
        import pandas
        from io import StringIO

        # Set the Experiment
        mlflow.set_experiment(self.exp_name)
        self.experiment_id = mlflow.get_experiment_by_name(self.exp_name).experiment_id
        with mlflow.start_run():
            self.run_id = mlflow.active_run().info.run_id
            loggy.info(f"Running MLFlow logs in {str(self.run_id)}")
            self.outputs_loc = OUTPUTS_LOCATION + f'/{self.run_id}'
            os.makedirs(self.outputs_loc,exist_ok = True)

        # Load the data set into a pandas dataframe.
        self.clean_df = pandas.read_csv('../data/consumer_complaints_with_narrative.csv')

        # Generate Sample and Log It
        df = self.clean_df.sample(frac=.05, random_state=42)
        df.to_html(self.outputs_loc+"/data_sample.html")

        # Divide Dataframe
        self.feature_names = ["product", "sub_product", "issue", "sub_issue", "state", "zip_code", 
                              "company", "company_response", "timely_response", 
                              "consumer_disputed", "consumer_complaint_narrative"]
        self.features_df = self.clean_df[self.feature_names].copy()


        self.one_hot_features = ['product', 'sub_product', 'company_response', 'state', 'issue']
        self.one_hot_df = self.clean_df[self.one_hot_features].copy()


        self.numeric_features = ['zip_code']
        self.numeric_features_df = self.clean_df.copy()


        self.text_features = ['consumer_complaint_narrative']
        self.text_features_df = self.clean_df[self.text_features].copy()


        # Next Step we want to calculate Statistics for the DataFrame
        self.next(self.compute_statistics, self.process_one_hot, 
                  self.process_numeric_features,self.process_text_features,self.process_label)
    
    @step
    def compute_statistics(self):
        """
        This step computes and stores statistics about the dataframe
        """
        import mlflow
        mlflow.set_experiment(self.exp_name)
        mlflow.start_run(run_id=self.run_id)

        run = Flow('TensorflowPipeline')
        file_location = self.outputs_loc+"/data_statistics_report.html"
        self.compute_statistics_result = pp.generate_pandas_profile(self.clean_df, file_location)
        self.next(self.join_process)
    
    @step
    def process_one_hot(self):
        """
        This step does one-hot encoding.
        """
        import numpy as np
        self.one_hot_dict = {}
        for feature in self.one_hot_df.columns:
            self.one_hot_dict[feature] = self.one_hot_df[feature].nunique()
            self.one_hot_df[feature] = self.one_hot_df[feature].astype("category").cat.codes

        self.next(self.join_process)

    @step
    def process_numeric_features(self):
        """
        This step processes numeric features. 
        Manages the zip_code
        """
        self.numeric_features_df = pp.change_zipcode_col(self.numeric_features_df)
        self.next(self.join_process)

    @step
    def process_text_features(self):
        """
        This step processes Text Features
        """
        self.text_features_df = self.text_features_df
        self.text_features = self.text_features

        self.one_hot_features = self.one_hot_features
        self.numeric_features = self.numeric_features

        self.run_id = self.run_id
        self.outputs_loc = self.outputs_loc
        self.next(self.join_process)
    
    @step
    def process_label(self):
        """
        This step maps the label from Yes -> 1 and No -> 0
        """
        import numpy as np
        self.y = np.asarray(self.clean_df["consumer_disputed"], dtype=np.uint8).reshape(-1)
        self.next(self.join_process)

    @step
    def join_process(self,inputs):
        """
        Joins the result of the four branchs
        1) process_one_hot
        2) process_numeric_features
        3) process_text_features
        4) process_label

        """
        from sklearn.model_selection import train_test_split
        
        
        # Get Final Dataframe
        self.raw_x = pd.concat([inputs.process_text_features.text_features_df,
                                inputs.process_one_hot.one_hot_df,
                                inputs.process_numeric_features.numeric_features_df['zip_code']],axis=1)
        assert self.raw_x.shape[1] == 7

        # Get Train/Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.raw_x, inputs.process_label.y, 
                                                                                test_size=0.2, random_state=42)
         
        # One Hot
        self.one_hot_dict = inputs.process_one_hot.one_hot_dict
        self.one_hot_train = pp.get_one_hot_vector(self.X_train, inputs.process_text_features.one_hot_features)
        self.one_hot_test = pp.get_one_hot_vector(self.X_test, inputs.process_text_features.one_hot_features)

        # Numeric
        self.numeric_train = [self.X_train['zip_code'].values]
        self.numeric_test = [self.X_test['zip_code'].values]

        # Text
        self.embedding_train = pp.get_text_embedding(self.X_train, inputs.process_text_features.text_features)
        self.embedding_test = pp.get_text_embedding(self.X_test, inputs.process_text_features.text_features)

        # Create final dataframe
        self.Xtrain = self.one_hot_train + self.numeric_train + self.embedding_train
        self.Xtest = self.one_hot_test + self.numeric_test + self.embedding_test

        self.ytrain = self.y_train
        self.ytest = self.y_test

        self.run_id = inputs.process_text_features.run_id
        self.outputs_loc = inputs.process_text_features.outputs_loc
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Step that gets the tensforflow model and performs the tests
        """
        import tensorflow as tf
        import tensorflow_hub as hub
        import mlflow
        import mlflow.tensorflow
        mlflow.set_experiment(self.exp_name)
        mlflow.tensorflow.autolog()

        # Get Data
        self.Xtest = self.Xtest
        self.ytest = self.ytest

        with mlflow.start_run(run_id = self.run_id):
    
            # Log the Data Dict
            mlflow.log_params(self.one_hot_dict)

            # Get the Model
            model = m.get_model(num_params = self.one_hot_dict,
                                learning_rate = self.learning_rate, 
                                show_summary=True,
                                plot_image = self.outputs_loc+'/model_image.png')

            # Define Callbacks
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("../models/TensorflowPipeline.h5",
                                                            save_best_only=True)
                                                            
            tensorboard_cb = tf.keras.callbacks.TensorBoard(u.get_run_logdir())

            # Train Model
            model.fit(x=self.Xtrain, y=self.ytrain ,batch_size = 32,validation_split=0.2, epochs=self.epochs,
                    callbacks=[checkpoint_cb,tensorboard_cb])

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """
        Function that evaluates the model
        """
        import tensorflow as tf
        import tensorflow_hub as hub
        import mlflow
        mlflow.set_experiment(self.exp_name)
        mlflow.start_run(run_id=self.run_id)
        self.outputs_loc = self.outputs_loc

        model = tf.keras.models.load_model("../models/TensorflowPipeline.h5", custom_objects={'KerasLayer':hub.KerasLayer})
        scores = model.evaluate(self.Xtest, self.ytest)
        self.eval_metrics = {'test_'+metric_n:score for metric_n, score in zip(model.metrics_names,scores)}

        # Log Evaluation
        mlflow.log_metrics(self.eval_metrics)
        self.next(self.end)


    @step
    def end(self):
        """
        End the flow.
        """
        import mlflow
        mlflow.set_experiment(self.exp_name)
        mlflow.start_run(run_id=self.run_id)

        mlflow.log_artifacts(self.outputs_loc)
        loggy.info("Metaflow Run Success")


if __name__ == '__main__':
    TensorflowPipeline()
