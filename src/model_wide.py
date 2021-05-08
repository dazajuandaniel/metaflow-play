"""
File that holds the Model
"""
import tensorflow as tf
import tensorflow_hub as hub

# Logger
import utils as u
from logger import get_logger
loggy = get_logger(__name__)

def get_model(num_params, show_summary=True):
    """
    Function defines a Keras model and returns the model as Keras object
    Args:
        num_params
        show_summary
    """    
    loggy.info(str(num_params))
    
    # one-hot categorical features
    num_products = num_params['product']
    num_sub_products = num_params['sub_product']
    num_company_responses = num_params['company_response']
    num_states = num_params['state']
    num_issues = num_params['issue']

    input_product = tf.keras.Input(shape=(num_products,), name="product_xf")
    input_sub_product = tf.keras.Input(shape=(num_sub_products,), name="sub_product_xf")
    input_company_response = tf.keras.Input(shape=(num_company_responses,), name="company_response_xf")
    input_state = tf.keras.Input(shape=(num_states,), name="state_xf")
    input_issue = tf.keras.Input(shape=(num_issues,), name="issue_xf")
    
    # numeric features
    input_zip_code = tf.keras.Input(shape=(1,), name="zip_code_xf")

    # text features
    input_narrative = tf.keras.Input(shape=(1,), name="narrative_xf", dtype=tf.string)

    # embed text features
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.KerasLayer(module_url)
    reshaped_narrative = tf.reshape(input_narrative, [-1])
    embed_narrative = embed(reshaped_narrative) 
    deep_ff = tf.keras.layers.Reshape((512, ), input_shape=(1, 512))(embed_narrative)
    
    deep = tf.keras.layers.Dense(256, activation='relu')(deep_ff)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)

    wide_ff = tf.keras.layers.concatenate([input_product, input_sub_product, 
                                           input_company_response, 
                                           input_state, input_issue, 
                                           input_zip_code])
    
    wide = tf.keras.layers.Dense(16, activation='relu')(wide_ff)


    both = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(both) 

    _inputs = [input_product, input_sub_product, input_company_response,  
               input_state, input_issue, input_zip_code, input_narrative]

    keras_model = tf.keras.models.Model(_inputs, output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.compile(optimizer=optimizer,
                        loss='binary_crossentropy',  
                        metrics=[tf.keras.metrics.BinaryAccuracy(),
                                 tf.keras.metrics.TruePositives(),
                                 tf.keras.metrics.Precision(),
                                 tf.keras.metrics.AUC(),
                                 tf.keras.metrics.Accuracy(),
                                 tf.keras.metrics.Recall()])
    if show_summary:
        stringlist = []
        keras_model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = " \n".join(stringlist)
        loggy.info(short_model_summary)

    return keras_model