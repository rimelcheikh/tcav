from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
import keras.backend as K
import pickle

import tcav.model as tcav_model
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.activation_generator as act_gen
import tensorflow as tf
import tcav.utils_plot as utils_plot
from keras.utils import plot_model

from torchvision import models



# Modified version of PublicImageModelWrapper in TCAV's models.py
# This class takes a session which contains the already loaded graph.
# This model also assumes softmax is used with categorical crossentropy.
class CustomPublicImageModelWrapper(tcav_model.ImageModelWrapper):
    def __init__(self, sess, labels, image_shape,
                endpoints_dict, name, image_value_range):
        super(self.__class__, self).__init__(image_shape)
        
        self.sess = sess
        self.labels = tf.io.gfile.GFile(labels).read().splitlines()
        self.model_name = name
        self.image_value_range = image_value_range

        # get endpoint tensors
        self.ends = {'input': endpoints_dict['input_tensor'], 'prediction': endpoints_dict['prediction_tensor']}
        
        self.bottlenecks_tensors = self.get_bottleneck_tensors()
        
        # load the graph from the backend
        graph = tf.compat.v1.get_default_graph()

        # Construct gradient ops.
        with graph.as_default():
            self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None])

            self.pred = tf.expand_dims(self.ends['prediction'][0], 0)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(
                        self.y_input,
                        self.ends['prediction'].get_shape().as_list()[1]),
                    logits=self.pred))
        self._make_gradient_tensors()

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)

    @staticmethod
    def create_input(t_input, image_value_range):
        """Create input tensor."""
        def forget_xy(t):
            """Forget sizes of dimensions [1, 2] of a 4d tensor."""
            zero = tf.identity(0)
            return t[:, zero:, zero:, :]

        t_prep_input = t_input
        if len(t_prep_input.shape) == 3:
            t_prep_input = tf.expand_dims(t_prep_input, 0)
        t_prep_input = forget_xy(t_prep_input)
        lo, hi = image_value_range
        t_prep_input = lo + t_prep_input * (hi-lo)
        return t_input, t_prep_input

    @staticmethod
    def get_bottleneck_tensors():
        """Add Inception bottlenecks and their pre-Relu versions to endpoints dict."""
        graph = tf.compat.v1.get_default_graph()
        bn_endpoints = {}
        for op in graph.get_operations():
            # change this below string to change which layers are considered bottlenecks
            # use 'ConcatV2' for InceptionV3
            # use 'MaxPool' for VGG16 (for example)
            if 'ConcatV2' in op.type:
                name = op.name.split('/')[0]
                bn_endpoints[name] = op.outputs[0]
            
        return bn_endpoints
      
def get_model(model_name):
    
    if model_name == 'inceptionv3':
        return InceptionV3(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
    elif model_name == 'resnet18':
        return models.resnet18(pretrained=True)
      

def run_tcav_custom(target, concept, dataset, bottleneck, model_name, working_dir, num_random_exp):

    # the name of the parent directory that results are stored (only if you want to cache)
    project_name = 'tcav_test_'+str(target)
    working_dir = working_dir#"./tmp/" + user + '/' + project_name
    source_dir = "./tcav/tcav_examples/image_models/"+dataset+"/data/"
    
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir + '/activations/'
    
    # where CAVs are stored. 
    # You can say None if you don't wish to store any.
    cav_dir = working_dir + '/cavs/'
    
     
          
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)
    
    # this is a regularizer penalty parameter for linear classifier to get CAVs. 
    alphas = [0.1]   
    

    
    # Create TensorFlow session.
    sess = utils.create_session(interactive=True)
    #sess = K.get_session()

    # Your code for training and creating a model here. In this example, I saved the model previously
    # using model.save and am loading it again in keras here using load_model.
    #model = load_model('./experiment_models/model.h5')
    model = get_model(model_name)
    inceptionv3 = get_model('inceptionv3')
      
    # input is the first tensor, logit and prediction is the final tensor.
    # note that in keras, these arguments should be exactly the same for other models (e.g VGG16), except for the model name
    endpoints_v3 = dict(
        input=model.inputs[0].name,
        input_tensor=model.inputs[0],
        logit=model.outputs[0].name,
        prediction=model.outputs[0].name,
        prediction_tensor=model.outputs[0],
    )
    
    # endpoints_v3 should look like this
    #endpoints_v3 = {
    #    'input': 'input_1:0',
    #    'input_tensor': <tf.Tensor 'input_1:0' shape=(?, 224, 224, 3) dtype=float32>,
    #    'logit': 'dense_2/Softmax:0',
    #    'prediction': 'dense_2/Softmax:0',
    #    'prediction_tensor': <tf.Tensor 'dense_2/Softmax:0' shape=(?, 2) dtype=float32>
    #}
    
    
    
    
    # instance of model wrapper, change the labels and other arguments to whatever you need
    LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"

    mymodel = CustomPublicImageModelWrapper(sess, 
            LABEL_PATH, [299, 299, 3], endpoints_v3, 
            'InceptionV3_public', (-1, 1))
    
    #plot_model(model, to_file='inceptionV3.png', show_shapes=True, show_layer_names=True)

    
    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=200)
    
    mytcav = tcav.TCAV(sess,
            target, concept, bottleneck,
            act_generator, alphas,
            cav_dir=cav_dir,
            num_random_exp=11)
    
    print ('This may take a while... Go get coffee!')
    results = mytcav.run(run_parallel=False)
    print ('done!',results,'\n+++++++++++++++')
    utils_plot.plot_results(results, 0, working_dir, num_random_exp=num_random_exp)
    #result.append(results)
    
    
    with open(working_dir+'/tcav_res_'+target+'.pkl', 'wb') as fp:
        pickle.dump(results, fp)
        print('dictionary saved successfully to file')
        
    
    sess.close()
