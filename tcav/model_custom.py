import keras.applications as models
from keras.models import Model, load_model
import keras.backend as K
import pickle

import tcav.tcav.model as tcav_model
import tcav.tcav.tcav as tcav
import tcav.tcav.utils as utils
import tcav.tcav.activation_generator as act_gen
import tensorflow as tf
import tcav.tcav.utils_plot as utils_plot
from keras.utils import plot_model




# Modified version of PublicImageModelWrapper in TCAV's models.py
# This class takes a session which contains the already loaded graph.
# This model also assumes softmax is used with categorical crossentropy.
class CustomPublicImageModelWrapper(tcav_model.ImageModelWrapper):
    def __init__(self, sess, labels, image_shape,
                endpoints_dict, name, image_value_range, model):
        
        super(self.__class__, self).__init__(image_shape)
        
        self.sess = sess
        self.labels = tf.io.gfile.GFile(labels).read().splitlines()
        self.model_name = name
        self.image_value_range = image_value_range
        self.model = model

        # get endpoint tensors
        self.ends = {'input': endpoints_dict['input_tensor'], 'prediction': endpoints_dict['prediction_tensor']}
        
        
        #self.bottlenecks_tensors = self.get_bottleneck_tensors(self.model_name)
        self.get_bottleneck_tensors_2()
        
        
        
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
        self._make_gradient_tensors(self.bottlenecks_tensors)

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)


    def get_bottleneck_tensors_2(self):
      self.bottlenecks_tensors = {}
      layers = self.model.layers
      for layer in layers:
        if 'input' not in layer.name and 'activation' not in layer.name and 'batch_normalization' not in layer.name and 'conv2d' not in layer.name:
          self.bottlenecks_tensors[layer.name] = layer.output

    def get_inputs_and_outputs_and_ends(self):
      self.ends['input'] = self.model.inputs[0]
      self.ends['prediction'] = self.model.outputs[0]

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

    """@staticmethod 
    def get_bottleneck_tensors(model_name):
        Add bottlenecks and their pre-Relu versions to endpoints dict.
        if model_name == 'inceptionv3':
            bn_name = 'ConcatV2'
        elif model_name.split('_')[0] == 'resnet':
            bn_name = 'AddV2'
        elif model_name.split('_')[0] == 'vgg':
            bn_name = 'MaxPool'
            
        graph = tf.compat.v1.get_default_graph()
        bn_endpoints = {}
        for op in graph.get_operations():
            for op in graph.get_operations():
                if op.type not in z:
                    z.append(op.type)
            "for op in graph.get_operations():
                if op.type in ['Placeholder']:
                    print(op.name)
            
            if bn_name in op.type:
                name = op.name.split('/')[0]
                bn_endpoints[name] = op.outputs[0]
            
        return bn_endpoints"""
      
def get_model(model_name):
    
    if model_name == 'inceptionv3':
        return models.InceptionV3(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        ), [299, 299, 3]
    
    elif model_name == 'resnet_101':
        return models.ResNet101(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        ), [224, 224, 3]
    
    elif model_name == 'vgg_16':
        return models.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        ), [224, 224, 3]
    
    
      

def run_tcav_custom(target, concept, dataset, bottleneck, model_name, working_dir, data_dir, num_random_exp, alphas, model_cav):

    # the name of the parent directory that results are stored (only if you want to cache)
    project_name = 'tcav_test_'+str(target)
    working_dir = working_dir#"./tmp/" + user + '/' + project_name
    source_dir = "./data/"#+dataset
    
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir.rsplit('/',1)[0]+ '/activations/'
    
    # where CAVs are stored. 
    # You can say None if you don't wish to store any.
    cav_dir = working_dir.rsplit('/',1)[0]+ '/cavs/'
    
     
          
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)
      
    

    
    # Create TensorFlow session.
    sess = utils.create_session(interactive=True)
    #sess = K.get_session()

    # Your code for training and creating a model here. In this example, I saved the model previously
    # using model.save and am loading it again in keras here using load_model.
    #model = load_model('./experiment_models/model.h5')
    model = get_model(model_name)[0]
      
    # input is the first tensor, logit and prediction is the final tensor.
    # note that in keras, these arguments should be exactly the same for other models (e.g VGG16), except for the model name
    endpoints = dict(
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
    
    
    
    
    #TODO
    # instance of model wrapper, change the labels and other arguments to whatever you need
    LABEL_PATH = data_dir+dataset + "/imagenet_comp_graph_label_strings.txt"

    mymodel = CustomPublicImageModelWrapper(sess, 
            LABEL_PATH, get_model(model_name)[1], endpoints, 
            model_name, (-1, 1), model)
    
    #plot_model(model, to_file=model_name+'.png', show_shapes=True, show_layer_names=True)

    
    act_generator = act_gen.ImageActivationGenerator(mymodel, data_dir, activation_dir, max_examples=200)
    
    mytcav = tcav.TCAV(sess,
            target, concept, bottleneck,
            act_generator, alphas,
            cav_dir=cav_dir,
            num_random_exp=num_random_exp,
            model_cav=model_cav)
    
    print ('This may take a while... Go get coffee!')
    results = mytcav.run(run_parallel=False)
    print ('done!',results,'\n+++++++++++++++')
    utils_plot.plot_results(results, 0, working_dir, num_random_exp=num_random_exp)
    #result.append(results)
    
    
    with open(working_dir+'/tcav_res_'+target+'.pkl', 'wb') as fp:
        pickle.dump(results, fp)
        print('dictionary saved successfully to file')
        
    
    #sess.close()
