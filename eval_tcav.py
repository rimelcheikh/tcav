# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:57:14 2023

@author: rielcheikh
"""

import sys, os
import tensorflow as tf
import numpy as np

from tcav.run_tcav import run_tcav
from tcav.tcav.model_custom import run_tcav_custom 
import tcav.tcav.utils as utils
import tcav.tcav.utils_plot as utils_plot
import pickle
from os.path import exists


#import tcav.repro_check as repro_check

import pdb
#pdb.set_trace()


#eval_save_dir = "./tmp/" + user + '/' + project_name

result = {}

tf.compat.v1.enable_eager_execution() 

"""targets = ['dalmatian', 'zebra','lion', 'tiger', 'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'persian+cat', 'horse', 
           'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'hippopotamus', 'leopard', 'moose', 'spider+monkey', 
           'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 
           'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'giant+panda', 'deer', 'bobcat', 'pig', 
           'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow', 'dolphin']"""



#'plant','papyrus','paper','concrete','soapsuds','chess','crackle','rock','crystal','common marigold',
#'marigold','double knit','knitwear', 'lace','rattlesnake master','track'
#['striped','dotted','blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crystalline', 'frilly', 'knitted', 'lacelike', 'scaly','veined']




#bottleneck is the last layer before flatten operation
#inceptionv3 ['mixed10']  #['mixed3a']#,'mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']  
#bottleneck = ['mixed10']

#ResNet101 ['conv5_block3_add_1,conv5_block3_add']
#bottleneck = ['conv5_block3_add_1']

#vgg_16 ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool','block1_pool_1','block2_pool_1','block3_pool_1','block4_pool_1','block5_pool_1']
#bottleneck = ['block5_pool_1']






def run_eval_tcav(targets, concepts, dataset, model_name, bottleneck, num_random_exp, alphas, model_cav, res_dir,data_dir):

    class_tcav_score = {}
    
    for target in targets:
    
        #try:
        project_name = 'tcav_test_'+str(target)
        working_dir = res_dir + project_name
    
        if not exists(working_dir):
            os.makedirs(working_dir)
        
        if not exists(working_dir+'/tcav_res_'+target+'.pkl'):
            run_tcav_custom(target, concepts, dataset, bottleneck, model_name, working_dir, data_dir, num_random_exp, alphas, model_cav)
        else:
            print(working_dir+'/tcav_res_'+target+'.pkl'+'  already computed')
            
        """with open(working_dir+'/tcav_res_'+target+'.pkl', 'rb') as fp:
            tcav_results = pickle.load(fp)
            class_tcav_score[target] = {}
            #search for random batch that gives max TCAV score (i_up) for concept
            for i in range(len(tcav_results)):
                if not tcav_results[i]['cav_concept'][:6] == 'random':
                    res = tcav_results[i]                    
                    try : 
                        if(class_tcav_score[target][tcav_results[i]['cav_concept']] < res['i_up']):
                            class_tcav_score[target][tcav_results[i]['cav_concept']] = res['i_up']
                    except:
                        class_tcav_score[target][tcav_results[i]['cav_concept']] = res['i_up']"""
                        
            #i_expl = class_tcav_score.index(np.max(class_tcav_score))
            #instances_logit_scores =  np.squeeze(tcav_results[i_expl]['logits'])[:,80]
            #sp_coeff[target] = utils.spearmans_rank(instances_tcav_scores, instances_logit_scores)[0][1]
     
        """except Exception as error:
            print('Error for target ', target, ':', error)"""
            
    
"""    #computing spearman when concept is fixed
    sp_coeff_targets = {}
    for c in concepts:
        rationale_dict = {}
        rationale = []
        tcav = []
        for t in targets:
            #TODO
            rationale_dict[t] = get_asso_strength(t,c.split('-')[0])
            rationale.append(get_asso_strength(t,c.split('-')[0]))
        
            tcav.append(class_tcav_score[t][c])
            
        sp_coeff_targets[c] = utils.spearmans_rank(rationale, tcav)[0][1]
    
    
    with open("./tcav/tcav_results/" + user +'/result_fixed_concept.pkl', 'wb') as fp:
        pickle.dump(sp_coeff_targets, fp)
        print('Spearman coeff on fixed concepts saved successfully to file')
    
    
    with open("./tcav/tcav_results/" + user +'/result_fixed_concept.pkl', 'rb') as fp:
        print('Results:', pickle.load(fp))
    
    
    
    
    
    
    #computing spearman when (target,concept) are fixed
    sp_coeff = {}
    tcav_dict = {}
    rationale_dict = {}
    for target in targets:
        rationale = {}
        for c in concepts:
            #rationale_dict[c] = get_asso_strength(target,c.split('-')[0])
            rationale[c] = (get_asso_strength(target,c.split('-')[0]))
        
        tcav_dict[target] = class_tcav_score[target]
        rationale_dict[target] = rationale
        
    r, s = {}, {}
    r_vect, s_vect = [], []
    for t in targets:
        for c in concepts:
            r[t,c] = rationale_dict[t][c]
            s[t,c] = tcav_dict[t][c]
            r_vect.append(rationale_dict[t][c])
            s_vect.append(tcav_dict[t][c])
    sp_coeff = utils.spearmans_rank(r_vect, s_vect)[0][1]
    
    with open("./tcav/tcav_results/" + user +'/result_fixed_target_and_concept.pkl', 'wb') as fp:
        pickle.dump(sp_coeff, fp)
        print('Spearman coeff on fixed targets saved successfully to file')
    
    
    with open("./tcav/tcav_results/" + user +'/result_fixed_target_and_concept.pkl', 'rb') as fp:
        print('Results:', pickle.load(fp))
    
    
    
    
    #computing spearman when target is fixed
    sp_coeff_concepts = {}
    for target in targets:
        rationale_dict = {}
        rationale = []
        for c in concepts:
            rationale_dict[c] = get_asso_strength(target,c.split('-')[0])
            rationale.append(get_asso_strength(target,c.split('-')[0]))
        
        tcav_dict = class_tcav_score[target]
        tcav = []
        for v in class_tcav_score[target].keys():
            tcav.append(class_tcav_score[target][v])
        
        sp_coeff_concepts[target] = utils.spearmans_rank(rationale, tcav)[0][1]
    
    
    with open("./tcav/tcav_results/" + user +'/result_fixed_target.pkl', 'wb') as fp:
        pickle.dump(sp_coeff_concepts, fp)
        print('Spearman coeff on fixed targets saved successfully to file') 
    
    
    with open("./tcav/tcav_results/" + user +'/result_fixed_target.pkl', 'rb') as fp:
        print('Results:', pickle.load(fp))"""











